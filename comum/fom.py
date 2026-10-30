# ===== Bloco 7: modelo fracionário (FOM) e genoma do NSGA-II =====
# Módulo compartilhado: só define o modelo e o genoma. As verificações rápidas que estavam
# no fim do bloco agora ficam na etapa 04_fom_verificacao.py.
# Implementa as equações da seção Mathematical Modeling:
#   p[n] = |x[n]|^2
#   D^alpha s_j = tau_j^(-alpha) (p - s_j), discretizada por Grünwald-Letnikov truncada em L amostras
#   y[n] = sum_k sum_m [c0_km + sum_j cj_km s_j[n]] x[n-m] |x[n-m]|^(k-1)

import time                                           # medir o tempo de cada avaliação
import numpy as np                                    # álgebra linear e vetores complexos
from comum.dados import x, y, CORTES, nmse_db         # registro, divisão e métrica (módulo de dados)

try:                                                  # numba compila a recursão para código de máquina
    from numba import njit                            # decorador de compilação
except ImportError:                                   # se o numba não estiver instalado...
    def njit(*a, **k):                                # ...define um decorador que não faz nada
        return a[0] if a and callable(a[0]) else (lambda f: f)   # devolve a própria função, sem compilar
    print("numba ausente: a recursão vai rodar em Python puro (muito mais lenta)")   # aviso

# ---------------- Espaço de busca (limites do genoma) ----------------
J_MAX = 3                                             # número máximo de estados lentos
K_MIN, K_MAX = 1, 9                                   # ordem não linear do polinômio rápido
M_MIN, M_MAX = 0, 20                                  # memória curta do polinômio (amostras)
LOG_L_MIN, LOG_L_MAX = 1.0, 3.0                       # log10 do truncamento L: de 10 a 1 000 amostras
ALFA_MIN, ALFA_MAX = 0.05, 1.0                         # ordem fracionária comum a todos os estados
LOG_TAU_MIN, LOG_TAU_MAX = 0.0, 4.0                   # log10 da constante de tempo: de 1 a 10 000 amostras
L_MAX = int(round(10 ** LOG_L_MAX))                   # maior L possível (1 000 amostras)
AQUEC = L_MAX                                         # amostras iniciais do treino fora do ajuste, iguais para todos
RCOND_FOM = 1e-4                                      # mesmo corte do SVD truncado usado nos GMPs
BLOCO = 19_662                                        # tamanho dos pedaços da QR incremental (só controla memória)

# Genoma: vetor real de tamanho fixo; os genes inteiros são arredondados na decodificação
NOMES_GENES = ["J", "K", "M", "log10_L", "alfa"] + [f"log10_tau{j+1}" for j in range(J_MAX)]   # rótulo de cada gene
LIM_INF = np.array([1, K_MIN, M_MIN, LOG_L_MIN, ALFA_MIN] + [LOG_TAU_MIN] * J_MAX, dtype=float)  # limite inferior
LIM_SUP = np.array([J_MAX, K_MAX, M_MAX, LOG_L_MAX, ALFA_MAX] + [LOG_TAU_MAX] * J_MAX, dtype=float)  # limite superior
N_GENES = len(NOMES_GENES)                            # 5 + J_MAX = 8 genes


def decodificar(g):                                   # transforma o vetor de genes nos parâmetros do modelo
    g = np.clip(np.asarray(g, dtype=float), LIM_INF, LIM_SUP)   # garante que todos os genes estão dentro dos limites
    J = int(round(g[0]))                              # número de estados lentos (inteiro)
    K = int(round(g[1]))                              # ordem não linear (inteiro)
    M = int(round(g[2]))                              # memória curta (inteiro)
    L = int(round(10 ** g[3]))                        # truncamento em amostras (inteiro, busca em escala log)
    alfa = float(g[4])                                # ordem fracionária (real)
    taus = np.sort(10 ** g[5:5 + J])                  # constantes de tempo dos J estados ativos, em ordem crescente
    return {"J": J, "K": K, "M": M, "L": L, "alfa": alfa, "taus": taus}   # dicionário com a estrutura


def genoma_aleatorio(rng):                            # sorteia um genoma uniforme dentro dos limites
    return LIM_INF + rng.random(N_GENES) * (LIM_SUP - LIM_INF)   # valor uniforme entre os limites de cada gene


# ---------------- Estados lentos: Grünwald-Letnikov truncada ----------------
def pesos_gl(alfa, L):                                # pesos w_0..w_L da definição de Grünwald-Letnikov
    w = np.empty(L + 1)                               # vetor de pesos
    w[0] = 1.0                                        # w_0 = 1
    for i in range(1, L + 1):                         # recursão w_i = (1 - (alfa+1)/i) w_{i-1}
        w[i] = (1.0 - (alfa + 1.0) / i) * w[i - 1]  # peso i a partir do anterior
    return w                                          # pesos prontos


@njit(cache=True)                                     # compila a função na primeira chamada
def _recursao_gl(p, w, a, s_ini):                     # s[n] = (a p[n] - sum_{i=1}^{L} w_i s[n-i]) / (1 + a)
    L = len(w) - 1                                    # comprimento do truncamento
    ext = np.empty(L + len(p))                        # vetor com L amostras de histórico antes de n = 0
    ext[:L] = s_ini                                   # histórico inicial constante
    for n in range(len(p)):                           # percorre o registro amostra a amostra
        acc = 0.0                                     # acumulador da soma de convolução
        for i in range(1, L + 1):                     # soma sobre as L amostras anteriores
            acc += w[i] * ext[L + n - i]              # acumula w_i s[n-i]
        ext[L + n] = (a * p[n] - acc) / (1.0 + a)     # estado atual (forma implícita isolada)
    return ext[L:]                                    # devolve só o trecho do registro


def estados_lentos(p, alfa, taus, L):                 # calcula os J estados s_j[n] sobre o registro inteiro
    w = pesos_gl(alfa, L)                             # pesos comuns a todos os estados (mesmo alfa)
    p0 = p[:L].mean()                                 # potência média do início do registro
    S = np.empty((len(taus), len(p)))                 # uma linha por estado
    for j, tau in enumerate(taus):                    # percorre os estados
        a = tau ** (-alfa)                            # coeficiente tau^(-alfa)
        s_ini = p0 * a / (a + w.sum())                # regime permanente da recursão truncada para p0
        S[j] = _recursao_gl(p, w, a, s_ini)           # estado ao longo de todo o registro
    return S                                          # matriz J x N com os estados


# ---------------- Regressores do FOM ----------------
def base_mp(xs, K, M):                                # colunas x[n-m] |x[n-m]|^(k-1), k = 1..K, m = 0..M
    cols = np.empty((len(xs), K * (M + 1)), dtype=complex)   # matriz do polinômio de memória
    j = 0                                             # índice da coluna
    for k in range(1, K + 1):                         # ordem não linear
        for m in range(M + 1):                        # atraso
            xm = np.zeros_like(xs)                    # x[n-m] com zeros antes do início do trecho
            xm[m:] = xs[:len(xs) - m] if m > 0 else xs   # desloca o trecho m amostras
            cols[:, j] = xm * np.abs(xm) ** (k - 1)   # termo do polinômio de memória
            j += 1                                    # próxima coluna
    return cols                                       # matriz pronta


def base_fom(a, b, K, M, S):                          # regressores das amostras a..b-1 com o passado real do registro
    ini = max(0, a - M)                               # inclui M amostras de histórico, se existirem
    B = base_mp(x[ini:b], K, M)[a - ini:]             # polinômio rápido sem as linhas de aquecimento
    blocos = [B] + [B * S[j, a:b, None] for j in range(S.shape[0])]   # [c0 | c1 s1 | ... | cJ sJ]
    return np.hstack(blocos)                          # matriz (b-a) x K(M+1)(J+1)


def ajuste_truncado(gerador, n_col):                  # mínimos quadrados com colunas normalizadas e SVD truncado
    R = np.zeros((0, n_col + 1), dtype=complex)       # fator R acumulado de [Phi | y]
    for P, yb in gerador:                             # pedaços da base e da saída
        R = np.linalg.qr(np.vstack([R, np.hstack([P, yb[:, None]])]), mode="r")   # QR incremental
    Ra, bq = R[:n_col, :n_col], R[:n_col, n_col]      # regressores e projeção da saída
    normas = np.linalg.norm(R[:, :n_col], axis=0)     # norma L2 de cada coluna
    normas[normas == 0] = 1.0                         # evita divisão por zero em colunas nulas
    U, sv, Vh = np.linalg.svd(Ra / normas)            # valores singulares da base normalizada
    manter = sv > sv[0] * RCOND_FOM                   # direções acima do corte
    c = Vh[manter].conj().T @ ((U[:, manter].conj().T @ bq) / sv[manter])   # solução truncada
    cond = sv[0] / sv[-1] if sv[-1] > 0 else np.inf   # número de condição da base normalizada
    return c / normas, int(manter.sum()), cond        # coeficientes na escala original, posto e condição


# ---------------- Avaliação de um genoma (o que o NSGA-II vai chamar) ----------------
P_X = np.abs(x) ** 2                                  # potência instantânea p[n] do registro inteiro (calculada uma vez)


def avaliar_fom(g, conjunto="val"):                   # devolve os três objetivos e informações do ajuste
    t0 = time.time()                                  # início da contagem de tempo
    th = decodificar(g)                               # estrutura proposta pelo genoma
    J, K, M, L = th["J"], th["K"], th["M"], th["L"]   # parâmetros inteiros
    S = estados_lentos(P_X, th["alfa"], th["taus"], L)   # estados lentos sobre o registro inteiro
    n_col = K * (M + 1) * (J + 1)                     # número de coeficientes complexos
    i0, i1 = CORTES["treino"]                         # limites do treino
    pedacos = [(a, min(a + BLOCO, i1)) for a in range(i0 + AQUEC, i1, BLOCO)]   # treino sem as AQUEC primeiras amostras
    c, posto, cond = ajuste_truncado(((base_fom(a, b, K, M, S), y[a:b]) for a, b in pedacos), n_col)   # ajuste no treino
    j0, j1 = CORTES[conjunto]                         # limites do conjunto avaliado
    y_hat = np.concatenate([base_fom(a, min(a + BLOCO, j1), K, M, S) @ c for a in range(j0, j1, BLOCO)])  # previsão contínua
    nmse = nmse_db(y[j0:j1], y_hat)                   # objetivo 1: NMSE em dB
    n_par = 2 * n_col + (1 + J)                       # objetivo 2: parâmetros reais (coeficientes + alfa + taus)
    log_cond = np.log10(cond)                         # objetivo 3: log10 do número de condição
    if not np.isfinite(nmse):                         # proteção contra ajustes que divergem
        nmse = 100.0                                  # valor ruim que o NSGA-II vai descartar
    objetivos = np.array([nmse, n_par, log_cond])     # vetor de objetivos, todos a minimizar
    info = {**th, "n_coef": n_col, "posto": posto, "tempo_s": time.time() - t0}   # detalhes para registro
    return objetivos, info                            # objetivos para o NSGA-II e detalhes para registro
