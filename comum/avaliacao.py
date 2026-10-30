# ===== Módulo compartilhado: ajuste dos modelos e previsão em qualquer trecho do registro =====
# Usado pela etapa 12 (EVM e ACPR). Repete a construção das etapas 02, 03 e 11, mas devolve uma
# função prever(a, b) que gera a saída do modelo para as amostras a..b-1 de qualquer parte do
# registro, inclusive o treino. Os coeficientes são sempre ajustados só no treino.

import numpy as np                                    # vetores
from comum.dados import x, y, CORTES                  # registro e divisão
from comum.gmp import SEG, desloca, segmentos         # segmento do benchmark e deslocamentos
from comum.fom import decodificar, estados_lentos, base_fom, ajuste_truncado, P_X, AQUEC, BLOCO   # FOM

RCOND = 1e-4                                          # corte do SVD truncado
CFG_BENCH = dict(Ka=5, La=30, Kb=4, Lb=30, Mb=5, Kc=4, Lc=30, Mc=5)   # GMP de benchmark (1 350 coef.)
CFG_CAUSAL = dict(Ka=5, La=30, Kb=4, Lb=30, Mb=5, Kc=0, Lc=0, Mc=0)   # GMP causal (750 coef.)
MEM_CAUSAL = 29 + 5                                   # passado usado pelo GMP causal


def base_gmp_geral(s, Ka, La, Kb, Lb, Mb, Kc, Lc, Mc):   # regressores do GMP para um trecho s
    cols = []                                         # colunas da base
    for k in range(Ka):                               # alinhados x[n-q] |x[n-q]|^k
        for q in range(La):
            xq = desloca(s, q)
            cols.append(xq * np.abs(xq) ** k)
    for k in range(1, Kb + 1):                        # atrasados x[n-q] |x[n-q-m]|^k
        for q in range(Lb):
            for m in range(1, Mb + 1):
                cols.append(desloca(s, q) * np.abs(desloca(s, q + m)) ** k)
    for k in range(1, Kc + 1):                        # adiantados x[n-q] |x[n-q+m]|^k
        for q in range(Lc):
            for m in range(1, Mc + 1):
                cols.append(desloca(s, q) * np.abs(desloca(s, q - m)) ** k)
    return np.column_stack(cols)                      # matriz (amostras) x coeficientes


def ajuste_qr(pedacos, n_col):                        # mínimos quadrados truncados com colunas normalizadas
    R = np.zeros((0, n_col + 1), dtype=complex)       # fator R acumulado de [Phi | y]
    for P, yb in pedacos:
        R = np.linalg.qr(np.vstack([R, np.hstack([P, yb[:, None]])]), mode="r")   # QR incremental
    Ra, bq = R[:n_col, :n_col], R[:n_col, n_col]
    normas = np.linalg.norm(R[:, :n_col], axis=0)
    U, sv, Vh = np.linalg.svd(Ra / normas)
    manter = sv > sv[0] * RCOND
    c = Vh[manter].conj().T @ ((U[:, manter].conj().T @ bq) / sv[manter])
    return c / normas                                 # coeficientes na escala original


def _qual_conjunto(n):                                # a que conjunto pertence a amostra n
    for nome, (a, b) in CORTES.items():
        if a <= n < b:
            return nome, a
    raise ValueError(n)


def modelo_gmp_benchmark():                           # protocolo segmentado (zeros nas bordas de cada segmento)
    i0, i1 = CORTES["treino"]
    n_col = 5 * 30 + 4 * 30 * 5 * 2                   # 1 350 coeficientes
    c = ajuste_qr(((base_gmp_geral(x[i0 + a:i0 + b], **CFG_BENCH), y[i0 + a:i0 + b])
                   for a, b in segmentos(i1 - i0)), n_col)

    def prever(a, b):                                 # respeita os segmentos de cada conjunto, como o benchmark
        saida = np.empty(b - a, dtype=complex)
        n = a
        while n < b:
            nome, ini = _qual_conjunto(n)             # conjunto e início dele
            fim_conj = CORTES[nome][1]
            s0 = ini + ((n - ini) // SEG) * SEG       # início do segmento que contém n
            s1 = min(s0 + SEG, fim_conj)              # fim do segmento
            Ys = base_gmp_geral(x[s0:s1], **CFG_BENCH) @ c   # previsão do segmento inteiro
            fim = min(b, s1)
            saida[n - a:fim - a] = Ys[n - s0:fim - s0]
            n = fim
        return saida
    return prever


def modelo_gmp_causal():                              # protocolo contínuo (passado real do registro)
    n_col = 5 * 30 + 4 * 30 * 5                       # 750 coeficientes

    def base(a, b):
        ini = max(0, a - MEM_CAUSAL)
        return base_gmp_geral(x[ini:b], **CFG_CAUSAL)[a - ini:]
    i0, i1 = CORTES["treino"]
    c = ajuste_qr(((base(a, min(a + SEG, i1)), y[a:min(a + SEG, i1)])
                   for a in range(i0 + MEM_CAUSAL, i1, SEG)), n_col)

    def prever(a, b):
        return np.concatenate([base(n, min(n + SEG, b)) @ c for n in range(a, b, SEG)])
    return prever


def modelo_fom(g8):                                   # FOM ou baseline a partir do genoma de 8 genes
    th = decodificar(g8)
    J, K, M, L = th["J"], th["K"], th["M"], th["L"]
    S = estados_lentos(P_X, th["alfa"], th["taus"], L)   # estados no registro inteiro
    n_col = K * (M + 1) * (J + 1)
    i0, i1 = CORTES["treino"]
    c, _, _ = ajuste_truncado(((base_fom(a, min(a + BLOCO, i1), K, M, S), y[a:min(a + BLOCO, i1)])
                               for a in range(i0 + AQUEC, i1, BLOCO)), n_col)   # mesmas amostras da busca

    def prever(a, b):
        return np.concatenate([base_fom(n, min(n + BLOCO, b), K, M, S) @ c for n in range(a, b, BLOCO)])
    return prever