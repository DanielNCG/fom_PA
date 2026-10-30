# ===== Etapa 02 (bloco 5): GMP de benchmark do OpenDPD para o APA_200MHz =====
# Uso: python 02_gmp_benchmark.py
# Grava resultados/gmp_benchmark_apa.npz. Pico de memória perto de 2 GB.

import numpy as np                                    # álgebra linear e vetores complexos
from comum.dados import x, y, CORTES, nmse_db, PASTA_RESULTADOS   # registro, divisão, métrica e pasta de saída
from comum.gmp import SEG, desloca, segmentos         # segmento do benchmark e funções de deslocamento

# Configuração publicada em benchmark/results/benchmark_report_results.json
Ka, La = 5, 30                                        # termos alinhados: potências 0..4 de |x|, atrasos 0..29
Kb, Lb, Mb = 4, 30, 5                                 # envoltória atrasada: potências 1..4, atrasos 0..29, defasagens 1..5
Kc, Lc, Mc = 4, 30, 5                                 # envoltória adiantada: potências 1..4, atrasos 0..29, avanços 1..5
RCOND = 1e-4                                          # corte do SVD truncado: descarta valores singulares < 1e-4 do maior
N_COEF = Ka * La + Kb * Lb * Mb + Kc * Lc * Mc        # total de coeficientes complexos: 150 + 600 + 600 = 1 350


def base_gmp(s):                                      # monta a matriz de regressores do GMP para um trecho s
    Phi = np.empty((len(s), N_COEF), dtype=complex)   # reserva uma coluna por coeficiente
    j = 0                                             # índice da próxima coluna a preencher
    for k in range(Ka):                               # potências 0..Ka-1 da envoltória
        for q in range(La):                           # atrasos 0..La-1 do sinal
            xq = desloca(s, q)                        # x[n - q]
            Phi[:, j] = xq * np.abs(xq) ** k          # termo alinhado: x[n-q] |x[n-q]|^k
            j += 1                                    # avança para a próxima coluna
    for k in range(1, Kb + 1):                        # potências 1..Kb da envoltória
        for q in range(Lb):                           # atrasos 0..Lb-1 do sinal
            for m in range(1, Mb + 1):                # defasagem 1..Mb da envoltória em relação ao sinal
                Phi[:, j] = desloca(s, q) * np.abs(desloca(s, q + m)) ** k   # x[n-q] |x[n-q-m]|^k
                j += 1                                # avança para a próxima coluna
    for k in range(1, Kc + 1):                        # potências 1..Kc da envoltória
        for q in range(Lc):                           # atrasos 0..Lc-1 do sinal
            for m in range(1, Mc + 1):                # avanço 1..Mc da envoltória (amostras futuras)
                Phi[:, j] = desloca(s, q) * np.abs(desloca(s, q - m)) ** k   # x[n-q] |x[n-q+m]|^k
                j += 1                                # avança para a próxima coluna
    return Phi                                        # matriz (amostras do trecho) x 1 350


# --- Ajuste no conjunto de treino ---
i0, i1 = CORTES["treino"]                             # limites do treino no registro concatenado
x_tr, y_tr = x[i0:i1], y[i0:i1]                       # entrada e saída de treino
R = np.zeros((0, N_COEF + 1), dtype=complex)          # fator R acumulado da QR de [Phi | y] (começa vazio)
for a, b in segmentos(len(x_tr)):                     # percorre o treino em segmentos (atrasos zerados em cada um)
    bloco = np.hstack([base_gmp(x_tr[a:b]), y_tr[a:b, None]])   # regressores do segmento com a saída como última coluna
    R = np.linalg.qr(np.vstack([R, bloco]), mode="r") # QR incremental: guarda só R, sem a matriz inteira na memória
Ra = R[:N_COEF, :N_COEF]                              # parte de R que corresponde aos regressores
bq = R[:N_COEF, N_COEF]                               # Q^H y: projeção da saída na base ortonormal
normas = np.linalg.norm(R[:, :N_COEF], axis=0)        # norma L2 de cada coluna de Phi (igual à norma da coluna de R)
U, sv, Vh = np.linalg.svd(Ra / normas)                # SVD das colunas normalizadas, como faz o benchmark
manter = sv > sv[0] * RCOND                           # direções singulares acima do corte
posto = int(manter.sum())                             # número de direções mantidas (posto efetivo)
c_norm = Vh[manter].conj().T @ ((U[:, manter].conj().T @ bq) / sv[manter])   # solução de mínimos quadrados truncada
c_gmp = c_norm / normas                               # desfaz a normalização: coeficientes na escala original

# --- Avaliação em validação e teste ---
def prever_gmp(xb):                                   # saída prevista para um bloco, segmento a segmento
    return np.concatenate([base_gmp(xb[a:b]) @ c_gmp for a, b in segmentos(len(xb))])

nmse = {}                                             # guarda o NMSE de cada conjunto
for nome in ("val", "teste"):                         # avalia validação e teste
    j0, j1 = CORTES[nome]                             # limites do conjunto
    nmse[nome] = nmse_db(y[j0:j1], prever_gmp(x[j0:j1]))   # NMSE em dB entre saída medida e prevista

np.savez(PASTA_RESULTADOS / "gmp_benchmark_apa.npz", coef=c_gmp, Ka=Ka, La=La, Kb=Kb, Lb=Lb, Mb=Mb,
         Kc=Kc, Lc=Lc, Mc=Mc, rcond=RCOND, seg=SEG)   # salva coeficientes e configuração para reutilizar

print(f"coeficientes complexos: {N_COEF}  ({2 * N_COEF} parâmetros reais)")       # tamanho nominal do modelo
print(f"posto efetivo após o corte: {posto} de {N_COEF}  (benchmark: 650)")         # quantas direções sobreviveram
print(f"maior valor singular: {sv[0]:.4f}  (benchmark: 12.8312)")                   # escala do problema normalizado
print(f"norma L2 dos coeficientes: {np.linalg.norm(c_gmp):.4f}  (benchmark: 7.8917)")  # tamanho da solução
print(f"NMSE validação: {nmse['val']:.2f} dB  (benchmark: -38.70)")                 # erro na validação
print(f"NMSE teste:     {nmse['teste']:.2f} dB  (benchmark: -38.66)")               # erro no teste
