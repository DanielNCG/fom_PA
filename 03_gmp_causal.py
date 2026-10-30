# ===== Etapa 03 (bloco 6): GMP causal (sem amostras futuras) =====
# Uso: python 03_gmp_causal.py
# Mesma configuração do GMP de benchmark, sem a família de termos adiantados (Kc = 0).
# Grava resultados/gmp_causal_apa.npz.

import numpy as np                                    # álgebra linear e vetores complexos
from comum.dados import x, y, CORTES, fs, nmse_db, PASTA_RESULTADOS   # registro, divisão, f_s, métrica e pasta de saída
from comum.gmp import SEG, desloca, segmentos         # segmento do benchmark e funções de deslocamento

Ka_c, La_c = 5, 30                                    # termos alinhados: potências 0..4 de |x|, atrasos 0..29
Kb_c, Lb_c, Mb_c = 4, 30, 5                           # envoltória atrasada: potências 1..4, atrasos 0..29, defasagens 1..5
RCOND_C = 1e-4                                        # mesmo corte do SVD truncado usado no benchmark
N_CAUSAL = Ka_c * La_c + Kb_c * Lb_c * Mb_c           # coeficientes complexos: 150 + 600 = 750
MEM = (Lb_c - 1) + Mb_c                               # amostra mais antiga usada: x[n-34]; o modelo precisa de 34 amostras de passado


def base_gmp_causal(s):                               # matriz de regressores do GMP causal para um trecho s
    Phi = np.empty((len(s), N_CAUSAL), dtype=complex) # reserva uma coluna por coeficiente
    j = 0                                             # índice da próxima coluna a preencher
    for k in range(Ka_c):                             # potências 0..Ka_c-1 da envoltória
        for q in range(La_c):                         # atrasos 0..La_c-1 do sinal
            xq = desloca(s, q)                        # x[n - q] (função de comum/gmp.py)
            Phi[:, j] = xq * np.abs(xq) ** k          # termo alinhado: x[n-q] |x[n-q]|^k
            j += 1                                    # avança para a próxima coluna
    for k in range(1, Kb_c + 1):                      # potências 1..Kb_c da envoltória
        for q in range(Lb_c):                         # atrasos 0..Lb_c-1 do sinal
            for m in range(1, Mb_c + 1):              # defasagem 1..Mb_c da envoltória (só passado)
                Phi[:, j] = desloca(s, q) * np.abs(desloca(s, q + m)) ** k   # x[n-q] |x[n-q-m]|^k
                j += 1                                # avança para a próxima coluna
    return Phi                                        # matriz (amostras do trecho) x 750


def base_continua(a, b):                              # regressores das amostras a..b-1 usando o passado real do registro
    ini = max(0, a - MEM)                             # começa MEM amostras antes, se existirem
    return base_gmp_causal(x[ini:b])[a - ini:]        # monta com o histórico e descarta as linhas de aquecimento


def ajustar_qr(blocos_regressores, blocos_saida):     # mínimos quadrados truncados a partir de pedaços da base
    R = np.zeros((0, N_CAUSAL + 1), dtype=complex)    # fator R acumulado da QR de [Phi | y] (começa vazio)
    for P, yb in zip(blocos_regressores, blocos_saida):   # percorre os pedaços (gerados sob demanda)
        aum = np.hstack([P, yb[:, None]])             # junta a saída como última coluna
        R = np.linalg.qr(np.vstack([R, aum]), mode="r")   # QR incremental: guarda só o fator R
    Ra = R[:N_CAUSAL, :N_CAUSAL]                      # parte de R que corresponde aos regressores
    bq = R[:N_CAUSAL, N_CAUSAL]                       # Q^H y: projeção da saída na base ortonormal
    normas = np.linalg.norm(R[:, :N_CAUSAL], axis=0)  # norma L2 de cada coluna da base
    U, sv, Vh = np.linalg.svd(Ra / normas)            # SVD das colunas normalizadas
    manter = sv > sv[0] * RCOND_C                     # direções acima do corte de 1e-4
    c = Vh[manter].conj().T @ ((U[:, manter].conj().T @ bq) / sv[manter])   # solução truncada (escala normalizada)
    return c / normas, int(manter.sum()), sv          # coeficientes na escala original, posto e valores singulares


i0, i1 = CORTES["treino"]                             # limites do treino no registro concatenado
resultados = {}                                       # guarda os resultados de cada protocolo

# --- Protocolo 1: segmentado, igual ao benchmark (atrasos zerados no início de cada segmento) ---
pedacos = segmentos(i1 - i0)                          # segmentos de SEG amostras dentro do treino (função de comum/gmp.py)
c_seg, posto_seg, sv_seg = ajustar_qr(                # ajusta com a base de cada segmento montada isoladamente
    (base_gmp_causal(x[i0 + a:i0 + b]) for a, b in pedacos),   # regressores de cada segmento
    (y[i0 + a:i0 + b] for a, b in pedacos))           # saída medida de cada segmento
res = {}                                              # NMSE deste protocolo
for nome in ("val", "teste"):                         # avalia validação e teste
    j0, j1 = CORTES[nome]                             # limites do conjunto
    y_hat = np.concatenate([base_gmp_causal(x[j0 + a:j0 + b]) @ c_seg      # previsão segmento a segmento
                            for a, b in segmentos(j1 - j0)])
    res[nome] = nmse_db(y[j0:j1], y_hat)              # NMSE em dB
resultados["segmentado"] = (res, posto_seg)           # registra NMSE e posto

# --- Protocolo 2: contínuo (cada amostra usa o passado real, inclusive o do bloco anterior) ---
pedacos = [(a, min(a + SEG, i1)) for a in range(i0 + MEM, i1, SEG)]   # treino sem as MEM primeiras amostras (sem passado completo)
c_cont, posto_cont, sv_cont = ajustar_qr(             # ajusta com a base montada sobre o registro contínuo
    (base_continua(a, b) for a, b in pedacos),        # regressores com histórico real
    (y[a:b] for a, b in pedacos))                     # saída medida correspondente
res = {}                                              # NMSE deste protocolo
for nome in ("val", "teste"):                         # avalia validação e teste
    j0, j1 = CORTES[nome]                             # limites do conjunto
    y_hat = np.concatenate([base_continua(a, min(a + SEG, j1)) @ c_cont    # previsão com o fim do bloco anterior como passado
                            for a in range(j0, j1, SEG)])
    res[nome] = nmse_db(y[j0:j1], y_hat)              # NMSE em dB
resultados["contínuo"] = (res, posto_cont)            # registra NMSE e posto

np.savez(PASTA_RESULTADOS / "gmp_causal_apa.npz", coef_segmentado=c_seg, coef_continuo=c_cont,
         Ka=Ka_c, La=La_c, Kb=Kb_c, Lb=Lb_c, Mb=Mb_c, rcond=RCOND_C, seg=SEG)   # salva coeficientes e configuração

print(f"GMP causal: {N_CAUSAL} coeficientes complexos ({2 * N_CAUSAL} parâmetros reais), "
      f"memória de {MEM} amostras ({MEM / fs * 1e9:.1f} ns), nenhuma amostra futura")   # resumo do modelo
for protocolo, (res, posto) in resultados.items():   # uma linha por protocolo
    print(f"  {protocolo:10s}: posto {posto} de {N_CAUSAL} | "
          f"NMSE val {res['val']:.2f} dB, teste {res['teste']:.2f} dB")   # posto efetivo e NMSE
