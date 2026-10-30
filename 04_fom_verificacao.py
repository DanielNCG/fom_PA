# ===== Etapa 04 (fim do bloco 7): verificações rápidas do modelo fracionário =====
# Uso: python 04_fom_verificacao.py
# Compara a recursão com um filtro de 1a ordem e avalia três genomas de exemplo.
# Também serve para compilar a recursão do numba (fica em cache para as etapas seguintes).

import numpy as np                                    # vetores
from comum.fom import (P_X, pesos_gl, _recursao_gl,   # potência instantânea e recursão de Grünwald-Letnikov
                       NOMES_GENES, LIM_INF, LIM_SUP, genoma_aleatorio, avaliar_fom)   # genoma e avaliação

from scipy.signal import lfilter                      # filtro IIR de referência
s_gl = _recursao_gl(P_X[:5000], pesos_gl(1.0, 1), 1 / 50.0, 0.0)   # alfa = 1, L = 1, tau = 50
s_ref = lfilter([1 / 51], [1, -50 / 51], P_X[:5000])  # filtro de 1a ordem equivalente (Euler implícito)
print(f"alfa=1, L=1 vs. filtro de 1a ordem: erro máximo {np.max(np.abs(s_gl - s_ref)):.1e}")   # deve ser ~1e-16

print("genes:", NOMES_GENES)                          # ordem dos genes no vetor
print("limites inferiores:", LIM_INF)                 # limites usados pelo NSGA-II
print("limites superiores:", LIM_SUP)                 # (mesma ordem dos genes)

exemplos = {                                          # genomas escritos à mão para testar a avaliação
    "alfa=1, J=1":   [1, 5, 4, 2.0, 1.0, 2.0, 0, 0],  # um estado exponencial, tau = 100 amostras
    "alfa=0.6, J=2": [2, 5, 4, 3.0, 0.6, 1.0, 3.0, 0],  # dois estados fracionários, tau = 10 e 1 000 amostras
}                                                     # fim dos exemplos
rng = np.random.default_rng(0)                        # gerador de números aleatórios com semente fixa
exemplos["aleatório"] = genoma_aleatorio(rng)         # um genoma sorteado

for nome, g in exemplos.items():                      # avalia cada exemplo
    obj, info = avaliar_fom(g)                        # objetivos e detalhes
    print(f"{nome:14s} J={info['J']} K={info['K']} M={info['M']} L={info['L']} "          # estrutura decodificada
          f"alfa={info['alfa']:.2f} taus={np.round(info['taus'], 1)} | "                       # parâmetros dos estados
          f"NMSE val {obj[0]:.2f} dB, {int(obj[1])} parâm. reais, log10(cond) {obj[2]:.1f}, "  # os três objetivos
          f"posto {info['posto']}/{info['n_coef']} [{info['tempo_s']:.1f} s]")                 # posto efetivo e tempo
