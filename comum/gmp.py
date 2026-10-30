# ===== Funções comuns aos dois GMPs (benchmark e causal) =====
# Vieram do bloco 5 do notebook: SEG, desloca e segmentos são usados pelas etapas 02 e 03.

import numpy as np                                    # vetores complexos
from comum.dados import spec                          # especificação do dataset

SEG = int(spec["nperseg"])                            # tamanho do segmento usado pelo benchmark: 19 662 amostras


def desloca(s, d):                                    # devolve s[n - d] com zeros fora do trecho
    out = np.zeros_like(s)                            # começa com um vetor de zeros do mesmo tamanho e tipo
    if 0 <= d < len(s):                               # atraso (d >= 0): olha para o passado
        out[d:] = s[:len(s) - d]                      # copia o sinal empurrado d posições para a frente
    elif -len(s) < d < 0:                             # avanço (d < 0): olha para o futuro
        out[:d] = s[-d:]                              # copia o sinal puxado |d| posições para trás
    return out                                        # amostras sem passado/futuro disponível ficam em zero


def segmentos(n):                                     # índices de início e fim de cada segmento de SEG amostras
    return [(i, min(i + SEG, n)) for i in range(0, n, SEG)]
