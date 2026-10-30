# ===== Bloco 8: NSGA-II implementado do zero (NumPy) =====
# Módulo compartilhado (etapas 05 e 06). Segue os passos da subseção "NSGA-II Implementation and Validation":
#   1) inicialização, 2) ordenação por frentes + distância de aglomeração,
#   3) torneio binário com comparação aglomerada, 4) SBX + mutação polinomial,
#   5) avaliação, 6) seleção elitista (pais + filhos), 7) término após G gerações.

import time                                           # medir o tempo de cada geração
import numpy as np                                    # vetores e sorteios


# ---------------- Passo 2a: ordenação rápida por não dominância ----------------
def ordenacao_nao_dominada(F):                        # F: matriz (indivíduos x objetivos), todos a minimizar
    n = F.shape[0]                                    # número de indivíduos
    menor_igual = (F[:, None, :] <= F[None, :, :]).all(axis=2)   # i não é pior que j em nenhum objetivo
    menor = (F[:, None, :] < F[None, :, :]).any(axis=2)          # i é melhor que j em pelo menos um objetivo
    domina = menor_igual & menor                      # domina[i, j] = True se i domina j
    n_dominado = domina.sum(axis=0)                   # quantos indivíduos dominam cada j
    rank = np.full(n, -1)                             # frente de cada indivíduo (0 = primeira frente)
    frentes = []                                      # lista de frentes (índices)
    atual = np.where(n_dominado == 0)[0]              # primeira frente: ninguém domina
    r = 0                                             # número da frente atual
    while atual.size > 0:                             # enquanto houver indivíduos sem frente
        rank[atual] = r                               # marca a frente
        frentes.append(atual)                         # guarda a frente
        n_dominado = n_dominado - domina[atual].sum(axis=0)   # remove a influência da frente atual
        n_dominado[rank >= 0] = -1                    # já classificados não entram de novo
        atual = np.where(n_dominado == 0)[0]          # próxima frente: quem deixou de ser dominado
        r += 1                                        # avança o número da frente
    return rank, frentes                              # frente de cada indivíduo e lista de frentes


# ---------------- Passo 2b: distância de aglomeração ----------------
def distancia_aglomeracao(F, frentes):                # calcula a distância dentro de cada frente
    dist = np.zeros(F.shape[0])                       # uma distância por indivíduo
    for fr in frentes:                                # percorre as frentes
        if fr.size <= 2:                              # frentes com 1 ou 2 indivíduos: todos são extremos
            dist[fr] = np.inf
            continue                                  # passa para a próxima frente
        for m in range(F.shape[1]):                   # percorre os objetivos
            ordem = fr[np.argsort(F[fr, m])]          # indivíduos da frente ordenados pelo objetivo m
            faixa = F[ordem[-1], m] - F[ordem[0], m]  # amplitude do objetivo m nesta frente
            dist[ordem[0]] = dist[ordem[-1]] = np.inf # extremos recebem distância infinita
            if faixa == 0:                            # objetivo constante na frente: não contribui
                continue
            dist[ordem[1:-1]] += (F[ordem[2:], m] - F[ordem[:-2], m]) / faixa   # vizinhos normalizados pela faixa
    return dist                                       # distância de aglomeração de cada indivíduo


# ---------------- Passo 3: torneio binário com comparação aglomerada ----------------
def torneio_binario(rank, dist, n_pais, rng):         # escolhe n_pais índices por torneio de dois
    a = rng.integers(0, rank.size, n_pais)            # primeiro competidor de cada torneio
    b = rng.integers(0, rank.size, n_pais)            # segundo competidor de cada torneio
    a_vence = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (dist[a] > dist[b]))   # frente menor ou mais isolado
    empate = (rank[a] == rank[b]) & (dist[a] == dist[b])   # empate total
    sorteio = rng.random(n_pais) < 0.5                # desempate aleatório
    a_vence = a_vence | (empate & sorteio)            # aplica o desempate
    return np.where(a_vence, a, b)                    # índice do vencedor de cada torneio


# ---------------- Passo 4a: cruzamento SBX (versão com limites de Deb) ----------------
def sbx(P1, P2, lim_inf, lim_sup, pc, eta_c, rng): # P1, P2: pais pareados (pares x genes)
    C1, C2 = P1.copy(), P2.copy()                     # filhos começam iguais aos pais
    n_pares, n_genes = P1.shape                       # dimensões
    cruza = rng.random(n_pares) < pc                  # quais pares cruzam (probabilidade P_c)
    gene = rng.random((n_pares, n_genes)) < 0.5       # cada gene cruza com probabilidade 0,5 (como no pymoo)
    difere = np.abs(P1 - P2) > 1e-14                  # genes iguais nos dois pais não mudam
    m = cruza[:, None] & gene & difere                # máscara final dos genes que cruzam
    y1 = np.minimum(P1, P2)[m]                        # menor valor do par
    y2 = np.maximum(P1, P2)[m]                        # maior valor do par
    yl = np.broadcast_to(lim_inf, P1.shape)[m]        # limite inferior de cada gene
    yu = np.broadcast_to(lim_sup, P1.shape)[m]        # limite superior de cada gene
    u = rng.random(y1.size)                           # sorteio uniforme por gene
    ex = 1.0 / (eta_c + 1.0)                          # expoente da distribuição

    def beta_q(beta):                                 # fator de espalhamento respeitando o limite
        alfa = 2.0 - beta ** (-(eta_c + 1.0))         # massa de probabilidade dentro do limite
        return np.where(u <= 1.0 / alfa, (u * alfa) ** ex, (1.0 / (2.0 - u * alfa)) ** ex)

    bq1 = beta_q(1.0 + 2.0 * (y1 - yl) / (y2 - y1))   # espalhamento para o lado do limite inferior
    c1 = 0.5 * ((y1 + y2) - bq1 * (y2 - y1))          # filho próximo do menor pai
    bq2 = beta_q(1.0 + 2.0 * (yu - y2) / (y2 - y1))   # espalhamento para o lado do limite superior
    c2 = 0.5 * ((y1 + y2) + bq2 * (y2 - y1))          # filho próximo do maior pai
    c1, c2 = np.clip(c1, yl, yu), np.clip(c2, yl, yu) # recorta nos limites
    troca = rng.random(y1.size) < 0.5                 # troca aleatória de qual filho recebe cada valor
    C1[m] = np.where(troca, c2, c1)                   # gene do primeiro filho
    C2[m] = np.where(troca, c1, c2)                   # gene do segundo filho
    return C1, C2                                     # dois filhos por par


# ---------------- Passo 4b: mutação polinomial (versão com limites de Deb) ----------------
def mutacao_polinomial(X, lim_inf, lim_sup, pm, eta_m, rng):   # X: filhos (indivíduos x genes)
    X = X.copy()                                      # não altera o vetor original
    m = rng.random(X.shape) < pm                      # cada gene muta com probabilidade P_m
    x = X[m]                                          # valores que vão mutar
    xl = np.broadcast_to(lim_inf, X.shape)[m]         # limites inferiores correspondentes
    xu = np.broadcast_to(lim_sup, X.shape)[m]         # limites superiores correspondentes
    d1 = (x - xl) / (xu - xl)                         # distância relativa ao limite inferior
    d2 = (xu - x) / (xu - xl)                         # distância relativa ao limite superior
    u = rng.random(x.size)                            # sorteio uniforme por gene
    ex = 1.0 / (eta_m + 1.0)                          # expoente da distribuição
    baixo = u <= 0.5                                  # metade das mutações desce, metade sobe
    v1 = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - d1) ** (eta_m + 1.0)          # termo para descer
    v2 = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - d2) ** (eta_m + 1.0)  # termo para subir
    dq = np.where(baixo, v1 ** ex - 1.0, 1.0 - v2 ** ex)   # perturbação normalizada
    X[m] = np.clip(x + dq * (xu - xl), xl, xu)        # aplica a perturbação e recorta nos limites
    return X                                          # filhos mutados


# ---------------- Passo 6: seleção elitista ----------------
def selecao_ambiental(F, n):                          # escolhe n sobreviventes entre pais + filhos
    rank, frentes = ordenacao_nao_dominada(F)         # frentes da população combinada
    dist = distancia_aglomeracao(F, frentes)          # distância de aglomeração
    escolhidos = []                                   # índices dos sobreviventes
    for fr in frentes:                                # preenche frente a frente
        if len(escolhidos) + fr.size <= n:            # a frente cabe inteira
            escolhidos.extend(fr.tolist())
        else:                                         # a frente não cabe: corta pela aglomeração
            resto = n - len(escolhidos)               # vagas restantes
            ordem = fr[np.argsort(-dist[fr], kind="stable")]   # mais isolados primeiro
            escolhidos.extend(ordem[:resto].tolist())
            break                                     # população completa
    return np.array(escolhidos)                       # índices dos n sobreviventes


# ---------------- Algoritmo completo ----------------
def nsga2(avaliar, lim_inf, lim_sup, N=40, G=50, pc=0.9, eta_c=20.0, pm=None, eta_m=20.0,
          semente=0, verbose=True):                   # avaliar(x) -> vetor de objetivos (todos a minimizar)
    rng = np.random.default_rng(semente)              # gerador com semente fixa
    lim_inf = np.asarray(lim_inf, dtype=float)        # limites como vetores
    lim_sup = np.asarray(lim_sup, dtype=float)
    n_genes = lim_inf.size                            # número de genes
    pm = 1.0 / n_genes if pm is None else pm          # P_m padrão: 1 / número de genes
    assert N % 2 == 0, "N precisa ser par"            # os pais são pareados dois a dois

    # Passo 1: inicialização uniforme dentro dos limites
    X = lim_inf + rng.random((N, n_genes)) * (lim_sup - lim_inf)   # população inicial
    F = np.array([avaliar(x) for x in X])             # objetivos da população inicial
    historico_X, historico_F = [X.copy()], [F.copy()] # tudo o que foi avaliado (para análise posterior)

    for g in range(G):                                # Passo 7: repete por G gerações
        t0 = time.time()                              # início da geração
        rank, frentes = ordenacao_nao_dominada(F)     # Passo 2: frentes da população atual
        dist = distancia_aglomeracao(F, frentes)      # Passo 2: aglomeração
        pais = torneio_binario(rank, dist, N, rng)    # Passo 3: N pais por torneio
        P1, P2 = X[pais[0::2]], X[pais[1::2]]         # pareia os pais (N/2 pares)
        C1, C2 = sbx(P1, P2, lim_inf, lim_sup, pc, eta_c, rng)   # Passo 4: cruzamento
        Q = mutacao_polinomial(np.vstack([C1, C2]), lim_inf, lim_sup, pm, eta_m, rng)   # Passo 4: mutação
        FQ = np.array([avaliar(q) for q in Q])        # Passo 5: avaliação dos N filhos
        historico_X.append(Q.copy())                  # guarda os filhos avaliados
        historico_F.append(FQ.copy())
        XR, FR = np.vstack([X, Q]), np.vstack([F, FQ])   # Passo 6: pais + filhos (2N)
        sobrev = selecao_ambiental(FR, N)             # Passo 6: N sobreviventes
        X, F = XR[sobrev], FR[sobrev]                 # nova população
        if verbose:                                   # resumo da geração
            n1 = int((ordenacao_nao_dominada(F)[0] == 0).sum())   # tamanho da primeira frente
            print(f"geração {g + 1:3d}/{G}: primeira frente com {n1:3d} indivíduos "
                  f"| melhor f1 {F[:, 0].min():.3f} | {time.time() - t0:.1f} s")

    rank, _ = ordenacao_nao_dominada(F)               # frentes da população final
    frente = rank == 0                                # máscara da primeira frente
    return {"X": X, "F": F, "X_frente": X[frente], "F_frente": F[frente],   # resultado final
            "historico_X": np.vstack(historico_X), "historico_F": np.vstack(historico_F)}   # todas as avaliações
