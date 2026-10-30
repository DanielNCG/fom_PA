# ===== Etapa 05 (bloco 9): validação no ZDT1 contra o NSGA-II do pymoo =====
# Uso: python 05_validacao_zdt1.py   (requer: pip install pymoo matplotlib)
# Grava a figura em resultados/zdt1_frentes.pdf e .png.
# Protocolo da metodologia: N = 100, G = 250, SBX (P_c = 0,9, eta_c = 20),
# mutação polinomial (P_m = 1/30, eta_m = 20), 10 sementes para cada implementação,
# hipervolume com ponto de referência (1,1; 1,1), IGD com 1 000 pontos da frente real
# e teste U de Mann-Whitney bilateral a 5 %.

import numpy as np                                    # vetores
from scipy.stats import mannwhitneyu                  # teste estatístico
from pymoo.algorithms.moo.nsga2 import NSGA2          # NSGA-II de referência
from pymoo.operators.crossover.sbx import SBX         # cruzamento SBX do pymoo
from pymoo.operators.mutation.pm import PM            # mutação polinomial do pymoo
from pymoo.operators.sampling.rnd import FloatRandomSampling   # inicialização uniforme
from pymoo.problems import get_problem                # problemas de teste
from pymoo.optimize import minimize                   # laço de otimização do pymoo
from comum.nsga2 import nsga2, ordenacao_nao_dominada   # nossa implementação
from comum.dados import PASTA_RESULTADOS              # pasta de saída

N_VAR = 30                                            # variáveis de decisão do ZDT1
N_POP, N_GER = 100, 250                               # tamanho da população e número de gerações
SEMENTES = range(10)                                  # 10 execuções independentes
REF = np.array([1.1, 1.1])                            # ponto de referência do hipervolume


def zdt1(x):                                          # ZDT1 implementado à mão (mesma definição do pymoo)
    f1 = x[0]                                         # primeiro objetivo
    g = 1.0 + 9.0 * np.sum(x[1:]) / (N_VAR - 1)       # função auxiliar g
    f2 = g * (1.0 - np.sqrt(f1 / g))                  # segundo objetivo
    return np.array([f1, f2])                         # vetor de objetivos


def hipervolume_2d(F, ref):                           # área dominada pela frente até o ponto de referência
    F = F[(F[:, 0] < ref[0]) & (F[:, 1] < ref[1])]    # só pontos que dominam a referência
    if F.size == 0:                                   # nenhum ponto útil
        return 0.0
    F = F[np.argsort(F[:, 0])]                        # ordena por f1 crescente
    hv, f2_min = 0.0, ref[1]                          # área acumulada e menor f2 visto até agora
    for f1, f2 in F:                                  # varre da esquerda para a direita
        if f2 < f2_min:                               # só pontos não dominados acrescentam área
            hv += (ref[0] - f1) * (f2_min - f2)       # faixa horizontal nova
            f2_min = f2                               # atualiza o degrau
    return hv                                         # hipervolume


f1_real = np.linspace(0.0, 1.0, 1000)                 # 1 000 pontos uniformes em f1
FRENTE_REAL = np.column_stack([f1_real, 1.0 - np.sqrt(f1_real)])   # frente de Pareto exata: f2 = 1 - sqrt(f1)


def igd(F, frente_real=FRENTE_REAL):                  # distância média da frente real à solução mais próxima
    d = np.linalg.norm(frente_real[:, None, :] - F[None, :, :], axis=2)   # distâncias ponto a ponto
    return d.min(axis=1).mean()                       # média das menores distâncias


def frente_nao_dominada(F):                           # extrai a primeira frente de um conjunto de pontos
    rank, _ = ordenacao_nao_dominada(F)               # função de comum/nsga2.py
    return F[rank == 0]                               # só os não dominados


resultados = {"nosso": {"hv": [], "igd": []}, "pymoo": {"hv": [], "igd": []}}   # indicadores por semente
frentes_finais = {}                                   # guarda uma frente de cada para o gráfico
problema = get_problem("zdt1", n_var=N_VAR)           # ZDT1 do pymoo (30 variáveis)

for s in SEMENTES:                                    # 10 sementes
    r = nsga2(zdt1, np.zeros(N_VAR), np.ones(N_VAR), N=N_POP, G=N_GER, pc=0.9, eta_c=20,
              pm=1.0 / N_VAR, eta_m=20, semente=s, verbose=False)   # nossa implementação
    Fn = r["F_frente"]                                # primeira frente da população final
    resultados["nosso"]["hv"].append(hipervolume_2d(Fn, REF))   # hipervolume
    resultados["nosso"]["igd"].append(igd(Fn))        # IGD

    alg = NSGA2(pop_size=N_POP, sampling=FloatRandomSampling(),     # mesmo tamanho e inicialização uniforme
                crossover=SBX(prob=0.9, prob_var=0.5, eta=20),      # SBX com os mesmos parâmetros
                mutation=PM(prob=1.0, prob_var=1.0 / N_VAR, eta=20),   # cada gene muta com P_m = 1/30
                eliminate_duplicates=False)           # sem filtro de duplicatas (nossa versão não filtra)
    res = minimize(problema, alg, ("n_gen", N_GER + 1), seed=s, verbose=False)   # +1: o pymoo conta a inicialização
    Fp = frente_nao_dominada(res.pop.get("F"))        # primeira frente da população final do pymoo
    resultados["pymoo"]["hv"].append(hipervolume_2d(Fp, REF))   # mesmo cálculo de hipervolume
    resultados["pymoo"]["igd"].append(igd(Fp))        # mesmo cálculo de IGD
    if s == 0:                                        # guarda as frentes da primeira semente
        frentes_finais = {"nosso": Fn, "pymoo": Fp}
    print(f"semente {s}: HV nosso {resultados['nosso']['hv'][-1]:.5f} | pymoo {resultados['pymoo']['hv'][-1]:.5f} "
          f"|| IGD nosso {resultados['nosso']['igd'][-1]:.5f} | pymoo {resultados['pymoo']['igd'][-1]:.5f}")

print(f"\nHV da frente real (referência teórica): {hipervolume_2d(FRENTE_REAL, REF):.5f}")   # máximo atingível (aprox.)
aceito = True                                         # resultado do critério de aceitação
for ind in ("hv", "igd"):                             # compara os dois indicadores
    a = np.array(resultados["nosso"][ind])            # valores da nossa implementação
    b = np.array(resultados["pymoo"][ind])            # valores do pymoo
    p = mannwhitneyu(a, b, alternative="two-sided").pvalue   # teste U bilateral
    aceito &= p >= 0.05                               # aceita se não houver diferença significativa
    print(f"{ind.upper():3s}: nosso mediana {np.median(a):.5f} (IQR {np.subtract(*np.percentile(a, [75, 25])):.5f}) | "
          f"pymoo mediana {np.median(b):.5f} (IQR {np.subtract(*np.percentile(b, [75, 25])):.5f}) | p = {p:.3f}")
print("implementação aceita" if aceito else "diferença significativa: revisar a implementação")   # veredito

import matplotlib                                     # biblioteca de gráficos
matplotlib.use("Agg")                                 # gera a figura sem abrir janela
import matplotlib.pyplot as plt                       # gráfico das frentes
fig, ax = plt.subplots(figsize=(5, 4))                # uma figura pequena
ax.plot(FRENTE_REAL[:, 0], FRENTE_REAL[:, 1], color="0.6", lw=2, label="real Pareto front")   # frente exata
ax.plot(*frentes_finais["pymoo"].T, "s", ms=4, mfc="none", color="#c4561c", label="pymoo")   # pymoo
ax.plot(*frentes_finais["nosso"].T, "o", ms=3, color="#2f5fb3", label="our implementation")   # nossa
ax.set_xlabel("$f_1$"); ax.set_ylabel("$f_2$")        # rótulos dos eixos
ax.legend(frameon=False); ax.grid(alpha=0.3)          # legenda e grade
fig.tight_layout()                                    # ajusta as margens
for ext in ("pdf", "png"):                           # grava nos dois formatos
    fig.savefig(PASTA_RESULTADOS / f"zdt1_frentes.{ext}", dpi=200)
print(f"figura salva em {PASTA_RESULTADOS / 'zdt1_frentes.pdf'}")
