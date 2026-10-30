# ===== Etapa 07: integridade, frente de Pareto agregada e seleção contra o GMP causal =====
# Uso (na pasta do projeto, depois da etapa 06):
#   python 07_frente_agregada.py
#   python 07_frente_agregada.py --pasta resultados/fom --ref-nmse -38.54 --tolerancia 0.01
# Lê os 10 resultados da busca, confere se estão completos e consistentes, junta as primeiras
# frentes das sementes numa frente não dominada global, desenha os gráficos e exporta as
# soluções com NMSE de validação igual ou melhor que o do GMP causal, ordenadas por n_par.
# Não usa o conjunto de teste: f1 é o NMSE de validação, o mesmo critério da busca.

import argparse, json, os, sys                        # argumentos, leitura do registro e caminhos
import numpy as np                                    # vetores
import matplotlib                                     # biblioteca de gráficos
matplotlib.use("Agg")                                 # gera as figuras sem abrir janela
import matplotlib.pyplot as plt                       # gráficos
from matplotlib.colors import Normalize               # escala de cor comum a todas as figuras

RAIZ = os.path.dirname(os.path.abspath(__file__))     # pasta do projeto
sys.path.insert(0, RAIZ)                              # garante que comum/ seja encontrado
from comum.nsga2 import ordenacao_nao_dominada        # mesma ordenação usada na busca

# ---------------- Argumentos ----------------
parser = argparse.ArgumentParser(description="Integridade, frente agregada e seleção das soluções do FOM")
parser.add_argument("--pasta", default=os.path.join(RAIZ, "resultados", "fom"),
                    help="pasta com fom_semente{s}.npz e _registro.json")   # entrada
parser.add_argument("--saida", default=os.path.join(RAIZ, "resultados", "fom_analise"),
                    help="pasta para figuras e .npz de saída")   # saída
parser.add_argument("--sementes", type=int, nargs="+", default=list(range(10)))   # sementes esperadas
parser.add_argument("--N", type=int, default=40)      # população usada na busca
parser.add_argument("--G", type=int, default=50)      # gerações usadas na busca
parser.add_argument("--ref-nmse", type=float, default=-38.54,
                    help="NMSE de validação do GMP causal, protocolo contínuo (dB)")   # referência
parser.add_argument("--ref-npar", type=int, default=1500,
                    help="parâmetros reais do GMP causal")   # tamanho da referência
parser.add_argument("--tolerancia", type=float, default=0.01,
                    help="empate: NMSE até ref + tolerância (dB)")   # margem do empate
args = parser.parse_args()                            # lê os argumentos
os.makedirs(args.saida, exist_ok=True)                # cria a pasta de saída

# Limites do genoma (iguais aos de comum/fom.py; repetidos aqui para não carregar o dataset)
NOMES_GENES = ["J", "K", "M", "log10_L", "alfa", "log10_tau1", "log10_tau2", "log10_tau3"]   # ordem dos genes
LIM_INF = np.array([1, 1, 0, 1.0, 0.05, 0.0, 0.0, 0.0])   # limites inferiores
LIM_SUP = np.array([3, 9, 20, 3.0, 1.0, 4.0, 4.0, 4.0])   # limites superiores
PENALIDADE_F1 = 100.0                                 # NMSE atribuído aos ajustes que falharam
N_AVAL = args.N * (args.G + 1)                        # avaliações esperadas por semente (40 x 51 = 2 040)


def decodificar(g):                                   # mesma decodificação de comum/fom.py
    g = np.clip(np.asarray(g, dtype=float), LIM_INF, LIM_SUP)   # dentro dos limites
    J = int(round(g[0]))                              # número de estados lentos
    return {"J": J, "K": int(round(g[1])), "M": int(round(g[2])),   # inteiros arredondados
            "L": int(round(10 ** g[3])), "alfa": float(g[4]),       # truncamento e ordem fracionária
            "taus": np.sort(10 ** g[5:5 + J])}        # constantes de tempo ativas, em ordem crescente


def n_par(th):                                        # f2 recalculado a partir da estrutura
    return 2 * th["K"] * (th["M"] + 1) * (th["J"] + 1) + th["J"] + 1


# ======================================================================
# 1) Integridade
# ======================================================================
print("=== 1) integridade ===")
dados, problemas = {}, {}                             # resultados lidos e problemas encontrados por semente
for s in args.sementes:                               # percorre as sementes esperadas
    erros = []                                        # problemas desta semente
    arq = os.path.join(args.pasta, f"fom_semente{s}.npz")   # resultado principal
    arq_reg = os.path.join(args.pasta, f"fom_semente{s}_registro.json")   # registro detalhado
    if not os.path.exists(arq):                       # sem resultado não há o que conferir
        problemas[s] = ["arquivo .npz ausente"]
        continue
    with np.load(arq) as d:                           # carrega tudo para a memória
        d = {k: d[k] for k in d.files}
    if int(d["N"]) != args.N or int(d["G"]) != args.G:   # configuração diferente da esperada
        erros.append(f"N={int(d['N'])}, G={int(d['G'])}")
    if d["historico_F"].shape != (N_AVAL, 3) or d["historico_X"].shape != (N_AVAL, 8):   # busca incompleta
        erros.append(f"histórico com {d['historico_F'].shape[0]} avaliações (esperado {N_AVAL})")
    if not np.all(np.isfinite(d["historico_F"])):     # objetivos não finitos
        erros.append("objetivos não finitos no histórico")
    if np.any(d["historico_X"] < LIM_INF - 1e-9) or np.any(d["historico_X"] > LIM_SUP + 1e-9):   # genes fora dos limites
        erros.append("genes fora dos limites")
    f2_rec = np.array([n_par(decodificar(g)) for g in d["historico_X"]])   # f2 recalculado
    if not np.array_equal(f2_rec, d["historico_F"][:, 1]):   # f2 deve bater exatamente
        erros.append("f2 não bate com a decodificação dos genomas")
    pop_no_hist = all(np.any(np.all(d["historico_X"] == x, axis=1)) for x in d["X"])   # população final foi avaliada?
    if not pop_no_hist:
        erros.append("população final contém genomas que não estão no histórico")
    rank, _ = ordenacao_nao_dominada(d["F"])          # recalcula a primeira frente da população final
    if not np.array_equal(np.sort(d["F"][rank == 0], axis=0), np.sort(d["F_frente"], axis=0)):   # compara com a salva
        erros.append("primeira frente salva difere da recalculada")
    n_pen = int(np.sum(d["historico_F"][:, 0] >= PENALIDADE_F1))   # ajustes penalizados
    if os.path.exists(arq_reg):                       # o registro detalhado deve acompanhar o histórico
        with open(arq_reg) as f:
            reg = json.load(f)
        obj_reg = np.array([r["objetivos"] for r in reg])   # objetivos na ordem em que foram avaliados
        if obj_reg.shape != d["historico_F"].shape or not np.allclose(obj_reg, d["historico_F"]):
            erros.append("registro .json não bate com o histórico do .npz")
        n_erro_reg = sum(1 for r in reg if r.get("erro"))   # falhas numéricas registradas
    else:
        erros.append("registro .json ausente")
        n_erro_reg = None
    dados[s] = d                                      # guarda a semente para as etapas seguintes
    problemas[s] = erros                              # guarda os problemas (lista vazia = íntegra)
    print(f"  semente {s}: {'OK' if not erros else 'PROBLEMAS: ' + '; '.join(erros)} | "
          f"{d['historico_F'].shape[0]} avaliações, {n_pen} penalizadas, falhas no registro: {n_erro_reg}, "
          f"1a frente com {len(d['F_frente'])} soluções, {float(d['duracao_s']) / 3600:.2f} h")

integras = [s for s in args.sementes if s in dados and not problemas[s]]   # sementes que passaram em tudo
print(f"sementes íntegras: {len(integras)} de {len(args.sementes)}")
if not integras:                                      # nada a analisar
    sys.exit("nenhuma semente íntegra; confira a pasta indicada em --pasta")

# ======================================================================
# 2) Frente de Pareto agregada
# ======================================================================
print("\n=== 2) frente agregada ===")
X_all = np.vstack([dados[s]["X_frente"] for s in integras])   # genomas das primeiras frentes de todas as sementes
F_all = np.vstack([dados[s]["F_frente"] for s in integras])   # objetivos correspondentes
origem = np.concatenate([np.full(len(dados[s]["F_frente"]), s) for s in integras])   # semente de cada ponto

chave = [tuple(np.round(f, 10)) + tuple(np.round(x, 10)) for f, x in zip(F_all, X_all)]   # identifica duplicatas exatas
_, unicos = np.unique(np.array(chave), axis=0, return_index=True)   # mantém a primeira ocorrência
unicos = np.sort(unicos)                              # preserva a ordem original
X_all, F_all, origem = X_all[unicos], F_all[unicos], origem[unicos]   # sem duplicatas

rank, _ = ordenacao_nao_dominada(F_all)               # não dominância sobre a união
glob = rank == 0                                      # máscara da frente agregada
X_ag, F_ag, orig_ag = X_all[glob], F_all[glob], origem[glob]   # frente agregada
ordem = np.argsort(F_ag[:, 1])                        # ordena por número de parâmetros
X_ag, F_ag, orig_ag = X_ag[ordem], F_ag[ordem], orig_ag[ordem]
print(f"união das frentes: {len(F_all)} soluções distintas; frente agregada: {len(F_ag)}")
print("contribuição de cada semente:", {int(s): int(np.sum(orig_ag == s)) for s in integras})

melhor = np.array([dados[s]["F_frente"][:, 0].min() for s in integras])   # melhor NMSE de validação por semente
print(f"melhor NMSE de validação por semente: mediana {np.median(melhor):.2f} dB, "
      f"IQR {np.subtract(*np.percentile(melhor, [75, 25])):.2f} dB, "
      f"faixa [{melhor.min():.2f}, {melhor.max():.2f}] dB")

tau1 = np.array([decodificar(x)["taus"][0] for x in X_ag])   # menor constante de tempo de cada solução
print(f"tau mínimo <= 2 amostras em {np.mean(tau1 <= 2):.0%} da frente agregada")


def limites(X, rotulo):                               # fração das soluções com cada gene no limite do espaço de busca
    th = [decodificar(x) for x in X]                  # estruturas decodificadas
    partes = []                                       # textos de cada gene
    for i, nome in enumerate(NOMES_GENES[:5]):        # genes estruturais e alfa
        if nome in ("J", "K", "M"):                   # genes inteiros: compara o valor já arredondado
            v = np.array([t[nome] for t in th])       # valor decodificado
            inf, sup = np.mean(v == LIM_INF[i]), np.mean(v == LIM_SUP[i])   # fração exatamente no limite
        else:                                         # genes contínuos: margem de 2 % da faixa
            tol = 0.02 * (LIM_SUP[i] - LIM_INF[i])    # margem
            inf = np.mean(np.isclose(X[:, i], LIM_INF[i], atol=tol))   # fração perto do mínimo
            sup = np.mean(np.isclose(X[:, i], LIM_SUP[i], atol=tol))   # fração perto do máximo
        partes.append(f"{nome} {inf:.0%}/{sup:.0%}")  # inferior/superior
    print(f"  no limite inf./sup. ({rotulo}): " + ", ".join(partes))


print("genes nos limites do espaço de busca (um limite superior muito ocupado sugere ampliar a faixa):")
limites(X_ag, "frente agregada")                      # toda a frente

# ======================================================================
# 3) Gráficos
# ======================================================================
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})   # estilo discreto
TINTA = "#3d3d3d"                                     # cor neutra das linhas de referência e textos
CMAP = plt.get_cmap("Blues")                          # escala sequencial de um só matiz para log10 kappa
k_todos = np.concatenate([dados[s]["F_frente"][:, 2] for s in integras])   # kappa de todas as frentes
NORM = Normalize(vmin=np.floor(k_todos.min()), vmax=np.ceil(k_todos.max()))   # mesma escala em todas as figuras
y_min = min(F_all[:, 0].min(), args.ref_nmse) - 0.3   # limites comuns do eixo de NMSE
y_max = np.percentile(F_all[:, 0], 98) + 0.5          # corta só valores extremos
x_lim = (max(10, F_all[:, 1].min() * 0.8), max(F_all[:, 1].max(), args.ref_npar) * 1.25)   # eixo de parâmetros


def referencia(ax):                                   # desenha o GMP causal como ponto e linha horizontal
    ax.axhline(args.ref_nmse, color=TINTA, lw=0.8, ls="--", zorder=1)   # nível de NMSE do GMP causal
    ax.plot(args.ref_npar, args.ref_nmse, marker="D", ms=6, color=TINTA, mec="white", mew=1.5, zorder=4)   # o próprio GMP
    ax.annotate("causal GMP", (x_lim[0], args.ref_nmse), xytext=(3, 3), textcoords="offset points",
                ha="left", va="bottom", fontsize=7, color=TINTA)   # rótulo direto na ponta esquerda da linha


def painel_frente(ax, F, titulo):                     # NMSE x parâmetros, cor = log10 kappa
    o = np.lexsort((F[:, 0], F[:, 1]))                # ordena por parâmetros (empates pelo NMSE)
    env = np.minimum.accumulate(F[o, 0])              # melhor NMSE obtido com até n_par parâmetros
    ax.step(F[o, 1], env, where="post", color="#9e9e9e", lw=0.9, zorder=2)   # envelope NMSE x n_par (projeção 2D da frente)
    sc = ax.scatter(F[:, 1], F[:, 0], c=F[:, 2], cmap=CMAP, norm=NORM, s=22,
                    edgecolor=TINTA, linewidth=0.4, zorder=3)   # pontos coloridos pelo condicionamento
    referencia(ax)                                    # GMP causal
    ax.set_xscale("log"); ax.set_xlim(*x_lim); ax.set_ylim(y_min, y_max)   # escalas comuns
    ax.set_title(titulo, fontsize=8)                  # título do painel
    return sc                                         # para a barra de cor


def salvar(fig, nome):                                # grava a figura em PDF e PNG
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.saida, f"{nome}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)                                    # libera a memória


# 3a) Uma figura por semente
for s in integras:                                    # percorre as sementes
    fig, ax = plt.subplots(figsize=(3.5, 2.6))        # tamanho de uma coluna IEEE
    sc = painel_frente(ax, dados[s]["F_frente"], f"Seed {s}: first front ({len(dados[s]['F_frente'])} solutions)")
    ax.set_xlabel("Real parameters $n_{\\mathrm{par}}$"); ax.set_ylabel("Validation NMSE (dB)")   # eixos
    fig.colorbar(sc, ax=ax, label="$\\log_{10}\\kappa$")   # escala de cor
    salvar(fig, f"frente_semente{s}")                 # grava

# 3b) As 10 frentes num só painel 2 x 5 (para conferência ou apêndice)
n_col = 5                                             # colunas da grade
n_lin = int(np.ceil(len(integras) / n_col))           # linhas necessárias
fig, axs = plt.subplots(n_lin, n_col, figsize=(12, 2.4 * n_lin), sharex=True, sharey=True, squeeze=False)   # grade
for ax, s in zip(axs.flat, integras):                 # um painel por semente
    sc = painel_frente(ax, dados[s]["F_frente"], f"Seed {s}")
for ax in axs.flat[len(integras):]:                   # painéis sobrando ficam vazios
    ax.set_visible(False)
for ax in axs[-1]:                                    # rótulo do eixo x só na última linha
    ax.set_xlabel("$n_{\\mathrm{par}}$")
for ax in axs[:, 0]:                                  # rótulo do eixo y só na primeira coluna
    ax.set_ylabel("Val. NMSE (dB)")
fig.colorbar(sc, ax=axs, label="$\\log_{10}\\kappa$", shrink=0.8)   # uma barra de cor para todos
salvar(fig, "frentes_10_sementes")                    # grava

# 3c) Frente agregada: NMSE x parâmetros e log10 kappa x parâmetros (dois painéis, um eixo y cada)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.6), sharex=True)   # uma coluna IEEE
ax1.scatter(F_all[~glob, 1], F_all[~glob, 0], s=8, color="#d0d0d0", zorder=2, label="seed fronts (dominated)")   # pontos dominados
sc = painel_frente(ax1, F_ag, "")                     # frente agregada colorida por kappa
ax1.set_ylabel("Validation NMSE (dB)")                # eixo y de cima
from matplotlib.lines import Line2D                    # marcadores da legenda
ax1.legend(handles=[Line2D([], [], marker="o", ls="", color="#d0d0d0", ms=4, label="seed fronts (dominated)"),
                    Line2D([], [], marker="o", ls="", mfc=CMAP(0.7), mec=TINTA, mew=0.4, ms=5, label="aggregated front"),
                    Line2D([], [], color="#9e9e9e", lw=0.9, label="best NMSE up to $n_{\\mathrm{par}}$")],
           loc="upper right", fontsize=6.5, frameon=False)   # legenda com marcadores neutros
ax2.scatter(F_all[~glob, 1], F_all[~glob, 2], s=8, color="#d0d0d0", zorder=2)   # dominados
ax2.scatter(F_ag[:, 1], F_ag[:, 2], c=F_ag[:, 2], cmap=CMAP, norm=NORM, s=22,
            edgecolor=TINTA, linewidth=0.4, zorder=3)  # frente agregada
ax2.axhline(np.log10(1 / np.finfo(float).eps), color=TINTA, lw=0.8, ls=":")   # limite da precisão dupla (~15,65)
ax2.annotate("double-precision limit", (x_lim[0], np.log10(1 / np.finfo(float).eps)), xytext=(4, -10),
             textcoords="offset points", fontsize=7, color=TINTA)   # rótulo direto
ax2.set_xlabel("Real parameters $n_{\\mathrm{par}}$"); ax2.set_ylabel("$\\log_{10}\\kappa$")   # eixos
ax2.set_xscale("log"); ax2.set_xlim(*x_lim)           # mesma escala do painel de cima
salvar(fig, "frente_agregada")                        # grava

# ======================================================================
# 4) Soluções que empatam ou superam o GMP causal (NMSE de validação)
# ======================================================================
print("\n=== 4) seleção contra o GMP causal ===")
limite = args.ref_nmse + args.tolerancia              # empate: até a referência mais a tolerância
sel = F_ag[:, 0] <= limite                            # máscara sobre a frente agregada
X_sel, F_sel, orig_sel = X_ag[sel], F_ag[sel], orig_ag[sel]   # soluções escolhidas
o = np.lexsort((F_sel[:, 0], F_sel[:, 1]))            # por parâmetros; empate desfeito pelo NMSE
X_sel, F_sel, orig_sel = X_sel[o], F_sel[o], orig_sel[o]
th_sel = [decodificar(x) for x in X_sel]              # estruturas decodificadas
if len(X_sel):                                        # mesma verificação de limites só nas selecionadas
    limites(X_sel, "selecionadas")

print(f"critério: NMSE de validação <= {limite:.2f} dB (GMP causal {args.ref_nmse:.2f} dB + {args.tolerancia} dB)")
print(f"{len(F_sel)} de {len(F_ag)} soluções da frente agregada atendem ao critério")
print(f"{'#':>3s} {'semente':>7s} {'NMSE val':>9s} {'n_par':>6s} {'%GMP':>5s} {'log10k':>7s} | "
      f"{'J':>2s} {'K':>2s} {'M':>3s} {'L':>5s} {'alfa':>6s}  taus")
for i, (f, th, s) in enumerate(zip(F_sel, th_sel, orig_sel)):   # uma linha por solução
    print(f"{i:3d} {int(s):7d} {f[0]:9.2f} {int(f[1]):6d} {100 * f[1] / args.ref_npar:4.0f}% {f[2]:7.2f} | "
          f"{th['J']:2d} {th['K']:2d} {th['M']:3d} {th['L']:5d} {th['alfa']:6.3f}  {np.round(th['taus'], 2)}")

taus_mat = np.full((len(th_sel), 3), np.nan)          # constantes de tempo com NaN nos estados inativos
for i, th in enumerate(th_sel):
    taus_mat[i, :th["J"]] = th["taus"]
np.savez(os.path.join(args.saida, "solucoes_vs_gmp_causal.npz"),
         genoma=X_sel, objetivos=F_sel, semente=orig_sel,   # genomas, [f1, f2, f3] e semente de origem
         J=np.array([t["J"] for t in th_sel]), K=np.array([t["K"] for t in th_sel]),   # estrutura decodificada
         M=np.array([t["M"] for t in th_sel]), L=np.array([t["L"] for t in th_sel]),
         alfa=np.array([t["alfa"] for t in th_sel]), taus=taus_mat,
         ref_nmse=args.ref_nmse, ref_npar=args.ref_npar, tolerancia=args.tolerancia,   # critério usado
         nomes_genes=np.array(NOMES_GENES))           # ordem dos genes
np.savez(os.path.join(args.saida, "frente_agregada.npz"),
         genoma=X_ag, objetivos=F_ag, semente=orig_ag, nomes_genes=np.array(NOMES_GENES))   # frente completa
print(f"\nfiguras e arquivos gravados em {args.saida}")