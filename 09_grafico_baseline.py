# ===== Etapa 09: gráfico da frente de Pareto do baseline alfa = 1 =====
# Uso (na pasta do projeto, depois da etapa 08):
#   python 09_grafico_baseline.py
#   python 09_grafico_baseline.py --semente 0 --ref-nmse -38.54
# Gera a frente do baseline no mesmo estilo e na mesma escala de cor das figuras da etapa 07.
# Se encontrar o resultado da mesma semente da busca fracionária, gera também uma figura
# com os dois envelopes NMSE x n_par sobrepostos.

import argparse, os, sys                              # argumentos e caminhos
import numpy as np                                    # vetores
import matplotlib                                     # biblioteca de gráficos
matplotlib.use("Agg")                                 # gera as figuras sem abrir janela
import matplotlib.pyplot as plt                       # gráficos
from matplotlib.colors import Normalize               # escala de cor fixa
from matplotlib.lines import Line2D                   # marcadores da legenda

RAIZ = os.path.dirname(os.path.abspath(__file__))     # pasta do projeto

# ---------------- Argumentos ----------------
parser = argparse.ArgumentParser(description="Frente de Pareto do baseline alfa = 1")
parser.add_argument("--semente", type=int, default=0)   # semente do baseline
parser.add_argument("--pasta", default=os.path.join(RAIZ, "resultados", "baseline"))   # resultados da etapa 08
parser.add_argument("--pasta-fom", default=os.path.join(RAIZ, "resultados", "fom"))    # resultados da etapa 06
parser.add_argument("--saida", default=os.path.join(RAIZ, "resultados", "baseline_analise"))   # figuras
parser.add_argument("--ref-nmse", type=float, default=-38.54)   # NMSE de validação do GMP causal (dB)
parser.add_argument("--ref-npar", type=int, default=1500)       # parâmetros reais do GMP causal
args = parser.parse_args()                            # lê os argumentos
os.makedirs(args.saida, exist_ok=True)                # cria a pasta de saída

# ---------------- Leitura e conferência ----------------
arq = os.path.join(args.pasta, f"baseline_semente{args.semente}.npz")   # resultado do baseline
if not os.path.exists(arq):                           # sem resultado não há gráfico
    sys.exit(f"não encontrei {arq}; rode a etapa 08 ou indique --pasta")
with np.load(arq) as d:                               # carrega tudo para a memória
    d = {k: d[k] for k in d.files}
N, G = int(d["N"]), int(d["G"])                       # configuração da busca
assert d["historico_F"].shape[0] == N * (G + 1), "histórico incompleto"   # execução completa
F = d["F_frente"]                                     # primeira frente: [NMSE val, n_par, log10 kappa]
X = d["X_frente"]                                     # genomas: J, K, M, log10_tau1..3
n_pen = int(np.sum(d["historico_F"][:, 0] >= 100.0))  # ajustes penalizados
print(f"baseline, semente {args.semente}: N = {N}, G = {G}, {d['historico_F'].shape[0]} avaliações, "
      f"{n_pen} penalizadas, primeira frente com {len(F)} soluções")
print(f"  n_par {int(F[:, 1].min())}-{int(F[:, 1].max())} | NMSE val {F[:, 0].min():.2f} a {F[:, 0].max():.2f} dB | "
      f"log10 kappa {F[:, 2].min():.2f}-{F[:, 2].max():.2f}")
ok = F[:, 0] <= args.ref_nmse + 0.01                  # mesmo critério de empate da etapa 07
print(f"  soluções com NMSE <= {args.ref_nmse + 0.01:.2f} dB: {int(ok.sum())}"
      + (f", a menor com {int(F[ok, 1].min())} parâmetros" if ok.any() else ""))

# ---------------- Estilo (igual ao da etapa 07) ----------------
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})   # estilo discreto
TINTA = "#3d3d3d"                                     # cor neutra das referências e textos
CMAP = plt.get_cmap("Blues")                          # escala sequencial de um só matiz para log10 kappa
NORM = Normalize(vmin=0, vmax=17)                     # mesma faixa de cor das figuras do FOM (0 a 17)
x_lim = (max(4, F[:, 1].min() * 0.8), max(F[:, 1].max(), args.ref_npar) * 1.25)   # eixo de parâmetros
y_lim = (min(F[:, 0].min(), args.ref_nmse) - 0.3, np.percentile(F[:, 0], 98) + 0.5)   # eixo de NMSE


def envelope(F):                                      # melhor NMSE obtido com até n_par parâmetros
    o = np.lexsort((F[:, 0], F[:, 1]))                # ordena por n_par (empates pelo NMSE)
    return F[o, 1], np.minimum.accumulate(F[o, 0])    # abscissas e escada


def referencia(ax):                                   # GMP causal: linha horizontal e ponto
    ax.axhline(args.ref_nmse, color=TINTA, lw=0.8, ls="--", zorder=1)   # nível do GMP causal
    ax.plot(args.ref_npar, args.ref_nmse, marker="D", ms=6, color=TINTA, mec="white", mew=1.5, zorder=4)   # o próprio GMP
    ax.annotate("causal GMP", (x_lim[0], args.ref_nmse), xytext=(3, 3), textcoords="offset points",
                ha="left", va="bottom", fontsize=7, color=TINTA)   # rótulo direto


def salvar(fig, nome):                                # grava em PDF e PNG
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.saida, f"{nome}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)                                    # libera a memória
    print(f"figura salva: {os.path.join(args.saida, nome)}.pdf")


# ---------------- Figura 1: frente do baseline ----------------
fig, ax = plt.subplots(figsize=(3.5, 2.6))            # uma coluna IEEE
xs, ys = envelope(F)                                  # escada NMSE x n_par
ax.step(xs, ys, where="post", color="#9e9e9e", lw=0.9, zorder=2)   # envelope
sc = ax.scatter(F[:, 1], F[:, 0], c=F[:, 2], cmap=CMAP, norm=NORM, s=22,
                edgecolor=TINTA, linewidth=0.4, zorder=3)   # pontos coloridos pelo condicionamento
referencia(ax)                                        # GMP causal
ax.set_xscale("log"); ax.set_xlim(*x_lim); ax.set_ylim(*y_lim)   # escalas
ax.set_xlabel("Real parameters $n_{\\mathrm{par}}$"); ax.set_ylabel("Validation NMSE (dB)")   # eixos
fig.colorbar(sc, ax=ax, label="$\\log_{10}\\kappa$")   # escala de cor
salvar(fig, f"frente_baseline_semente{args.semente}")   # grava

# ---------------- Figura 2 (opcional): envelopes do baseline e do FOM na mesma semente ----------------
arq_fom = os.path.join(args.pasta_fom, f"fom_semente{args.semente}.npz")   # resultado fracionário da mesma semente
if os.path.exists(arq_fom):                           # só se o arquivo estiver disponível
    with np.load(arq_fom) as f:
        F_fom = f["F_frente"]                         # primeira frente do FOM
    fig, ax = plt.subplots(figsize=(3.5, 2.4))        # uma coluna IEEE
    for Fx, cor, nome in ((F_fom, "#2f5fb3", "FOM"), (F, "#c4561c", "baseline ($\\alpha=1$)")):   # duas séries
        xs, ys = envelope(Fx)                         # escada de cada modelo
        ax.step(xs, ys, where="post", color=cor, lw=1.2, zorder=3, label=nome)   # envelope
        ax.scatter(Fx[:, 1], Fx[:, 0], s=10, color=cor, alpha=0.35, linewidth=0, zorder=2)   # pontos da frente
    referencia(ax)                                    # GMP causal
    lo = min(F_fom[:, 1].min(), F[:, 1].min()) * 0.8  # eixo comum
    ax.set_xscale("log"); ax.set_xlim(max(4, lo), x_lim[1])
    ax.set_ylim(min(F_fom[:, 0].min(), y_lim[0]), y_lim[1])
    ax.set_xlabel("Real parameters $n_{\\mathrm{par}}$"); ax.set_ylabel("Validation NMSE (dB)")   # eixos
    ax.legend(loc="upper right", fontsize=7, frameon=False)   # legenda
    salvar(fig, f"envelopes_fom_baseline_semente{args.semente}")   # grava
else:
    print(f"(sem {arq_fom}: figura de comparação não gerada)")