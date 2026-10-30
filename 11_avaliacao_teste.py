# ===== Etapa 11: avaliação no conjunto de teste =====
# Uso (na pasta do projeto, depois da etapa 10):
#   python 11_avaliacao_teste.py
#   python 11_avaliacao_teste.py --janela 256 --inicio-janela 1200   # escolhe o trecho do gráfico temporal
# Avalia no conjunto de teste os 6 modelos da etapa 10, o GMP de benchmark (protocolo segmentado)
# e o GMP causal (protocolo contínuo). Para cada modelo calcula as mesmas métricas da Tabela
# dos modelos selecionados: n_par, coeficientes complexos, posto retido, log10 kappa e NMSE de
# validação e de teste. Grava as métricas, as saídas previstas e quatro figuras:
#   1) NMSE de teste x n_par;  2) NMSE de validação x teste;
#   3) resposta temporal |y| medida x prevista num trecho do teste;  4) PSD da saída e dos erros.

import argparse, json, os, sys, time                  # argumentos, gravação, caminhos e tempo
import numpy as np                                    # vetores
import matplotlib                                     # biblioteca de gráficos
matplotlib.use("Agg")                                 # gera as figuras sem abrir janela
import matplotlib.pyplot as plt                       # gráficos
from scipy.signal import welch                        # densidade espectral de potência

RAIZ = os.path.dirname(os.path.abspath(__file__))     # pasta do projeto
sys.path.insert(0, RAIZ)                              # garante que comum/ seja encontrado
from comum.dados import x, y, CORTES, fs, nmse_db, PASTA_RESULTADOS   # registro, divisão, f_s e métrica
from comum.gmp import SEG, desloca, segmentos         # segmento do benchmark e deslocamentos
from comum.fom import (decodificar, estados_lentos, base_fom, ajuste_truncado,   # modelo fracionário
                       P_X, AQUEC, BLOCO)

# ---------------- Argumentos ----------------
parser = argparse.ArgumentParser(description="Avaliação dos modelos no conjunto de teste")
parser.add_argument("--modelos", default=str(PASTA_RESULTADOS / "modelos_selecionados.json"))   # etapa 10
parser.add_argument("--saida", default=str(PASTA_RESULTADOS / "teste"))   # pasta de saída
parser.add_argument("--janela", type=int, default=200)          # amostras no gráfico temporal
parser.add_argument("--inicio-janela", type=int, default=None)  # início (índice dentro do teste); padrão: centrado no pico de |x|
args = parser.parse_args()                            # lê os argumentos
os.makedirs(args.saida, exist_ok=True)                # cria a pasta de saída
if not os.path.exists(args.modelos):                  # a etapa 10 precisa ter rodado
    sys.exit(f"não encontrei {args.modelos}; rode a etapa 10 ou indique --modelos")

RCOND = 1e-4                                          # corte do SVD truncado (igual em todos os modelos)


# ======================================================================
# GMPs (mesma construção das etapas 02 e 03)
# ======================================================================
def base_gmp_geral(s, Ka, La, Kb, Lb, Mb, Kc, Lc, Mc):   # regressores do GMP para um trecho s
    cols = []                                         # colunas da base
    for k in range(Ka):                               # termos alinhados x[n-q] |x[n-q]|^k
        for q in range(La):
            xq = desloca(s, q)
            cols.append(xq * np.abs(xq) ** k)
    for k in range(1, Kb + 1):                        # envoltória atrasada x[n-q] |x[n-q-m]|^k
        for q in range(Lb):
            for m in range(1, Mb + 1):
                cols.append(desloca(s, q) * np.abs(desloca(s, q + m)) ** k)
    for k in range(1, Kc + 1):                        # envoltória adiantada x[n-q] |x[n-q+m]|^k
        for q in range(Lc):
            for m in range(1, Mc + 1):
                cols.append(desloca(s, q) * np.abs(desloca(s, q - m)) ** k)
    return np.column_stack(cols)                      # matriz (amostras) x coeficientes


CFG_BENCH = dict(Ka=5, La=30, Kb=4, Lb=30, Mb=5, Kc=4, Lc=30, Mc=5)   # GMP de benchmark (1 350 coef.)
CFG_CAUSAL = dict(Ka=5, La=30, Kb=4, Lb=30, Mb=5, Kc=0, Lc=0, Mc=0)   # GMP causal (750 coef.)
MEM_CAUSAL = 29 + 5                                   # amostras de passado que o GMP causal usa


def ajuste_qr(pedacos, n_col):                        # mínimos quadrados truncados (mesmo procedimento das etapas 02/03)
    R = np.zeros((0, n_col + 1), dtype=complex)       # fator R acumulado de [Phi | y]
    for P, yb in pedacos:                             # pedaços da base e da saída
        R = np.linalg.qr(np.vstack([R, np.hstack([P, yb[:, None]])]), mode="r")   # QR incremental
    Ra, bq = R[:n_col, :n_col], R[:n_col, n_col]      # regressores e projeção da saída
    normas = np.linalg.norm(R[:, :n_col], axis=0)     # norma de cada coluna
    U, sv, Vh = np.linalg.svd(Ra / normas)            # SVD das colunas normalizadas
    manter = sv > sv[0] * RCOND                       # direções acima do corte
    c = Vh[manter].conj().T @ ((U[:, manter].conj().T @ bq) / sv[manter])   # solução truncada
    return c / normas, int(manter.sum()), float(np.log10(sv[0] / sv[-1]))   # coeficientes, posto e log10 kappa


def gmp_benchmark():                                  # protocolo segmentado: zeros nas bordas de cada segmento
    cfg = CFG_BENCH
    n_col = cfg["Ka"] * cfg["La"] + cfg["Kb"] * cfg["Lb"] * cfg["Mb"] + cfg["Kc"] * cfg["Lc"] * cfg["Mc"]
    i0, i1 = CORTES["treino"]                         # limites do treino
    c, posto, lk = ajuste_qr(((base_gmp_geral(x[i0 + a:i0 + b], **cfg), y[i0 + a:i0 + b])
                              for a, b in segmentos(i1 - i0)), n_col)   # ajuste segmento a segmento

    def prever(nome):                                 # previsão de um conjunto, segmento a segmento
        j0, j1 = CORTES[nome]
        return np.concatenate([base_gmp_geral(x[j0 + a:j0 + b], **cfg) @ c for a, b in segmentos(j1 - j0)])
    return {"n_coef": n_col, "n_par": 2 * n_col, "posto": posto, "log10_kappa": lk,
            "y_val": prever("val"), "y_teste": prever("teste")}


def gmp_causal():                                     # protocolo contínuo: passado real do registro
    cfg = CFG_CAUSAL
    n_col = cfg["Ka"] * cfg["La"] + cfg["Kb"] * cfg["Lb"] * cfg["Mb"]

    def base(a, b):                                   # regressores das amostras a..b-1 com histórico
        ini = max(0, a - MEM_CAUSAL)
        return base_gmp_geral(x[ini:b], **cfg)[a - ini:]
    i0, i1 = CORTES["treino"]
    pedacos = [(a, min(a + SEG, i1)) for a in range(i0 + MEM_CAUSAL, i1, SEG)]   # treino sem as amostras sem passado
    c, posto, lk = ajuste_qr(((base(a, b), y[a:b]) for a, b in pedacos), n_col)

    def prever(nome):
        j0, j1 = CORTES[nome]
        return np.concatenate([base(a, min(a + SEG, j1)) @ c for a in range(j0, j1, SEG)])
    return {"n_coef": n_col, "n_par": 2 * n_col, "posto": posto, "log10_kappa": lk,
            "y_val": prever("val"), "y_teste": prever("teste")}


# ======================================================================
# FOM e baseline (mesmo ajuste de comum/fom.py, devolvendo também as previsões)
# ======================================================================
def avaliar_estrutura(g8, familia):                   # ajusta no treino e prevê validação e teste
    th = decodificar(g8)                              # estrutura
    J, K, M, L = th["J"], th["K"], th["M"], th["L"]
    S = estados_lentos(P_X, th["alfa"], th["taus"], L)   # estados lentos no registro inteiro
    n_col = K * (M + 1) * (J + 1)                     # coeficientes complexos
    i0, i1 = CORTES["treino"]
    pedacos = [(a, min(a + BLOCO, i1)) for a in range(i0 + AQUEC, i1, BLOCO)]   # mesmas amostras de ajuste da busca
    c, posto, cond = ajuste_truncado(((base_fom(a, b, K, M, S), y[a:b]) for a, b in pedacos), n_col)

    def prever(nome):                                 # previsão contínua
        j0, j1 = CORTES[nome]
        return np.concatenate([base_fom(a, min(a + BLOCO, j1), K, M, S) @ c for a in range(j0, j1, BLOCO)])
    n_par = 2 * n_col + J + (1 if familia == "FOM" else 0)   # alfa só conta no FOM
    return {"n_coef": n_col, "n_par": n_par, "posto": posto, "log10_kappa": float(np.log10(cond)),
            "y_val": prever("val"), "y_teste": prever("teste")}


# ======================================================================
# Avaliação
# ======================================================================
with open(args.modelos) as f:                         # modelos da etapa 10
    selecionados = json.load(f)["modelos"]
ROTULO = {"menor": "S", "empate": "E", "melhor": "A"}   # mesma nomenclatura do artigo
nome_artigo = lambda m: f"{'FOM' if m['familia'] == 'FOM' else 'BL'}-{ROTULO[m['papel']]}"   # FOM-S, BL-E, ...

j0, j1 = CORTES["teste"]; y_teste = y[j0:j1]          # saída medida no teste
v0, v1 = CORTES["val"]; y_val = y[v0:v1]              # saída medida na validação
resultados = {}                                       # métricas e previsões por modelo

for nome, func in (("Benchmark GMP", gmp_benchmark), ("Causal GMP", gmp_causal)):   # referências
    t0 = time.time()
    r = func()
    r["familia"] = "GMP"
    resultados[nome] = r
    print(f"{nome:14s} ajustado em {time.time() - t0:.0f} s")

for m in selecionados:                                # 3 FOMs e 3 baselines
    t0 = time.time()
    r = avaliar_estrutura(np.array(m["genoma_fom"]), m["familia"])
    r["familia"] = m["familia"]
    r["nmse_val_busca"] = m["nmse_val_db"]           # valor registrado durante a busca (conferência)
    resultados[nome_artigo(m)] = r
    print(f"{nome_artigo(m):14s} ajustado em {time.time() - t0:.0f} s")

for nome, r in resultados.items():                    # NMSE de validação e de teste
    r["nmse_val"] = nmse_db(y_val, r["y_val"])
    r["nmse_teste"] = nmse_db(y_teste, r["y_teste"])

# ---------------- Tabela no terminal ----------------
print(f"\n{'modelo':14s} {'n_par':>6s} {'coef.':>6s} {'posto':>6s} {'log10k':>7s} {'NMSE val':>9s} {'NMSE teste':>11s} {'teste-val':>10s}")
for nome, r in resultados.items():
    print(f"{nome:14s} {r['n_par']:6d} {r['n_coef']:6d} {r['posto']:6d} {r['log10_kappa']:7.2f} "
          f"{r['nmse_val']:9.2f} {r['nmse_teste']:11.2f} {r['nmse_teste'] - r['nmse_val']:10.2f}")
for nome, r in resultados.items():                    # confere se a reavaliação reproduz a busca
    if "nmse_val_busca" in r and abs(r["nmse_val"] - r["nmse_val_busca"]) > 0.01:
        print(f"aviso: {nome} deu NMSE de validação {r['nmse_val']:.3f} dB; na busca foi {r['nmse_val_busca']:.3f} dB")

# ---------------- Gravação ----------------
metricas = {n: {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                for k, v in r.items() if not k.startswith("y_")} for n, r in resultados.items()}   # sem as séries
with open(os.path.join(args.saida, "metricas_teste.json"), "w") as f:   # métricas legíveis
    json.dump(metricas, f, indent=2)
with open(os.path.join(args.saida, "metricas_teste.csv"), "w") as f:    # mesma tabela em CSV
    f.write("modelo,familia,n_par,n_coef,posto,log10_kappa,nmse_val_db,nmse_teste_db\n")
    for n, r in resultados.items():
        f.write(f"{n},{r['familia']},{r['n_par']},{r['n_coef']},{r['posto']},{r['log10_kappa']:.4f},"
                f"{r['nmse_val']:.4f},{r['nmse_teste']:.4f}\n")
np.savez(os.path.join(args.saida, "previsoes_teste.npz"), y_teste=y_teste, x_teste=x[j0:j1],   # séries do teste
         **{n.replace(" ", "_").replace("-", "_"): r["y_teste"] for n, r in resultados.items()})

# ======================================================================
# Figuras
# ======================================================================
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})   # estilo das etapas 07 e 09
TINTA = "#3d3d3d"                                     # textos e referências
COR = {"GMP": "#52514e", "FOM": "#2a78d6", "baseline": "#eb6834"}   # cor por família (identidade)
MARC = {"Benchmark GMP": "s", "Causal GMP": "D", "S": "v", "E": "o", "A": "^"}   # forma por papel
nomes = list(resultados)                              # ordem fixa: GMPs, FOMs, baselines


def marcador(n):                                      # forma do marcador de cada modelo
    return MARC.get(n, MARC.get(n.split("-")[-1], "o"))


def salvar(fig, nome):                                # grava em PDF e PNG
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.saida, f"{nome}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"figura salva: {os.path.join(args.saida, nome)}.pdf")


# 1) NMSE de teste x número de parâmetros
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for n in nomes:
    r = resultados[n]
    ax.scatter(r["n_par"], r["nmse_teste"], s=36, marker=marcador(n), color=COR[r["familia"]],
               edgecolor="white", linewidth=0.8, zorder=3)   # um ponto por modelo
    ax.annotate(n, (r["n_par"], r["nmse_teste"]), xytext=(4, 3), textcoords="offset points",
                fontsize=6.5, color=TINTA)            # rótulo direto
ax.axhline(resultados["Causal GMP"]["nmse_teste"], color=TINTA, lw=0.7, ls="--", zorder=1)   # nível do GMP causal
ax.set_xscale("log"); ax.set_xlabel("Real parameters $n_{\\mathrm{par}}$"); ax.set_ylabel("Test NMSE (dB)")
salvar(fig, "nmse_teste_vs_npar")

# 1b) Detalhe dos modelos com NMSE de teste abaixo de -38 dB (os S ficam muito acima)
fig, ax = plt.subplots(figsize=(3.5, 2.4))
for n in nomes:
    r = resultados[n]
    if r["nmse_teste"] > -37:                         # só os modelos grandes
        continue
    ax.scatter(r["n_par"], r["nmse_teste"], s=36, marker=marcador(n), color=COR[r["familia"]],
               edgecolor="white", linewidth=0.8, zorder=3)
    ax.annotate(n, (r["n_par"], r["nmse_teste"]), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=TINTA)
ax.axhline(resultados["Causal GMP"]["nmse_teste"], color=TINTA, lw=0.7, ls="--", zorder=1)
ax.margins(x=0.12, y=0.15)                            # folga para os rótulos não serem cortados
ax.set_xlabel("Real parameters $n_{\\mathrm{par}}$"); ax.set_ylabel("Test NMSE (dB)")
salvar(fig, "nmse_teste_vs_npar_detalhe")

# 2) Validação x teste (halteres): um segmento por modelo
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for i, n in enumerate(nomes):
    r = resultados[n]
    ax.plot([r["nmse_val"], r["nmse_teste"]], [i, i], color="#bdbdbd", lw=1.2, zorder=1)   # ligação
    ax.scatter(r["nmse_val"], i, s=24, facecolor="white", edgecolor=COR[r["familia"]], linewidth=1.2, zorder=3)   # validação (vazado)
    ax.scatter(r["nmse_teste"], i, s=24, color=COR[r["familia"]], zorder=3)   # teste (cheio)
ax.set_yticks(range(len(nomes))); ax.set_yticklabels(nomes); ax.invert_yaxis()   # um modelo por linha
ax.set_xlabel("NMSE (dB)")
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", mfc="white", mec=TINTA, label="validation"),
                   plt.Line2D([], [], marker="o", ls="", color=TINTA, label="test")],
          loc="lower right", fontsize=7, frameon=False)
salvar(fig, "nmse_val_vs_teste")

# 3) Resposta temporal: |y| medido x |y| previsto num trecho do teste
if args.inicio_janela is None:                        # padrão: trecho centrado no pico de |x| do teste
    pico = int(np.argmax(np.abs(x[j0:j1])))
    ini = int(np.clip(pico - args.janela // 2, 0, len(y_teste) - args.janela))
else:
    ini = int(np.clip(args.inicio_janela, 0, len(y_teste) - args.janela))
idx = np.arange(ini, ini + args.janela)               # índices dentro do teste
t_ns = (idx - ini) / fs * 1e9                         # tempo em ns a partir do início do trecho
n_col_fig = 2                                         # duas colunas de painéis
n_lin_fig = int(np.ceil(len(nomes) / n_col_fig))
fig, axs = plt.subplots(n_lin_fig, n_col_fig, figsize=(7.16, 1.45 * n_lin_fig), sharex=True, sharey=True)
for ax, n in zip(axs.flat, nomes):                    # um painel por modelo
    r = resultados[n]
    ax.plot(t_ns, np.abs(y_teste[idx]), color="#9e9e9e", lw=1.8, label="measured")   # medido
    ax.plot(t_ns, np.abs(r["y_teste"][idx]), color=COR[r["familia"]], lw=0.9, label="model")   # modelo
    ax.fill_between(t_ns, 0, np.abs(y_teste[idx] - r["y_teste"][idx]), color=COR[r["familia"]],
                    alpha=0.25, lw=0, label="$|y-\\hat{y}|$")   # erro instantâneo
    ax.set_title(f"{n} (test NMSE {r['nmse_teste']:.2f} dB)", fontsize=7.5)
for ax in axs.flat[len(nomes):]:                      # painel sobrando fica vazio
    ax.set_visible(False)
for ax in axs[-1]:
    ax.set_xlabel(f"Time (ns), test samples {ini}–{ini + args.janela - 1}")
for ax in axs[:, 0]:
    ax.set_ylabel("$|y|$")
axs.flat[0].legend(loc="upper right", fontsize=6.5, frameon=False, ncol=3)   # legenda única
fig.tight_layout()
salvar(fig, "resposta_temporal_teste")

# 3b) Erro instantâneo |y - y_hat| no mesmo trecho, só para os modelos grandes (escala própria)
grandes = [n for n in nomes if resultados[n]["nmse_teste"] < -37]   # exclui os modelos S
fig, axs = plt.subplots(len(grandes), 1, figsize=(3.5, 0.75 * len(grandes) + 0.5), sharex=True, sharey=True)
for ax, n in zip(np.atleast_1d(axs), grandes):        # um painel por modelo, mesma escala vertical
    r = resultados[n]
    ax.plot(t_ns, np.abs(y_teste[idx] - r["y_teste"][idx]), color=COR[r["familia"]], lw=0.9)   # erro
    ax.text(0.01, 0.92, n, transform=ax.transAxes, fontsize=7, va="top", color=TINTA)   # rótulo direto
np.atleast_1d(axs)[-1].set_xlabel(f"Time (ns), test samples {ini}–{ini + args.janela - 1}")
fig.supylabel("$|y-\\hat{y}|$", fontsize=8)
fig.tight_layout()
salvar(fig, "erro_temporal_teste")

# 4) PSD da saída medida e dos erros dos modelos (eixo de frequência em MHz, centrado)
nper = 512                                            # tamanho do segmento de Welch (curvas mais suaves)
f_hz, P_y = welch(y_teste, fs=fs, nperseg=nper, return_onesided=False, scaling="density")   # saída medida
ordem_f = np.argsort(f_hz)                            # frequências em ordem crescente
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ref = P_y.max()                                       # normaliza pelo pico da saída
ax.plot(f_hz[ordem_f] / 1e6, 10 * np.log10(P_y[ordem_f] / ref), color="#9e9e9e", lw=1.2, label="measured output")
for n in ("Causal GMP", "FOM-E", "BL-E", "FOM-A", "BL-A"):   # erros dos modelos de maior precisão
    if n not in resultados:
        continue
    r = resultados[n]
    _, P_e = welch(y_teste - r["y_teste"], fs=fs, nperseg=nper, return_onesided=False, scaling="density")
    ls = "--" if n.endswith("-A") else "-"            # tracejado para os modelos A
    ax.plot(f_hz[ordem_f] / 1e6, 10 * np.log10(P_e[ordem_f] / ref), color=COR[r["familia"]], lw=0.8, ls=ls,
            label=f"error, {n}")
ax.set_xlabel("Frequency offset (MHz)"); ax.set_ylabel("PSD (dB, rel. output peak)")
ax.set_xlim(-fs / 2e6, fs / 2e6)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=6.5, frameon=False, ncol=2)   # legenda abaixo do eixo
salvar(fig, "psd_erros_teste")

print(f"\nmétricas e figuras gravadas em {args.saida}")