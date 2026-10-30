# ===== Etapa 12: EVM (com demodulação das portadoras LTE) e ACPR =====
# Uso (na pasta do projeto, depois da etapa 10):
#   python 12_evm_acpr.py
# Modelos: os 6 selecionados na etapa 10, o GMP de benchmark e o GMP causal, todos ajustados só no treino.
#
# EVM. O sinal tem 5 portadoras LTE (TM3.1a, 256-QAM) com FFT de 32 768 amostras por símbolo OFDM
# (spec.json). Cada portadora é deslocada para a frequência zero e isolada por um filtro passa-faixa;
# os símbolos são localizados pela correlação do prefixo cíclico na entrada x, e a FFT de cada símbolo
# dá as 1 200 subportadoras de dados. O registro inteiro tem só dois símbolos completos:
#   símbolo 1: inteiramente no treino;  símbolo 2: ~33 % treino, ~60 % validação, ~7 % teste.
# O conjunto de teste (19 662 amostras) é menor que uma FFT (32 768), então nenhum símbolo cabe nele.
# O EVM principal é o do símbolo 2; o do símbolo 1 é impresso para comparação.
#   EVM do PA   : subportadoras da saída (medida ou modelada) contra as da entrada x, com um ganho
#                 complexo por portadora (correção linear), em % e dB.
#   EVM do erro : subportadoras do modelo contra as da saída medida, sem ganho (erro do modelo
#                 no domínio da constelação).
# ACPR. Calculado só no conjunto de teste, pela PSD de Welch: potência em 36 MHz (1 200 x 30 kHz)
# nos canais adjacentes (±120 MHz) e alternados (±160 MHz) dividida pela potência da portadora
# vizinha (±80 MHz). Reporta o ACPR da saída medida, o de cada modelo e a diferença.

import argparse, json, os, sys, time                  # argumentos, gravação, caminhos e tempo
import numpy as np                                    # vetores
import matplotlib                                     # biblioteca de gráficos
matplotlib.use("Agg")                                 # gera as figuras sem abrir janela
import matplotlib.pyplot as plt                       # gráficos
from scipy.signal import welch, find_peaks            # PSD e detecção dos picos de correlação

RAIZ = os.path.dirname(os.path.abspath(__file__))     # pasta do projeto
sys.path.insert(0, RAIZ)                              # garante que comum/ seja encontrado
from comum.dados import x, y, N, CORTES, fs, spec, PASTA_RESULTADOS   # registro e especificação
from comum.avaliacao import modelo_gmp_benchmark, modelo_gmp_causal, modelo_fom   # modelos

# ---------------- Argumentos ----------------
parser = argparse.ArgumentParser(description="EVM e ACPR dos modelos")
parser.add_argument("--modelos", default=str(PASTA_RESULTADOS / "modelos_selecionados.json"))   # etapa 10
parser.add_argument("--saida", default=str(PASTA_RESULTADOS / "evm_acpr"))   # pasta de saída
parser.add_argument("--nperseg-acpr", type=int, default=4096)   # segmento de Welch do ACPR (240 kHz por bin)
args = parser.parse_args()
os.makedirs(args.saida, exist_ok=True)
if not os.path.exists(args.modelos):
    sys.exit(f"não encontrei {args.modelos}; rode a etapa 10 ou indique --modelos")

# ---------------- Parâmetros do sinal (spec.json) ----------------
NFFT = int(spec["ofdm_nfft"])                         # 32 768 amostras por símbolo
CP = int(spec["cp_other"])                            # prefixo cíclico usado na sincronização (2 304)
SCS = float(spec["scs"])                              # 30 kHz entre subportadoras
N_SC = 1200                                           # subportadoras de dados de uma portadora LTE de 20 MHz (±600)
BW_SUB = float(spec["bw_sub_ch"])                     # 40 MHz entre centros de portadora
N_CH = int(spec["n_sub_ch"])                          # 5 portadoras
F_PORT = [(i - (N_CH - 1) / 2) * BW_SUB for i in range(N_CH)]   # centros: -80, -40, 0, 40, 80 MHz
BW_INT = N_SC * SCS                                   # banda de integração de um canal: 36 MHz


# ======================================================================
# Demodulação OFDM
# ======================================================================
def isolar(sinal, fc):                                # desloca a portadora fc para 0 Hz e filtra ±BW_SUB/2
    n = np.arange(len(sinal))
    s = sinal * np.exp(-2j * np.pi * fc * n / fs)     # portadora em banda base
    Fs = np.fft.fftshift(np.fft.fft(s))               # espectro do registro inteiro
    meio = len(s) // 2
    h = int(round(BW_SUB / 2 * len(s) / fs))          # meia largura do filtro em bins
    filtro = np.zeros_like(Fs)
    filtro[meio - h:meio + h + 1] = Fs[meio - h:meio + h + 1]   # passa-faixa retangular
    return np.fft.ifft(np.fft.ifftshift(filtro))      # portadora isolada no tempo


def sincronizar(sinal_bb):                            # inícios de prefixo cíclico por correlação (soma acumulada)
    M = len(sinal_bb) - NFFT
    prod = sinal_bb[:M] * np.conj(sinal_bb[NFFT:])    # s[n] s*[n + NFFT]
    cs = np.concatenate([[0], np.cumsum(prod)])
    corr = np.abs(cs[CP:] - cs[:-CP])                 # soma em janelas de CP amostras
    e = np.concatenate([[0], np.cumsum(np.abs(sinal_bb) ** 2)])
    e1 = e[CP:M + 1] - e[:M - CP + 1]                 # energia da janela n..n+CP
    e2 = e[NFFT + CP:] - e[NFFT:len(sinal_bb) - CP + 1]   # energia da janela n+NFFT..n+NFFT+CP
    r = corr[:len(e1)] / np.sqrt(e1 * e2[:len(e1)] + 1e-30)   # correlação normalizada
    picos, _ = find_peaks(r, height=0.5, distance=NFFT // 2)
    return [int(p) for p in picos if p + CP // 2 + NFFT <= len(sinal_bb)]   # só símbolos completos


def subportadoras(sinal_bb, ini):                     # FFT de um símbolo e as 1 200 subportadoras (sem DC)
    F = np.fft.fftshift(np.fft.fft(sinal_bb[ini:ini + NFFT]))
    dc = NFFT // 2
    return np.concatenate([F[dc - N_SC // 2:dc], F[dc + 1:dc + N_SC // 2 + 1]])


# Localização dos símbolos a partir da entrada (sinal limpo), na portadora central
x_central = isolar(x, 0.0)
inicios_cp = sincronizar(x_central)                   # inícios de CP dos símbolos completos


def curtose(sc):                                      # curtose de I e Q: mínima quando a FFT está alinhada ao símbolo
    r, i = sc.real, sc.imag
    return np.mean(r ** 4) / np.mean(r ** 2) ** 2 + np.mean(i ** 4) / np.mean(i ** 2) ** 2


JANELAS = []                                          # início da FFT de cada símbolo
for p in inicios_cp:                                  # ajuste fino em ±40 amostras ao redor do fim do CP (como no OpenDPD)
    cand = [p + CP + off for off in range(-40, 41) if p + CP + off + NFFT <= N]
    JANELAS.append(min(cand, key=lambda ini: curtose(subportadoras(x_central, ini))))


def fracao_conjuntos(ini):                            # fração da janela da FFT em cada conjunto
    return {nome: max(0, min(b, ini + NFFT) - max(a, ini)) / NFFT for nome, (a, b) in CORTES.items()}


print("símbolos OFDM completos no registro:")
for k, ini in enumerate(JANELAS):
    fr = fracao_conjuntos(ini)
    print(f"  símbolo {k + 1}: FFT nas amostras {ini}-{ini + NFFT - 1} | "
          + ", ".join(f"{n} {100 * v:.0f} %" for n, v in fr.items()))


def constelacoes(sinal):                              # subportadoras de cada símbolo e portadora: [símbolo][portadora]
    bb = [isolar(sinal, fc) for fc in F_PORT]
    return [[subportadoras(b, ini) for b in bb] for ini in JANELAS]


X_ref = constelacoes(x)                               # subportadoras da entrada (referência)
MASCARA = [[np.abs(Xc) ** 2 > 0.01 * np.median(np.abs(Xc) ** 2) for Xc in Xs] for Xs in X_ref]   # descarta bins vazios


def evm_pa(Y):                                        # EVM das subportadoras Y contra a entrada, um ganho por portadora
    res = []
    for Ys, Xs, Ms in zip(Y, X_ref, MASCARA):         # cada símbolo
        num = den = 0.0
        for Yc, Xc, m in zip(Ys, Xs, Ms):             # cada portadora
            g = np.vdot(Xc[m], Yc[m]) / np.vdot(Xc[m], Xc[m])   # ganho complexo por mínimos quadrados
            num += np.sum(np.abs(Yc[m] - g * Xc[m]) ** 2)
            den += np.sum(np.abs(g * Xc[m]) ** 2)
        res.append(np.sqrt(num / den))
    return res                                        # EVM rms (fração) por símbolo


def evm_erro(Yh, Ym):                                 # erro do modelo na constelação, contra a saída medida
    res = []
    for Yhs, Yms, Ms in zip(Yh, Ym, MASCARA):
        num = sum(np.sum(np.abs(a[m] - b[m]) ** 2) for a, b, m in zip(Yhs, Yms, Ms))
        den = sum(np.sum(np.abs(b[m]) ** 2) for b, m in zip(Yms, Ms))
        res.append(np.sqrt(num / den))
    return res


# ======================================================================
# ACPR no conjunto de teste
# ======================================================================
def potencia_canal(f, P, fc):                         # integra a PSD em BW_INT ao redor de fc
    m = np.abs(f - fc) <= BW_INT / 2
    return np.sum(P[m])


def acpr(sinal):                                      # ACPR adjacente e alternado, lados inferior e superior (dB)
    f, P = welch(sinal, fs=fs, nperseg=args.nperseg_acpr, return_onesided=False, scaling="density")
    p_inf, p_sup = potencia_canal(f, P, F_PORT[0]), potencia_canal(f, P, F_PORT[-1])   # portadoras das bordas
    d = {}
    for nome, off in (("adj", BW_SUB), ("alt", 2 * BW_SUB)):   # ±120 MHz e ±160 MHz
        d[f"{nome}_inf"] = 10 * np.log10(potencia_canal(f, P, F_PORT[0] - off) / p_inf)
        d[f"{nome}_sup"] = 10 * np.log10(potencia_canal(f, P, F_PORT[-1] + off) / p_sup)
    return d


# ======================================================================
# Modelos
# ======================================================================
with open(args.modelos) as f:
    selecionados = json.load(f)["modelos"]
ROTULO = {"menor": "S", "empate": "E", "melhor": "A"}
modelos = {"Benchmark GMP": modelo_gmp_benchmark, "Causal GMP": modelo_gmp_causal}   # construtores
for m in selecionados:
    nome = f"{'FOM' if m['familia'] == 'FOM' else 'BL'}-{ROTULO[m['papel']]}"
    modelos[nome] = (lambda g=np.array(m["genoma_fom"]): modelo_fom(g))
FAMILIA = {n: ("GMP" if "GMP" in n else "FOM" if n.startswith("FOM") else "baseline") for n in modelos}

j0, j1 = CORTES["teste"]                              # limites do teste
Y_med = constelacoes(y)                               # subportadoras da saída medida
evm_med = evm_pa(Y_med)                               # EVM do PA medido
acpr_med = acpr(y[j0:j1])                             # ACPR medido no teste
resultados = {"Measured": {"evm_pa": evm_med, "acpr": acpr_med}}
saidas = {}                                           # saídas previstas (registro inteiro)

for nome, construir in modelos.items():
    t0 = time.time()
    prever = construir()                              # ajusta no treino
    y_hat = prever(0, N)                              # saída prevista no registro inteiro
    saidas[nome] = y_hat
    Y_hat = constelacoes(y_hat)
    resultados[nome] = {"evm_pa": evm_pa(Y_hat), "evm_erro": evm_erro(Y_hat, Y_med), "acpr": acpr(y_hat[j0:j1])}
    print(f"{nome:14s} pronto em {time.time() - t0:.0f} s")

# ---------------- Relatório ----------------
pct = lambda v: 100 * v                               # fração -> %
db = lambda v: 20 * np.log10(v)                       # fração -> dB
S2 = len(JANELAS) - 1                                 # índice do último símbolo completo (símbolo 2)
print(f"\nEVM no símbolo {S2 + 1} (principal) e no símbolo 1 (só treino)")
print(f"{'modelo':14s} {'EVM PA s2':>10s} {'(dB)':>7s} {'dEVM s2':>8s} {'EVM erro s2':>12s} | {'EVM PA s1':>10s} {'EVM erro s1':>12s}")
for nome, r in resultados.items():
    e = r["evm_pa"]
    linha = f"{nome:14s} {pct(e[S2]):9.3f}% {db(e[S2]):7.2f} "
    linha += f"{pct(e[S2]) - pct(evm_med[S2]):+7.3f}  " if nome != "Measured" else f"{'':8s}  "
    linha += f"{db(r['evm_erro'][S2]):10.2f}dB | " if "evm_erro" in r else f"{'':12s} | "
    linha += f"{pct(e[0]):9.3f}% " + (f"{db(r['evm_erro'][0]):10.2f}dB" if "evm_erro" in r else "")
    print(linha)

print(f"\nACPR no conjunto de teste (dB, relativo à portadora vizinha)")
print(f"{'modelo':14s} {'adj inf':>8s} {'adj sup':>8s} {'alt inf':>8s} {'alt sup':>8s} | {'dadj inf':>9s} {'dadj sup':>9s}")
for nome, r in resultados.items():
    a = r["acpr"]
    linha = f"{nome:14s} {a['adj_inf']:8.2f} {a['adj_sup']:8.2f} {a['alt_inf']:8.2f} {a['alt_sup']:8.2f} | "
    if nome != "Measured":
        linha += f"{a['adj_inf'] - acpr_med['adj_inf']:+9.2f} {a['adj_sup'] - acpr_med['adj_sup']:+9.2f}"
    print(linha)

# ---------------- Gravação ----------------
saida_json = {"simbolos": [{"inicio_fft": ini, "fracao": fracao_conjuntos(ini)} for ini in JANELAS],
              "modelos": {n: {"evm_pa_pct": [pct(v) for v in r["evm_pa"]],
                              "evm_pa_db": [db(v) for v in r["evm_pa"]],
                              **({"evm_erro_db": [db(v) for v in r["evm_erro"]]} if "evm_erro" in r else {}),
                              "acpr_teste_db": r["acpr"]} for n, r in resultados.items()}}
with open(os.path.join(args.saida, "evm_acpr.json"), "w") as f:
    json.dump(saida_json, f, indent=2, default=float)
with open(os.path.join(args.saida, "evm_acpr.csv"), "w") as f:
    f.write("modelo,evm_pa_s2_pct,evm_pa_s2_db,evm_erro_s2_db,evm_pa_s1_pct,evm_erro_s1_db,"
            "acpr_adj_inf_db,acpr_adj_sup_db,acpr_alt_inf_db,acpr_alt_sup_db\n")
    for n, r in resultados.items():
        ee = r.get("evm_erro", [np.nan] * len(JANELAS))
        a = r["acpr"]
        f.write(f"{n},{pct(r['evm_pa'][S2]):.4f},{db(r['evm_pa'][S2]):.4f},{db(ee[S2]):.4f},"
                f"{pct(r['evm_pa'][0]):.4f},{db(ee[0]):.4f},"
                f"{a['adj_inf']:.4f},{a['adj_sup']:.4f},{a['alt_inf']:.4f},{a['alt_sup']:.4f}\n")

# ======================================================================
# Figuras
# ======================================================================
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
TINTA = "#3d3d3d"
COR = {"GMP": "#52514e", "FOM": "#2a78d6", "baseline": "#eb6834"}
nomes_mod = list(modelos)


def salvar(fig, nome):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.saida, f"{nome}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"figura salva: {os.path.join(args.saida, nome)}.pdf")


# 1) EVM do PA (símbolo 2): medido como linha de referência, um ponto por modelo
fig, ax = plt.subplots(figsize=(3.5, 2.5))
for i, n in enumerate(nomes_mod):
    ax.scatter(pct(resultados[n]["evm_pa"][S2]), i, s=30, color=COR[FAMILIA[n]], zorder=3)
ax.axvline(pct(evm_med[S2]), color=TINTA, lw=0.8, ls="--", zorder=1)
ax.annotate("measured", (pct(evm_med[S2]), len(nomes_mod) - 0.5), xytext=(3, 0), textcoords="offset points",
            fontsize=7, color=TINTA)
ax.set_yticks(range(len(nomes_mod))); ax.set_yticklabels(nomes_mod); ax.invert_yaxis()
ax.set_xlabel(f"EVM, symbol {S2 + 1} (%)")
salvar(fig, "evm_modelos")

# 2) Constelação da portadora central, símbolo 2: medida x modelo (GMP causal e FOM-E)
Yc_med = Y_med[S2][N_CH // 2]; Xc = X_ref[S2][N_CH // 2]; mc = MASCARA[S2][N_CH // 2]
g = np.vdot(Xc[mc], Yc_med[mc]) / np.vdot(Xc[mc], Xc[mc])   # ganho da saída medida (normalização comum)
mostrar = [n for n in ("Causal GMP", "FOM-E") if n in saidas]
fig, axs = plt.subplots(1, len(mostrar), figsize=(3.5, 1.9), sharex=True, sharey=True, squeeze=False)
for ax, n in zip(axs[0], mostrar):
    Yh = constelacoes(saidas[n])[S2][N_CH // 2]       # subportadoras do modelo
    esc = np.sqrt(np.mean(np.abs(Xc[mc]) ** 2))       # normaliza pela potência rms da referência
    ax.scatter((Yc_med[mc] / g).real / esc, (Yc_med[mc] / g).imag / esc, s=4, color="#bdbdbd", lw=0, label="measured")
    ax.scatter((Yh[mc] / g).real / esc, (Yh[mc] / g).imag / esc, s=1.5, color=COR[FAMILIA[n]], lw=0, label=n)
    ax.set_title(n, fontsize=7.5); ax.set_aspect("equal")
axs[0, 0].set_ylabel("Q"); [a.set_xlabel("I") for a in axs[0]]
salvar(fig, "constelacao_portadora_central")

# 3) PSD do teste com os canais usados no ACPR
f, P_m = welch(y[j0:j1], fs=fs, nperseg=args.nperseg_acpr, return_onesided=False, scaling="density")
o = np.argsort(f); ref = P_m.max()
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for fc in F_PORT:                                     # portadoras
    ax.axvspan((fc - BW_INT / 2) / 1e6, (fc + BW_INT / 2) / 1e6, color="#eeeeee", lw=0, zorder=0)
for off in (BW_SUB, 2 * BW_SUB):                      # canais adjacentes e alternados
    for fc in (F_PORT[0] - off, F_PORT[-1] + off):
        ax.axvspan((fc - BW_INT / 2) / 1e6, (fc + BW_INT / 2) / 1e6, color="#fde9dc", lw=0, zorder=0)
ax.plot(f[o] / 1e6, 10 * np.log10(P_m[o] / ref), color="#9e9e9e", lw=1.4, label="measured")
for n in ("Causal GMP", "FOM-E", "BL-E"):
    if n in saidas:
        _, P_h = welch(saidas[n][j0:j1], fs=fs, nperseg=args.nperseg_acpr, return_onesided=False, scaling="density")
        ax.plot(f[o] / 1e6, 10 * np.log10(P_h[o] / ref), color=COR[FAMILIA[n]], lw=0.8,
                ls="--" if n == "BL-E" else "-", label=n)
ax.set_xlim(-250, 250); ax.set_ylim(-70, 5)
ax.set_xlabel("Frequency offset (MHz)"); ax.set_ylabel("PSD (dB, rel. peak)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=6.5, frameon=False, ncol=4)
salvar(fig, "psd_acpr_teste")

print(f"\nresultados gravados em {args.saida}")