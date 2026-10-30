# ===== Etapa 10: seleção dos modelos para a avaliação no teste =====
# Uso (na pasta do projeto, depois das etapas 07 e 08):
#   python 10_selecionar_modelos.py
# Escolhe 3 soluções da frente agregada do FOM e 3 da frente do baseline alfa = 1:
#   "menor"  -> extremo de menor número de parâmetros (empate decidido pelo NMSE)
#   "empate" -> menor modelo com NMSE de validação <= GMP causal + tolerância (mesmo critério da etapa 07)
#   "melhor" -> extremo de menor NMSE de validação (empate decidido pelo número de parâmetros)
# Grava resultados/modelos_selecionados.npz e .json com n_par, NMSE de validação, log10 kappa,
# genoma original, genoma expandido para os 8 genes do FOM e estrutura decodificada.

import argparse, json, os, sys                        # argumentos, gravação em JSON e caminhos
import numpy as np                                    # vetores

RAIZ = os.path.dirname(os.path.abspath(__file__))     # pasta do projeto

# ---------------- Argumentos ----------------
parser = argparse.ArgumentParser(description="Seleção de 3 FOMs e 3 baselines para o teste")
parser.add_argument("--frente-fom", default=os.path.join(RAIZ, "resultados", "fom_analise", "frente_agregada.npz"))   # etapa 07
parser.add_argument("--baseline", default=os.path.join(RAIZ, "resultados", "baseline", "baseline_semente0.npz"))      # etapa 08
parser.add_argument("--saida", default=os.path.join(RAIZ, "resultados", "modelos_selecionados"))   # sem extensão
parser.add_argument("--ref-nmse", type=float, default=-38.54)   # NMSE de validação do GMP causal (dB)
parser.add_argument("--tolerancia", type=float, default=0.01)   # empate: até ref + tolerância
args = parser.parse_args()                            # lê os argumentos

# Limites do genoma de 8 genes (iguais aos de comum/fom.py; repetidos para não carregar o dataset)
NOMES_FOM = ["J", "K", "M", "log10_L", "alfa", "log10_tau1", "log10_tau2", "log10_tau3"]   # ordem dos genes
LIM_INF = np.array([1, 1, 0, 1.0, 0.05, 0.0, 0.0, 0.0])   # limites inferiores
LIM_SUP = np.array([3, 9, 20, 3.0, 1.0, 4.0, 4.0, 4.0])   # limites superiores
IDX_BASE = [0, 1, 2, 5, 6, 7]                         # posições de J, K, M e taus no genoma de 8 genes
LOG_L_BASE, ALFA_BASE = 1.0, 1.0                      # L = 10 e alfa = 1, fixos no baseline (etapa 08)


def expandir_baseline(g6):                            # genoma de 6 genes -> 8 genes do FOM
    g8 = np.empty(8)                                  # vetor completo
    g8[IDX_BASE] = g6                                 # J, K, M e taus
    g8[3], g8[4] = LOG_L_BASE, ALFA_BASE              # L e alfa fixos
    return g8


def decodificar(g8):                                  # mesma decodificação de comum/fom.py
    g = np.clip(np.asarray(g8, float), LIM_INF, LIM_SUP)   # dentro dos limites
    J = int(round(g[0]))                              # número de estados lentos
    return {"J": J, "K": int(round(g[1])), "M": int(round(g[2])), "L": int(round(10 ** g[3])),
            "alfa": float(g[4]), "taus": np.sort(10 ** g[5:5 + J]).tolist()}


def escolher(F):                                      # índices de "menor", "empate" e "melhor" numa frente
    menor = int(np.lexsort((F[:, 0], F[:, 1]))[0])    # menor n_par; empate pelo NMSE
    melhor = int(np.lexsort((F[:, 1], F[:, 0]))[0])   # menor NMSE; empate pelo n_par
    ok = np.where(F[:, 0] <= args.ref_nmse + args.tolerancia)[0]   # candidatos que empatam com o GMP causal
    empate = int(ok[np.lexsort((F[ok, 0], F[ok, 1]))[0]]) if ok.size else None   # o menor deles
    return {"menor": menor, "empate": empate, "melhor": melhor}


# ---------------- Leitura das duas frentes ----------------
for arq in (args.frente_fom, args.baseline):          # confere se os arquivos existem
    if not os.path.exists(arq):
        sys.exit(f"não encontrei {arq}; rode as etapas 07 e 08 ou indique o caminho")
with np.load(args.frente_fom) as d:                   # frente agregada do FOM (etapa 07)
    G_fom, F_fom, S_fom = d["genoma"], d["objetivos"], d["semente"]   # genomas, objetivos e semente de origem
with np.load(args.baseline) as d:                     # baseline (etapa 08)
    G_base, F_base = d["X_frente"], d["F_frente"]     # primeira frente da semente
    sem_base = int(os.path.basename(args.baseline).split("semente")[1].split(".")[0])   # semente pelo nome do arquivo

# ---------------- Seleção ----------------
modelos = []                                          # lista de modelos escolhidos
for familia, G, F, origem in (("FOM", G_fom, F_fom, S_fom),
                              ("baseline", G_base, F_base, np.full(len(F_base), sem_base))):   # duas frentes
    for papel, i in escolher(F).items():              # três papéis por frente
        if i is None:                                 # nenhuma solução empata com o GMP causal
            print(f"aviso: {familia} sem solução com NMSE <= {args.ref_nmse + args.tolerancia:.2f} dB")
            continue
        g8 = G[i] if familia == "FOM" else expandir_baseline(G[i])   # genoma no formato de avaliar_fom
        modelos.append({"nome": f"{familia}_{papel}", "familia": familia, "papel": papel,   # identificação
                        "semente": int(origem[i]),    # semente de onde veio a solução
                        "n_par": int(F[i, 1]), "nmse_val_db": float(F[i, 0]), "log10_kappa": float(F[i, 2]),   # objetivos
                        "genoma": G[i].tolist(),      # genoma como foi buscado (8 ou 6 genes)
                        "genoma_fom": g8.tolist(),    # genoma de 8 genes, pronto para comum.fom.avaliar_fom
                        "estrutura": decodificar(g8)})   # J, K, M, L, alfa e taus

# ---------------- Relatório ----------------
print(f"critério de empate: NMSE de validação <= {args.ref_nmse + args.tolerancia:.2f} dB "
      f"(GMP causal {args.ref_nmse:.2f} dB + {args.tolerancia} dB)\n")
print(f"{'modelo':16s} {'sem.':>4s} {'n_par':>6s} {'NMSE val':>9s} {'log10k':>7s} | "
      f"{'J':>2s} {'K':>2s} {'M':>3s} {'L':>5s} {'alfa':>6s}  taus | genoma")
for m in modelos:                                     # uma linha por modelo
    e = m["estrutura"]                                # estrutura decodificada
    print(f"{m['nome']:16s} {m['semente']:4d} {m['n_par']:6d} {m['nmse_val_db']:9.2f} {m['log10_kappa']:7.2f} | "
          f"{e['J']:2d} {e['K']:2d} {e['M']:3d} {e['L']:5d} {e['alfa']:6.3f}  {np.round(e['taus'], 2).tolist()} | "
          f"{np.round(m['genoma'], 4).tolist()}")

# ---------------- Gravação ----------------
criterio = {"ref_nmse": args.ref_nmse, "tolerancia": args.tolerancia,
            "frente_fom": args.frente_fom, "baseline": args.baseline}   # de onde e como os modelos foram escolhidos
with open(args.saida + ".json", "w") as f:            # versão legível
    json.dump({"criterio": criterio, "nomes_genes_fom": NOMES_FOM, "modelos": modelos}, f, indent=2)
np.savez(args.saida + ".npz",                         # versão para NumPy
         nome=np.array([m["nome"] for m in modelos]), familia=np.array([m["familia"] for m in modelos]),
         papel=np.array([m["papel"] for m in modelos]), semente=np.array([m["semente"] for m in modelos]),
         n_par=np.array([m["n_par"] for m in modelos]), nmse_val_db=np.array([m["nmse_val_db"] for m in modelos]),
         log10_kappa=np.array([m["log10_kappa"] for m in modelos]),
         genoma_fom=np.array([m["genoma_fom"] for m in modelos]),   # 8 genes para todos os modelos
         nomes_genes_fom=np.array(NOMES_FOM), **{k: v for k, v in criterio.items() if k.startswith("ref") or k == "tolerancia"})
print(f"\nmodelos salvos em {args.saida}.npz e {args.saida}.json")