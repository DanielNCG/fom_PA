# ===== Etapa 08: baseline com alfa = 1 (estados lentos exponenciais) =====
# Uso (na pasta do projeto):
#   python 08_baseline_alfa1.py                   # roda a semente 0 (N = 40, G = 50)
#   python 08_baseline_alfa1.py --teste-rapido    # busca minúscula (N = 8, G = 2) para conferir o encadeamento
#   python 08_baseline_alfa1.py --sementes 0 1    # outras sementes, se houver tempo
# Mesmo modelo, mesmo ajuste e mesmo NSGA-II da etapa 06, com alfa fixo em 1.
# Com alfa = 1 os pesos de Grünwald-Letnikov são w_0 = 1, w_1 = -1 e w_i = 0 para i >= 2:
# a recursão vira o Euler implícito de 1a ordem e o truncamento L não altera nada.
# Por isso alfa e L saem do genoma, que fica com 6 genes: J, K, M, log10_tau1..3.
# Retomada: sementes já concluídas são puladas; Ctrl+C deixa a semente em andamento pendente.

import argparse, os, sys, json, time, traceback      # argumentos, arquivos, registro, tempo e erros

# ---------------- Argumentos ----------------
parser = argparse.ArgumentParser(description="Baseline alfa = 1 do modelo de estados lentos")   # leitor de argumentos
parser.add_argument("--sementes", type=int, nargs="+", default=[0])   # padrão: só a semente 0
parser.add_argument("--teste-rapido", action="store_true")   # N = 8, G = 2, pasta separada
parser.add_argument("--pasta", default=None)          # pasta de saída (padrão: resultados/baseline)
parser.add_argument("--threads", type=int, default=None)   # threads de álgebra linear
parser.add_argument("--ref-nmse", type=float, default=-38.54)   # NMSE de validação do GMP causal (dB)
parser.add_argument("--tolerancia", type=float, default=0.01)   # empate: até ref + tolerância
args = parser.parse_args()                            # lê os argumentos
if args.threads:                                      # fixa as threads antes de importar o NumPy
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(args.threads)

import numpy as np                                    # vetores
from comum.dados import PASTA_RESULTADOS              # pasta de resultados do projeto
from comum.fom import avaliar_fom, pesos_gl, LIM_INF as LIM_INF_FOM, LIM_SUP as LIM_SUP_FOM, J_MAX   # modelo da etapa 06
from comum.nsga2 import nsga2                         # mesmo NSGA-II

# ---------------- Genoma do baseline ----------------
ALFA_FIXO = 1.0                                       # ordem inteira: estados exponenciais
LOG_L_FIXO = 1.0                                      # L = 10; com alfa = 1 qualquer L >= 1 dá o mesmo estado
assert np.all(pesos_gl(ALFA_FIXO, 10)[2:] == 0.0)     # confere que os pesos depois de w_1 são nulos
IDX_GENES = [0, 1, 2] + [5 + j for j in range(J_MAX)]   # posições de J, K, M e dos taus no genoma do FOM
NOMES_GENES = ["J", "K", "M"] + [f"log10_tau{j+1}" for j in range(J_MAX)]   # 6 genes
LIM_INF = LIM_INF_FOM[IDX_GENES]                      # mesmos limites da busca fracionária
LIM_SUP = LIM_SUP_FOM[IDX_GENES]
N_GENES = len(NOMES_GENES)                            # 6

# ---------------- Configuração (mesmos hiperparâmetros da busca fracionária) ----------------
TESTE_RAPIDO = args.teste_rapido                      # modo de conferência
N_POP = 8 if TESTE_RAPIDO else 40                     # tamanho da população
N_GER = 2 if TESTE_RAPIDO else 50                     # número de gerações
PC, ETA_C = 0.9, 20.0                                 # cruzamento SBX
PM, ETA_M = 1.0 / N_GENES, 20.0                       # mutação polinomial: P_m = 1 / número de genes = 1/6
PASTA_SAIDA = (args.pasta or str(PASTA_RESULTADOS / "baseline")) + ("_teste" if TESTE_RAPIDO else "")   # saída
os.makedirs(PASTA_SAIDA, exist_ok=True)               # cria a pasta
PENALIDADE_F1, PENALIDADE_F3 = 100.0, 17.0            # mesmas penalidades da etapa 06
CHAVES = ["X", "F", "X_frente", "F_frente", "historico_X", "historico_F", "duracao_s", "N", "G"]   # conteúdo obrigatório


def genoma_fom(g):                                    # expande o genoma de 6 genes para os 8 genes do FOM
    g8 = np.empty(len(LIM_INF_FOM))                   # vetor completo
    g8[IDX_GENES] = g                                 # J, K, M e taus vindos da busca
    g8[3], g8[4] = LOG_L_FIXO, ALFA_FIXO              # L e alfa fixos
    return g8


def decodificar_base(g):                              # estrutura do baseline (sem L e alfa)
    g = np.clip(np.asarray(g, float), LIM_INF, LIM_SUP)   # dentro dos limites
    J = int(round(g[0]))                              # número de estados
    return {"J": J, "K": int(round(g[1])), "M": int(round(g[2])), "taus": np.sort(10 ** g[3:3 + J])}


def n_par_base(th):                                   # parâmetros reais: coeficientes + J constantes de tempo (alfa não é livre)
    return 2 * th["K"] * (th["M"] + 1) * (th["J"] + 1) + th["J"]


# ---------------- Arquivos e verificação de sementes concluídas ----------------
def arquivo_resultado(s):                             # resultado principal de uma semente
    return os.path.join(PASTA_SAIDA, f"baseline_semente{s}.npz")


def arquivo_registro(s):                              # registro detalhado de uma semente
    return os.path.join(PASTA_SAIDA, f"baseline_semente{s}_registro.json")


def situacao_semente(s):                              # ("concluída" | "pendente" | "inválida", motivo)
    if not os.path.exists(arquivo_resultado(s)):      # nunca terminou
        return "pendente", "sem arquivo"
    try:                                              # confere se o arquivo está completo
        with np.load(arquivo_resultado(s)) as d:
            if [k for k in CHAVES if k not in d.files]:   # chaves ausentes
                return "inválida", "chaves ausentes"
            if int(d["N"]) != N_POP or int(d["G"]) != N_GER:   # outra configuração
                return "inválida", "configuração diferente"
            if d["historico_F"].shape[0] != N_POP * (N_GER + 1):   # execução incompleta
                return "inválida", "histórico incompleto"
    except Exception as erro:                         # arquivo corrompido
        return "inválida", f"não foi possível ler ({erro})"
    if not os.path.exists(arquivo_registro(s)):       # registro também é necessário
        return "inválida", "falta o registro .json"
    return "concluída", ""


def salvar_npz(caminho, **dados):                     # gravação atômica do .npz
    tmp = caminho.replace(".npz", ".tmp.npz")         # temporário com extensão .npz
    np.savez(tmp, **dados)                            # escreve
    os.replace(tmp, caminho)                          # renomeia de uma vez


def salvar_json(caminho, dados):                      # gravação atômica do .json
    with open(caminho + ".tmp", "w") as f:            # escreve no temporário
        json.dump(dados, f)
    os.replace(caminho + ".tmp", caminho)             # renomeia de uma vez


# ---------------- Função de objetivos ----------------
registro = []                                         # detalhes de cada avaliação da semente atual


def objetivos_base(g):                                # genoma de 6 genes -> [f1, f2, f3]
    th = decodificar_base(g)                          # estrutura
    try:                                              # mesma proteção da etapa 06
        obj, info = avaliar_fom(genoma_fom(g))        # avaliação com alfa = 1 e L = 10
        obj = np.array([obj[0], n_par_base(th), obj[2]])   # f2 sem alfa, que aqui não é parâmetro livre
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as erro:   # falha numérica
        obj = np.array([PENALIDADE_F1, n_par_base(th), PENALIDADE_F3])   # penalidade
        info = {**th, "n_coef": None, "posto": None, "tempo_s": None, "erro": str(erro)}
    registro.append({"genoma": np.asarray(g, float).tolist(), "objetivos": obj.tolist(),   # genoma e objetivos
                     **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()}})   # detalhes
    return obj                                        # vetor para o NSGA-II


def imprimir_solucoes(F, X, titulo):                  # tabela ordenada por número de parâmetros
    print(titulo)
    print(f"{'NMSE val':>9s} {'n_par':>6s} {'log10k':>7s} | {'J':>2s} {'K':>2s} {'M':>3s}  taus")
    for i in np.lexsort((F[:, 0], F[:, 1])):          # por n_par, empate pelo NMSE
        th = decodificar_base(X[i])                   # estrutura
        print(f"{F[i, 0]:9.2f} {int(F[i, 1]):6d} {F[i, 2]:7.2f} | {th['J']:2d} {th['K']:2d} {th['M']:3d}  {np.round(th['taus'], 2)}")


# ---------------- Execução ----------------
print(f"baseline alfa = 1 | genes: {NOMES_GENES} | N = {N_POP}, G = {N_GER}, P_m = 1/{N_GENES} | pasta: {PASTA_SAIDA}")
pendentes = [s for s in args.sementes if situacao_semente(s)[0] != "concluída"]   # o que falta rodar
for s in args.sementes:                               # situação de cada semente pedida
    print(f"  semente {s}: {situacao_semente(s)[0]}")

if pendentes:                                         # aquecimento: compila o numba antes de medir tempos
    objetivos_base(np.array([1, 2, 0, 1.0, 0.0, 0.0])); registro.clear()

for s in pendentes:                                   # roda as sementes pendentes
    registro.clear()                                  # registro novo
    t0 = time.time()                                  # início
    print(f"\n===== baseline, semente {s} =====", flush=True)
    try:                                              # uma falha não derruba as outras sementes
        res = nsga2(objetivos_base, LIM_INF, LIM_SUP, N=N_POP, G=N_GER, pc=PC, eta_c=ETA_C,
                    pm=PM, eta_m=ETA_M, semente=s, verbose=True)   # mesma busca da etapa 06
        duracao = time.time() - t0                    # duração total
        salvar_json(arquivo_registro(s), registro)    # registro primeiro
        salvar_npz(arquivo_resultado(s), X=res["X"], F=res["F"], X_frente=res["X_frente"],
                   F_frente=res["F_frente"], historico_X=res["historico_X"], historico_F=res["historico_F"],
                   duracao_s=duracao, N=N_POP, G=N_GER, nomes_genes=np.array(NOMES_GENES),
                   alfa=ALFA_FIXO)                    # .npz por último
        tempos = [r["tempo_s"] for r in registro if r.get("tempo_s")]   # tempos de avaliação
        print(f"semente {s} concluída em {duracao / 60:.1f} min; tempo médio por avaliação {np.mean(tempos):.2f} s")
    except KeyboardInterrupt:                         # interrupção manual
        print(f"\ninterrompido durante a semente {s}; ela continua pendente")
        break
    except Exception as erro:                         # outra falha: registra e segue
        print(f"semente {s} FALHOU: {erro!r}")
        traceback.print_exc()

# ---------------- Resumo das sementes concluídas ----------------
limite = args.ref_nmse + args.tolerancia              # critério de empate com o GMP causal
for s in args.sementes:                               # só o que terminou
    if situacao_semente(s)[0] != "concluída":
        continue
    with np.load(arquivo_resultado(s)) as d:          # carrega a semente
        F, X = d["F_frente"], d["X_frente"]
    imprimir_solucoes(F, X, f"\n--- baseline, semente {s}: primeira frente ({len(F)} soluções) ---")
    ok = F[:, 0] <= limite                            # soluções que empatam ou superam o GMP causal
    print(f"melhor NMSE de validação: {F[:, 0].min():.2f} dB | soluções <= {limite:.2f} dB: {int(ok.sum())}"
          + (f", a menor com {int(F[ok, 1].min())} parâmetros" if ok.any() else ""))
    o = np.lexsort((F[ok, 0], F[ok, 1]))              # selecionadas por n_par
    salvar_npz(os.path.join(PASTA_SAIDA, f"baseline_semente{s}_vs_gmp_causal.npz"),
               genoma=X[ok][o], objetivos=F[ok][o], ref_nmse=args.ref_nmse, tolerancia=args.tolerancia,
               nomes_genes=np.array(NOMES_GENES))     # exporta como na etapa 07