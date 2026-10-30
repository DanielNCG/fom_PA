# ===== Etapa 06 (bloco 10): busca do NSGA-II sobre o modelo fracionário =====
# Uso (na pasta do projeto):
#   python 06_busca_fom.py --situacao                 # só mostra quais sementes já terminaram
#   python 06_busca_fom.py                            # roda as sementes pendentes de 0 a 9, uma de cada vez
#   python 06_busca_fom.py --sementes 3 4             # roda só as sementes 3 e 4 (se estiverem pendentes)
#   python 06_busca_fom.py --processos 4              # roda 4 sementes ao mesmo tempo, cada uma num processo
#   python 06_busca_fom.py --teste-rapido             # busca minúscula (N = 8, G = 2) para conferir o encadeamento
# Retomada: antes de rodar, o script verifica quais sementes já têm resultado válido salvo
# e executa só as que faltam. Uma semente que falhe é registrada e não interrompe as demais.
# Interromper com Ctrl+C é seguro: a semente em andamento continua pendente e é refeita depois.

import argparse, os, sys, json, time, traceback, subprocess   # argumentos, arquivos, registro, tempo, erros e subprocessos

# ---------------- Argumentos de linha de comando ----------------
parser = argparse.ArgumentParser(description="Busca NSGA-II sobre o modelo fracionário (FOM)")   # leitor de argumentos
parser.add_argument("--sementes", type=int, nargs="+", default=None,
                    help="sementes a considerar (padrão: 0 a 9; no teste rápido, 0 e 1)")   # lista de sementes
parser.add_argument("--teste-rapido", action="store_true",
                    help="N = 8 e G = 2, gravando numa pasta separada")   # modo de conferência
parser.add_argument("--pasta", default=None,
                    help="pasta de saída (padrão: resultados/fom)")   # onde gravar os resultados
parser.add_argument("--processos", type=int, default=1,
                    help="quantas sementes rodar em paralelo (padrão: 1)")   # paralelismo entre sementes
parser.add_argument("--threads", type=int, default=None,
                    help="threads de álgebra linear por processo (padrão: núcleos / processos)")   # threads do BLAS
parser.add_argument("--situacao", action="store_true",
                    help="só mostra a situação das sementes e sai")   # consulta sem rodar nada
args = parser.parse_args()                            # lê os argumentos

# ---------------- Threads da álgebra linear (precisa vir antes de importar o NumPy) ----------------
n_nucleos = os.cpu_count() or 1                       # núcleos disponíveis na máquina
threads = args.threads or max(1, n_nucleos // max(1, args.processos))   # divide os núcleos entre os processos
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):   # bibliotecas usuais
    os.environ[var] = str(threads)                    # evita que processos paralelos disputem os mesmos núcleos

import numpy as np                                    # vetores (importado depois de fixar as threads)
from comum.dados import PASTA_RESULTADOS              # pasta de resultados do projeto

# ---------------- Configuração (Tabela de hiperparâmetros da metodologia) ----------------
TESTE_RAPIDO = args.teste_rapido                      # True: roda uma busca minúscula só para conferir o encadeamento
N_POP = 8 if TESTE_RAPIDO else 40                     # tamanho da população
N_GER = 2 if TESTE_RAPIDO else 50                     # número de gerações
SEMENTES = args.sementes if args.sementes is not None else ([0, 1] if TESTE_RAPIDO else list(range(10)))   # execuções independentes
PC, ETA_C = 0.9, 20.0                                 # cruzamento SBX
N_GENES = 8                                           # tamanho do genoma (5 genes + 3 constantes de tempo)
PM, ETA_M = 1.0 / N_GENES, 20.0                       # mutação polinomial (P_m = 1/8)
PASTA_BASE = args.pasta or str(PASTA_RESULTADOS / "fom")   # pasta base dos resultados da busca
PASTA_SAIDA = PASTA_BASE + ("_teste" if TESTE_RAPIDO else "")   # o teste rápido fica numa pasta separada
os.makedirs(PASTA_SAIDA, exist_ok=True)               # cria a pasta, se não existir

PENALIDADE_F1 = 100.0                                 # NMSE atribuído a ajustes que falham (metodologia: 100 dB)
PENALIDADE_F3 = 17.0                                  # log10 do número de condição para falhas (acima do teto numérico ~16)
CHAVES = ["X", "F", "X_frente", "F_frente", "historico_X", "historico_F", "duracao_s", "N", "G"]   # conteúdo obrigatório


def arquivo_resultado(s):                             # caminho do resultado principal de uma semente
    return os.path.join(PASTA_SAIDA, f"fom_semente{s}.npz")


def arquivo_registro(s):                              # caminho do registro detalhado de uma semente
    return os.path.join(PASTA_SAIDA, f"fom_semente{s}_registro.json")


def arquivo_log(s):                                   # caminho do log de uma semente rodada em paralelo
    return os.path.join(PASTA_SAIDA, f"fom_semente{s}.log")


# ---------------- Verificação das sementes já concluídas ----------------
def situacao_semente(s):                              # devolve ("concluída" | "pendente" | "inválida", motivo)
    arq = arquivo_resultado(s)                        # arquivo esperado
    if not os.path.exists(arq):                       # nunca terminou
        return "pendente", "sem arquivo"
    try:                                              # o arquivo existe: confere se está íntegro e completo
        with np.load(arq) as d:                       # abre o .npz
            faltando = [k for k in CHAVES if k not in d.files]   # chaves ausentes
            if faltando:
                return "inválida", f"faltam {faltando}"
            if int(d["N"]) != N_POP or int(d["G"]) != N_GER:   # rodada com outra configuração
                return "inválida", f"configuração N={int(d['N'])}, G={int(d['G'])} diferente da atual"
            n_aval = d["historico_F"].shape[0]        # número de avaliações registradas
            if n_aval != N_POP * (N_GER + 1):         # execução incompleta
                return "inválida", f"{n_aval} avaliações em vez de {N_POP * (N_GER + 1)}"
            if d["F_frente"].shape[0] == 0 or not np.all(np.isfinite(d["F"])):   # frente vazia ou valores inválidos
                return "inválida", "frente vazia ou objetivos não finitos"
    except Exception as erro:                         # arquivo corrompido (ex.: execução caiu durante a gravação)
        return "inválida", f"não foi possível ler ({erro})"
    if not os.path.exists(arquivo_registro(s)):       # o registro detalhado também é necessário para a análise
        return "inválida", "falta o registro .json"
    return "concluída", ""


def mostrar_situacao():                               # imprime a situação de todas as sementes
    situacao = {s: situacao_semente(s) for s in SEMENTES}   # verifica cada semente
    print(f"pasta: {PASTA_SAIDA} | N = {N_POP}, G = {N_GER}")
    for s, (estado, motivo) in situacao.items():      # uma linha por semente
        print(f"  semente {s}: {estado}" + (f" ({motivo})" if motivo else ""))
    return situacao


# ---------------- Gravação atômica ----------------
def salvar_npz(caminho, **dados):                     # grava num arquivo temporário e renomeia no fim
    tmp = caminho.replace(".npz", ".tmp.npz")         # nome temporário (np.savez exige a extensão .npz)
    np.savez(tmp, **dados)                            # escreve tudo no temporário
    os.replace(tmp, caminho)                          # troca de nome de uma vez: o arquivo final nunca fica pela metade


def salvar_json(caminho, dados):                      # mesma ideia para o registro em JSON
    tmp = caminho + ".tmp"                            # nome temporário
    with open(tmp, "w") as f:                         # escreve no temporário
        json.dump(dados, f)
    os.replace(tmp, caminho)                          # renomeia de uma vez


# ---------------- Verificação inicial ----------------
print("=== situação antes da execução ===")
situacao = mostrar_situacao()                         # quais sementes já estão prontas
pendentes = [s for s, (estado, _) in situacao.items() if estado != "concluída"]   # pendentes e inválidas
print(f"sementes a executar: {pendentes if pendentes else 'nenhuma'}")
if args.situacao or not pendentes:                    # só consulta, ou não há nada a fazer
    sys.exit(0)                                       # encerra sem carregar o modelo


# ---------------- Modo paralelo: um processo por semente ----------------
if args.processos > 1 and len(pendentes) > 1:         # vale a pena paralelizar
    from concurrent.futures import ThreadPoolExecutor  # dispara e acompanha os subprocessos
    from comum.fom import _recursao_gl, pesos_gl      # recursão do numba
    _recursao_gl(np.ones(10), pesos_gl(0.5, 3), 1.0, 0.0)   # compila uma vez e grava no cache antes de abrir os processos

    def rodar_semente(s):                             # executa este mesmo script para uma única semente
        cmd = [sys.executable, os.path.abspath(__file__), "--sementes", str(s),
               "--threads", str(threads), "--pasta", PASTA_BASE]   # mesma configuração, uma semente
        if TESTE_RAPIDO:                              # repassa o modo de teste
            cmd.append("--teste-rapido")
        print(f"semente {s}: iniciada (log em {arquivo_log(s)})", flush=True)
        t0 = time.time()                              # início da execução
        with open(arquivo_log(s), "w") as log:        # saída do processo vai para o log da semente
            ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode   # roda e espera terminar
        print(f"semente {s}: terminou em {(time.time() - t0) / 60:.1f} min (código {ret})", flush=True)
        return ret                                    # código de saída do processo

    print(f"\n{args.processos} processos em paralelo, {threads} thread(s) de álgebra linear cada")
    try:                                              # Ctrl+C também chega aos subprocessos
        with ThreadPoolExecutor(max_workers=args.processos) as pool:   # no máximo P sementes ao mesmo tempo
            list(pool.map(rodar_semente, pendentes))  # espera todas terminarem
    except KeyboardInterrupt:                         # interrupção manual
        print("\nexecução interrompida; as sementes em andamento continuam pendentes")
    print("\n=== situação após a execução ===")
    situacao = mostrar_situacao()                     # verifica de novo a partir dos arquivos
    restantes = [s for s, (estado, _) in situacao.items() if estado != "concluída"]   # o que ainda falta
    print("todas as sementes concluídas" if not restantes else
          f"ainda faltam as sementes {restantes}: rode o script de novo para executá-las")
    sys.exit(0)                                       # o processo coordenador termina aqui


# ---------------- Modo sequencial: carrega o modelo e o NSGA-II ----------------
from comum.fom import avaliar_fom, decodificar, LIM_INF, LIM_SUP   # modelo fracionário e genoma
from comum.nsga2 import nsga2                         # NSGA-II implementado do zero
assert len(LIM_INF) == N_GENES, "o tamanho do genoma mudou: ajuste N_GENES"   # garante P_m = 1/8 correto

registro = []                                         # guarda os detalhes de cada avaliação da execução atual


def objetivos_fom(g):                                 # recebe um genoma e devolve [f1, f2, f3]
    try:                                              # protege a busca contra falhas numéricas raras
        obj, info = avaliar_fom(g)                    # avaliação do modelo (objetivos e detalhes)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as erro:   # SVD que não converge, etc.
        th = decodificar(g)                           # estrutura do genoma, para calcular f2 mesmo assim
        n_par = 2 * th["K"] * (th["M"] + 1) * (th["J"] + 1) + th["J"] + 1   # número de parâmetros reais
        obj = np.array([PENALIDADE_F1, n_par, PENALIDADE_F3])   # objetivos de penalidade
        info = {**th, "n_coef": None, "posto": None, "tempo_s": None, "erro": str(erro)}   # registra a falha
    registro.append({"genoma": np.asarray(g, float).tolist(),   # genoma avaliado
                     "objetivos": np.asarray(obj, float).tolist(),   # [f1, f2, f3]
                     **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()}})   # detalhes
    return np.asarray(obj, float)                     # o nsga2 só precisa do vetor de objetivos


def resumo_frente(F_frente, X_frente):                # imprime a primeira frente ordenada pelo NMSE
    print(f"{'NMSE val':>9s} {'parâm.':>7s} {'log10 k':>8s} | " + " ".join(f"{n:>6s}" for n in ["J", "K", "M", "L", "alfa"]) + "  taus")
    for i in np.argsort(F_frente[:, 0]):              # da mais precisa para a menos precisa
        f1, f2, f3 = F_frente[i]                      # objetivos
        th = decodificar(X_frente[i])                 # estrutura decodificada
        print(f"{f1:9.2f} {int(f2):7d} {f3:8.1f} | {th['J']:>6d} {th['K']:>6d} {th['M']:>6d} "
              f"{th['L']:>6d} {th['alfa']:>6.3f}  {np.round(th['taus'], 1)}")


# ---------------- Aquecimento: compila a recursão antes de medir tempos ----------------
_ = objetivos_fom(np.array([1, 2, 0, 1.0, 0.5, 1.0, 0.0, 0.0]))   # genoma pequeno: força a compilação do numba
registro.clear()                                      # descarta a avaliação de aquecimento

# ---------------- Execuções ----------------
falhas = {}                                           # sementes que falharam nesta execução e o motivo
tempos = []                                           # tempos de avaliação de todas as sementes desta execução
for s in pendentes:                                   # só as sementes que faltam
    registro.clear()                                  # começa um registro novo para esta semente
    t0 = time.time()                                  # início da execução
    print(f"\n===== semente {s}: N = {N_POP}, G = {N_GER} =====", flush=True)
    try:                                              # uma falha nesta semente não interrompe as outras
        res = nsga2(objetivos_fom, LIM_INF, LIM_SUP, N=N_POP, G=N_GER, pc=PC, eta_c=ETA_C,
                    pm=PM, eta_m=ETA_M, semente=s, verbose=True)   # busca com o NSGA-II
        duracao = time.time() - t0                    # duração total da execução
        salvar_json(arquivo_registro(s), registro)    # primeiro o registro detalhado
        salvar_npz(arquivo_resultado(s), X=res["X"], F=res["F"], X_frente=res["X_frente"],
                   F_frente=res["F_frente"], historico_X=res["historico_X"],
                   historico_F=res["historico_F"], duracao_s=duracao, N=N_POP, G=N_GER)   # por último o .npz
        tempos += [r["tempo_s"] for r in registro if r.get("tempo_s")]   # acumula os tempos de avaliação
        print(f"semente {s} concluída em {duracao / 60:.1f} min; {len(registro)} avaliações; "
              f"primeira frente com {len(res['F_frente'])} soluções")
        resumo_frente(res["F_frente"], res["X_frente"])   # mostra a frente desta semente
    except KeyboardInterrupt:                         # interrupção manual: para tudo sem marcar a semente como falha
        print(f"\nexecução interrompida durante a semente {s}; ela continua pendente")
        break
    except Exception as erro:                         # qualquer outro problema: registra e segue para a próxima
        falhas[s] = repr(erro)                        # guarda o motivo
        print(f"semente {s} FALHOU após {(time.time() - t0) / 60:.1f} min: {erro!r}")
        traceback.print_exc()                         # mostra onde o erro aconteceu

# ---------------- Situação final ----------------
print("\n=== situação após a execução ===")
situacao = mostrar_situacao()                         # verifica de novo a partir dos arquivos
restantes = [s for s, (estado, _) in situacao.items() if estado != "concluída"]   # o que ainda falta
if falhas:                                            # falhas desta execução
    print("falhas nesta execução:", {s: m for s, m in falhas.items()})
print("todas as sementes concluídas" if not restantes else
      f"ainda faltam as sementes {restantes}: rode o script de novo para executá-las")
if tempos:                                            # estimativa para a execução completa
    print(f"tempo médio por avaliação: {np.mean(tempos):.2f} s (máx. {np.max(tempos):.1f} s); "
          f"estimativa para N = 40, G = 50: {np.mean(tempos) * 40 * 51 / 3600:.1f} h por semente")
