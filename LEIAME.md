# Busca do modelo fracionário (FOM) com NSGA-II

**Importante: Para executar a validação do NSGA-II implementado nesse projeto contra o NSGA-II do Pymoo é necessário
instalar o Pymoo separadamente (versão 0.6.2) e utilizar o Python 3.14.** 

## Instalação
    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Etapas (rodar a partir desta pasta)
| Script | Bloco do notebook | Saída |
|---|---|---|
| `01_carregar_dados.py` | carregamento | baixa `APA_200MHz/` e confere os números |
| `02_gmp_benchmark.py` | 5 | `resultados/gmp_benchmark_apa.npz` |
| `03_gmp_causal.py` | 6 | `resultados/gmp_causal_apa.npz` |
| `04_fom_verificacao.py` | 7 (verificações) | só impressão; compila o numba |
| `05_validacao_zdt1.py` | 9 | `resultados/zdt1_frentes.pdf/.png` |
| `06_busca_fom.py` | 10 | `resultados/fom/fom_semente{s}.npz` e `_registro.json` |

As etapas são independentes: cada uma importa o que precisa de `comum/`
(`dados.py`, `gmp.py`, `fom.py` = bloco 7, `nsga2.py` = bloco 8).

## Busca (etapa 06)
    python 06_busca_fom.py --teste-rapido              # conferência (N = 8, G = 2)
    python 06_busca_fom.py --situacao                  # quais sementes já terminaram
    python 06_busca_fom.py --processos 4               # sementes pendentes, 4 ao mesmo tempo
    python 06_busca_fom.py --sementes 5 6 7            # só essas sementes

Pode interromper com Ctrl+C e rodar de novo: sementes concluídas são puladas.
Em modo paralelo, a saída de cada semente vai para `resultados/fom/fom_semente{s}.log`.
Cada processo usa cerca de 1 a 2 GB de RAM; escolha `--processos` pela memória e pelos núcleos.
