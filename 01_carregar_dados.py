# ===== Etapa 01: download e conferência do APA_200MHz =====
# Uso: python 01_carregar_dados.py
# Na primeira execução baixa o dataset para a pasta APA_200MHz/; depois só confere os números.

from comum.dados import resumo                        # o import já baixa e carrega o dataset

if __name__ == "__main__":                            # só executa quando o arquivo é rodado diretamente
    resumo()                                          # imprime amostras, cortes, f_s, PAPR e ganho linear
