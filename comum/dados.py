# ===== Carregamento do APA_200MHz (OpenDPD) =====
# Módulo compartilhado: ao ser importado, baixa (só na primeira vez) os arquivos do dataset
# (Apache 2.0) e monta x, y, N, CORTES, fs e spec, usados por todas as etapas.

import json, urllib.request                           # leitura do spec.json e download dos arquivos
from pathlib import Path                              # caminhos independentes do sistema operacional
import numpy as np                                    # vetores complexos
import pandas as pd                                   # leitura dos CSV

RAIZ = Path(__file__).resolve().parent.parent         # pasta do projeto (a que contém os scripts)
BASE = "https://raw.githubusercontent.com/lab-emi/OpenDPD/main/datasets/APA_200MHz/"   # origem dos arquivos
PASTA = RAIZ / "APA_200MHz"                           # pasta local do dataset
PASTA_RESULTADOS = RAIZ / "resultados"                # pasta onde todas as etapas gravam suas saídas
PASTA.mkdir(exist_ok=True)                            # cria a pasta do dataset, se não existir
PASTA_RESULTADOS.mkdir(exist_ok=True)                 # cria a pasta de resultados, se não existir

arquivos = ["spec.json"] + [f"{b}_{s}.csv" for b in ("train", "val", "test") for s in ("input", "output")]   # arquivos necessários
for nome in arquivos:                                 # percorre os arquivos
    destino = PASTA / nome                            # caminho local
    if not destino.exists():                          # só baixa o que ainda não existe
        print(f"baixando {nome}...")                  # aviso de download
        urllib.request.urlretrieve(BASE + nome, destino)   # download do GitHub

with open(PASTA / "spec.json") as f:                  # especificação do dataset
    spec = json.load(f)
fs = spec["input_signal_fs"]                          # 983,04 MHz


def ler_iq(nome):                                     # lê um CSV com colunas I e Q e devolve o sinal complexo
    v = pd.read_csv(PASTA / nome).to_numpy(dtype=float)   # colunas I, Q
    return v[:, 0] + 1j * v[:, 1]                     # envoltória complexa


blocos = {"treino": "train", "val": "val", "teste": "test"}   # nome interno -> nome do arquivo
partes = {b: (ler_iq(f"{a}_input.csv"), ler_iq(f"{a}_output.csv")) for b, a in blocos.items()}   # entrada e saída de cada bloco

# Os três arquivos são trechos consecutivos da mesma captura; concatenados,
# o modo contínuo usa o fim de um bloco como passado do seguinte.
x = np.concatenate([partes[b][0] for b in blocos])    # entrada do PA no registro inteiro
y = np.concatenate([partes[b][1] for b in blocos])    # saída do PA no registro inteiro
N = len(x)                                            # número total de amostras

CORTES, ini = {}, 0                                   # limites (início, fim) de cada bloco no registro concatenado
for b in blocos:                                      # percorre treino, validação e teste
    fim = ini + len(partes[b][0])                     # fim do bloco atual
    CORTES[b] = (ini, fim)                            # guarda os limites
    ini = fim                                         # o próximo bloco começa onde este termina


def nmse_db(y, y_hat):                                # NMSE em dB entre a saída medida e a prevista
    return 10 * np.log10(np.sum(np.abs(y - y_hat) ** 2) / np.sum(np.abs(y) ** 2))


def resumo():                                         # conferência rápida (chamada pela etapa 01)
    p = np.abs(x) ** 2                                # potência instantânea da entrada
    print(spec["description"])                        # descrição do dataset
    print(f"amostras: {N}  |  cortes: {CORTES}")      # tamanho e divisão
    print(f"f_s = {fs / 1e6:.2f} MHz  ->  1 amostra = {1e9 / fs:.3f} ns, registro = {N / fs * 1e6:.1f} us")
    print(f"PAPR da entrada: {10 * np.log10(p.max() / p.mean()):.2f} dB  (spec: {spec['papr_db']} dB)")
    print(f"max|x| = {np.abs(x).max():.3f}, max|y| = {np.abs(y).max():.3f}  (normalizados pelo pico)")
    i0, i1 = CORTES["treino"]                         # limites do treino
    G = np.vdot(x[i0:i1], y[i0:i1]) / np.vdot(x[i0:i1], x[i0:i1])   # ganho linear por mínimos quadrados
    j0, j1 = CORTES["teste"]                          # limites do teste
    print(f"ganho linear: NMSE de teste = {nmse_db(y[j0:j1], G * x[j0:j1]):.2f} dB")
