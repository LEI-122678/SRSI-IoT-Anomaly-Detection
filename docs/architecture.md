# Arquitetura do Sistema

[← Voltar à documentação](README.md)

## 1. Visão geral

O projeto **SRSI — IoT Anomaly Detection** tem como objetivo estudar a deteção de anomalias em tráfego de redes IoT, comparando modelos de aprendizagem profunda em dois cenários:

- **Aprendizagem centralizada:** os modelos são treinados num único processo, com acesso ao conjunto de treino completo.
- **Aprendizagem federada:** os dados são distribuídos por cinco clientes lógicos, que treinam localmente os modelos. Um servidor agrega as atualizações através do algoritmo **FedAvg**, utilizando a framework **Flower**.

As arquiteturas analisadas são **GRU, LSTM e Transformer**.

## 2. Fluxo de funcionamento

```mermaid
flowchart TD
    A[Dataset ToN_IoT] --> B[Pré-processamento]
    B --> C[Agregação temporal por IP]
    C --> D[Criação de sequências]
    D --> E[Divisão dos dados]
    E --> F[Treino centralizado]
    E --> G[Treino federado com Flower]
    F --> H[Modelos treinados]
    G --> H
    H --> I[Avaliação]
    I --> J[Resultados e métricas]
```

O sistema transforma registos de tráfego em sequências temporais e utiliza modelos neuronais para classificar cada sequência como normal ou anómala.

## 3. Organização do código

| Diretório | Responsabilidade |
|---|---|
| `src/preprocessing/` | Exploração, análise e preparação dos dados |
| `src/models/` | Implementação das arquiteturas GRU, LSTM e Transformer |
| `src/training/` | Treino centralizado |
| `src/federated/` | Treino federado, clientes, servidores e FedAvg |
| `src/evaluation/` | Avaliação e comparação de modelos |
| `configs/` | Configurações experimentais |
| `tests/` | Testes e verificações |
| `data/` | Dados originais e processados |
| `results/` | Checkpoints, métricas, tabelas e figuras |

Os módulos Python são executados a partir da raiz do projeto, geralmente através de `python -m`.

## 4. Representação dos dados

As experiências V2-B e V2-C utilizam sequências de **cinco janelas temporais de um minuto**, associadas ao mesmo IP de origem.

Cada minuto é representado por seis características:

1. Número de ligações.
2. Soma dos bytes enviados pela origem.
3. Soma dos bytes enviados pelo destino.
4. Soma dos pacotes enviados pela origem.
5. Soma dos pacotes enviados pelo destino.
6. Duração média das ligações.

Os modelos recebem tensores com dimensões `(N, 5, 6)`.

A classificação binária indica a existência de ligações maliciosas no último minuto da sequência.

## 5. Modelos de aprendizagem profunda

As arquiteturas encontram-se nos seguintes ficheiros:

- `src/models/gru.py`
- `src/models/lstm.py`
- `src/models/transformer.py`

Nas experiências V2-C, os modelos utilizam uma dimensão oculta de 32. O Transformer utiliza quatro cabeças de atenção.

Estes módulos são partilhados pelo treino centralizado, pelo treino federado e pela avaliação.

## 6. Aprendizagem centralizada

O treino centralizado utiliza PyTorch e o otimizador Adam.

Os principais scripts são:

| Script | Finalidade |
|---|---|
| `src/training/train_v2.py` | Treino V2-B |
| `src/training/train_v2c.py` | Treino com validação interna V2-C |
| `src/training/train_v2c_final.py` | Treino definitivo V2-C |

Na fase final V2-C, o número de épocas foi definido previamente, sem utilizar os dados reservados de avaliação para selecionar hiperparâmetros.

## 7. Aprendizagem federada

A implementação utiliza **Flower**, com agregação **FedAvg**.

Cada cliente:

1. Recebe os parâmetros globais.
2. Treina localmente durante uma época.
3. Envia os parâmetros atualizados ao servidor.

O servidor agrega as atualizações, ponderando-as pelo número de exemplos de cada cliente.

Os principais ficheiros da experiência V2-C final são:

| Componente | Ficheiro |
|---|---|
| Particionamento | `src/federated/create_v2c_final_client_partitions.py` |
| Cliente | `src/federated/flower_client_v2c_final.py` |
| Servidor | `src/federated/flower_server_v2c_final.py` |
| Simulação | `src/federated/run_flower_simulation_v2c_final.py` |

São utilizados cinco clientes lógicos com diferentes proporções de tráfego normal e malicioso.

**Importante:** um cliente lógico não corresponde necessariamente a um dispositivo físico. Além disso, FedAvg, por si só, não garante privacidade nem segurança.

## 8. Experiências realizadas

### V2-B — Divisão temporal

Os dados foram divididos em treino, validação e teste com base nas datas.

Esta experiência permitiu comparar modelos centralizados e federados, mas revelou limitações relacionadas com a distribuição dos IPs, particularmente a sobreposição entre os IPs de treino e validação.

### V2-C — Generalização para IPs não vistos

Esta experiência procurou avaliar o desempenho em IPs de origem ausentes do treino.

Foram utilizados:

- 30 IPs para treino interno.
- 7 IPs para validação interna.
- 37 IPs para treino final.
- 3 IPs reservados para avaliação final.

O protocolo final foi registado em `results/v2c_final_protocol.json` antes da avaliação.

| Modelo | Épocas centralizadas | Rondas federadas |
|---|---:|---:|
| GRU | 1 | 15 |
| LSTM | 1 | 4 |
| Transformer | 1 | 3 |

Foram utilizadas três seeds: **42, 123 e 2026**, e um limiar de classificação de **0,5**.

## 9. Avaliação e armazenamento dos resultados

A avaliação final encontra-se implementada em:

`src/evaluation/evaluate_v2c_final.py`

Foram avaliadas **4 256 janelas provenientes de três IPs reservados**, utilizando os 18 checkpoints definitivos.

As principais métricas são accuracy, precision, recall, F1-score e taxa de falsos positivos (FPR).

Os resultados encontram-se em:

- `results/models/` — modelos centralizados.
- `results/federated/` — experiências federadas.
- `results/tables/` — métricas e comparações.
- `results/figures/` — figuras.
- `results/v2c_final_results_consolidated.json` — resultados finais consolidados.

## 10. Limitações e manutenção

A experiência V2-C avalia a generalização para três IPs de origem específicos e não demonstra generalização temporal ou para qualquer dispositivo IoT.

Os IPs reservados foram escolhidos após análise das suas distribuições. Além disso, as janelas temporais sobrepostas não são estatisticamente independentes.

Os scripts utilizam imports e caminhos específicos. Por esse motivo, qualquer alteração à estrutura de `src/`, `data/` ou `results/` deve ser acompanhada pela atualização das dependências e por testes de integridade.

Os resultados finais estão congelados e não devem ser utilizados para reajustar os modelos.

---

[← Voltar à documentação](README.md)