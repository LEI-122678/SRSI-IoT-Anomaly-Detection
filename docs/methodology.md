<!-- Gerado por gerar_documentacao_srsi.py; pode ser atualizado pelo script. -->

# Metodologia experimental

[Índice](README.md) · [Arquitetura](architecture.md) · [Dataset](dataset.md) · [Metodologia](methodology.md) · [Reprodutibilidade](reproducibility.md)

## Objetivo e hipóteses

Comparar a deteção binária de anomalias no ToN_IoT utilizando **GRU**, **LSTM** e **Transformer**, em treino **centralizado** e **federado**. O treino federado foi simulado com **Flower/FedAvg** e cinco clientes lógicos que recebem IPs de origem sem sobreposição entre clientes.

## Fluxo experimental

1. Preparar registos de tráfego e agregados por minuto e IP de origem.
2. Construir sequências de cinco instantes e seis características.
3. Avaliar o cenário V2-B, baseado em divisão por data, reconhecendo a sobreposição de IPs entre treino e validação.
4. Criar a validação interna V2-C separando IPs de origem.
5. Escolher antecipadamente as durações de treino a partir da validação interna.
6. Congelar o protocolo em `results/v2c_final_protocol.json`.
7. Treinar os modelos finais nos 37 IPs disponíveis para treino.
8. Avaliar uma única vez nos três IPs reservados, sem reajuste posterior.

## Configuração V2-C final

| Parâmetro | Valor |
|---|---|
| Arquiteturas | GRU, LSTM, Transformer |
| Seeds | 42, 123, 2026 |
| Características | 6 por minuto |
| Comprimento das sequências | 5 minutos |
| Dimensão oculta | 32 |
| Otimizador | Adam |
| Taxa de aprendizagem | 0,001 |
| Batch | 32 |
| Função de perda | BCEWithLogitsLoss |
| Limiar de decisão | 0,5 |
| Clientes Flower | 5 |
| Épocas locais | 1 por ronda |
| Estratégia | FedAvg, ponderada por `num-examples` |

### Duração de treino fixada antes da avaliação

| Modelo | Épocas centralizadas | Rondas federadas |
|---|---:|---:|
| GRU | 1 | 15 |
| LSTM | 1 | 4 |
| Transformer | 1 | 3 |

Estas durações derivam da mediana dos melhores pontos observados durante a validação interna. No treino final **não houve validação, early stopping ou seleção de checkpoints pelo conjunto de avaliação**.

## Avaliação

Métricas: `accuracy`, `precision`, `recall`, `F1-score`, `FPR`, e contagens `TN`, `FP`, `FN`, `TP`.

- `F1 = 2·TP / (2·TP + FP + FN)`.
- `Recall = TP / (TP + FN)`.
- `FPR = FP / (FP + TN)`, apenas se existir pelo menos um exemplo normal.

As métricas globais são calculadas agregando todas as janelas. Nos IPs mistos, a matriz de confusão é **somada antes do cálculo do F1**; o resultado não é a média aritmética dos F1 individuais de cada IP.

As tabelas apresentam média e desvio-padrão de três seeds. Como há apenas três seeds e janelas sobrepostas, estas estatísticas **não devem ser interpretadas como intervalos de confiança para uma população de dispositivos**.

## Integridade experimental

Os três IPs reservados foram escolhidos após análise das distribuições, pelo que o protocolo não constitui uma seleção aleatória nem um teste completamente cego. Os resultados finais foram preservados após a primeira avaliação; não realizar afinação com os dados de avaliação.

Ver também [Limitações](results/limitations.md) e [Reprodutibilidade](reproducibility.md).
