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

## Experiência complementar de segurança — Model poisoning

### Objetivo

Avaliar o impacto da adulteração de atualizações
enviadas por um cliente malicioso no treino federado
de uma GRU, utilizando Flower e FedAvg.

Esta experiência é independente da avaliação final
V2-C e utiliza exclusivamente dados de desenvolvimento.

### Configuração experimental

| Parâmetro | Valor |
|---|---|
| Arquitetura | GRU |
| Estratégia de agregação | FedAvg |
| Plataforma | Flower |
| Clientes lógicos | 5 |
| Cliente malicioso | Cliente 4 |
| Rondas federadas | 20 |
| Épocas locais | 1 por ronda |
| Batch | 32 |
| Taxa de aprendizagem | 0,001 |
| Semente | 42 |
| Janelas de treino | 15 951 |
| Janelas de validação | 5 060 |
| Fator de adulteração | -10 |
| Limiar de classificação | 0,5 |

Foram executados dois cenários com a mesma arquitetura,
partições e configuração de treino:

1. **Benigno:** os cinco clientes enviam atualizações
   legítimas dos parâmetros.
2. **Adversarial:** quatro clientes enviam atualizações
   legítimas e o cliente 4 adultera sistematicamente
   a sua atualização antes da agregação.

### Mecanismo de ataque

O cliente malicioso recebe os parâmetros globais,
executa uma época de treino local e calcula os
parâmetros que enviaria normalmente ao servidor.

Antes do envio, aplica a transformação:

θ_mal = θ_global + α × (θ_local - θ_global)

com α = -10.

Este fator inverte a direção da atualização local
e amplifica a sua magnitude dez vezes.

A adulteração afeta os parâmetros treináveis do modelo.
Não são modificados os dados nem os rótulos de treino,
pelo que o ataque é classificado como **model poisoning**,
e não como data poisoning.

O servidor utiliza FedAvg convencional, sem mecanismos
de filtragem ou rejeição de atualizações maliciosas.

### Avaliação e verificação

O desempenho foi acompanhado ao longo das 20 rondas
utilizando 5 060 janelas de validação interna.

Foram analisados o F1-score, a sensibilidade, a taxa
de falsos positivos e a perda de validação.

A consistência dos resultados adversariais foi
verificada através do carregamento independente
dos checkpoints GRU e do recálculo das previsões
com limiar de 0,5.

No cenário adversarial, foram verificados:

- O melhor checkpoint segundo a perda de validação,
  correspondente à ronda 1.
- O modelo global da ronda 20, preservado na
  execução de verificação.

A experiência não utilizou os três endereços IP
reservados para a avaliação final V2-C.

### Limitações metodológicas

A experiência considera uma única arquitetura,
uma semente, cinco clientes lógicos e um fator
de adulteração específico.

O ataque encontra-se ativo desde a primeira ronda.
Não foram avaliadas outras estratégias adversariais,
defesas robustas ou dispositivos IoT físicos.

Os resultados permitem analisar o impacto deste ataque
na configuração observada, mas não demonstram
vulnerabilidade universal dos sistemas federados
nem propriedades de confidencialidade.

Para consultar os comandos, artefactos e resultados,
ver [Demonstração federada e segurança](federated-security-demo.md).

---

Ver também [Limitações](results/limitations.md) e [Reprodutibilidade](reproducibility.md).
