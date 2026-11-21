# Documentação do projeto SRSI

[← Página principal](../README.md)

Este espaço documenta a implementação experimental de deteção binária de anomalias de tráfego IoT com **GRU, LSTM e Transformer**, em treino centralizado e federado com **Flower/FedAvg**.

## Guia de leitura

| Tema | Documento |
|---|---|
| Componentes de software, responsabilidades e fluxo de dados | [Arquitetura](architecture.md) |
| Dados ToN_IoT, características e construção de sequências | [Dataset](dataset.md) |
| Protocolo, validação, hiperparâmetros e métricas | [Metodologia](methodology.md) |
| Comparação dos cenários V2-B e V2-C | [Experiências](experiments/README.md) |
| Métricas finais e respetiva interpretação | [Resultados](results/README.md) |
| Pressupostos e ameaças à validade | [Limitações](results/limitations.md) |
| Preparação do ambiente e referência aos scripts | [Reprodutibilidade](reproducibility.md) |

## Resultados de referência

Os resultados finais V2-C (18 modelos, três IPs de origem reservados e 4 256 janelas) encontram-se em [`results/v2c_final_results_consolidated.json`](../results/v2c_final_results_consolidated.json). As versões originais das tabelas estão em [`results/tables/`](../results/tables/).

**A avaliação final já foi executada.** Não se devem escolher novos hiperparâmetros com base nos resultados reservados.
