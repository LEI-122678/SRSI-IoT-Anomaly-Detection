# Inventário técnico do código

[← Índice da documentação](README.md)

Este inventário identifica os scripts Python do projeto **SRSI — Deteção de Anomalias em Redes IoT**, agrupados pela estrutura real do repositório. As descrições sintetizam a finalidade de cada módulo; os nomes dos ficheiros, os imports e os caminhos de dados/resultados permanecem inalterados.

## Percurso experimental

| Fase | Módulos de referência |
|---|---|
| Exploração e preparação V2 | `src/preprocessing/create_v2_minute_features.py` |
| V2-B — treino e simulação | `src/training/train_v2.py`, `src/federated/run_flower_simulation.py` |
| V2-C — separação e validação interna | `src/preprocessing/prepare_v2_ip_generalization.py`, `src/preprocessing/prepare_v2c_internal_validation.py`, `src/training/train_v2c.py`, `src/federated/run_flower_simulation_v2c.py` |
| V2-C — treino final | `src/training/train_v2c_final.py`, `src/federated/run_flower_simulation_v2c_final.py` |
| V2-C — avaliação final (já realizada) | `src/evaluation/evaluate_v2c_final.py` |

Os scripts exploratórios e as experiências anteriores continuam arquivados por razões de rastreabilidade. **Não voltar a executar o teste final V2-C** nem utilizar os seus resultados para selecionar modelos.

## `src/models/`

| Script | Responsabilidade |
|---|---|
| [`gru.py`](../src/models/gru.py) | Definição do modelo GRU. |
| [`lstm.py`](../src/models/lstm.py) | Definição do modelo LSTM. |
| [`transformer.py`](../src/models/transformer.py) | Definição do modelo Transformer. |

## `src/preprocessing/`

| Script | Responsabilidade |
|---|---|
| [`analyze_all_datasets.py`](../src/preprocessing/analyze_all_datasets.py) | Análise exploratória dos conjuntos de dados. |
| [`analyze_attack_timeline.py`](../src/preprocessing/analyze_attack_timeline.py) | Análise temporal das ocorrências de ataque. |
| [`analyze_distribution_shift.py`](../src/preprocessing/analyze_distribution_shift.py) | Análise de diferenças de distribuição entre conjuntos. |
| [`analyze_hourly.py`](../src/preprocessing/analyze_hourly.py) | Análise exploratória por hora. |
| [`analyze_ip_shift.py`](../src/preprocessing/analyze_ip_shift.py) | Análise de alterações de distribuição por IP. |
| [`analyze_temporal.py`](../src/preprocessing/analyze_temporal.py) | Análise exploratória temporal. |
| [`analyze_v2_attack_hours.py`](../src/preprocessing/analyze_v2_attack_hours.py) | Análise das horas associadas a ataques V2. |
| [`analyze_v2_attack_runs.py`](../src/preprocessing/analyze_v2_attack_runs.py) | Análise de sequências consecutivas de ataques V2. |
| [`analyze_v2b_distribution_shift.py`](../src/preprocessing/analyze_v2b_distribution_shift.py) | Análise da mudança de distribuição na V2-B. |
| [`audit_raw_features.py`](../src/preprocessing/audit_raw_features.py) | Auditoria das características dos dados brutos. |
| [`create_pilot_dataset.py`](../src/preprocessing/create_pilot_dataset.py) | Construção do conjunto piloto inicial. |
| [`create_temporal_features.py`](../src/preprocessing/create_temporal_features.py) | Construção de características temporais exploratórias. |
| [`create_v2_minute_features.py`](../src/preprocessing/create_v2_minute_features.py) | Agregação das características V2 por minuto e IP. |
| [`evaluate_temporal_splits.py`](../src/preprocessing/evaluate_temporal_splits.py) | Análise de alternativas de divisão temporal. |
| [`evaluate_v2_windows.py`](../src/preprocessing/evaluate_v2_windows.py) | Avaliação exploratória das janelas da V2. |
| [`evaluate_window_sizes.py`](../src/preprocessing/evaluate_window_sizes.py) | Análise de diferentes tamanhos de janela. |
| [`explore_dataset.py`](../src/preprocessing/explore_dataset.py) | Exploração inicial do dataset. |
| [`fix_pilot_dataset.py`](../src/preprocessing/fix_pilot_dataset.py) | Correções na preparação do dataset piloto. |
| [`prepare_model_data.py`](../src/preprocessing/prepare_model_data.py) | Preparação de dados para modelos da fase inicial. |
| [`prepare_v2_ip_generalization.py`](../src/preprocessing/prepare_v2_ip_generalization.py) | Preparação do protocolo V2-C com separação por IP. |
| [`prepare_v2_temporal_data.py`](../src/preprocessing/prepare_v2_temporal_data.py) | Preparação da divisão temporal da V2-B. |
| [`prepare_v2c_internal_validation.py`](../src/preprocessing/prepare_v2c_internal_validation.py) | Preparação do treino/validação internos com IPs disjuntos. |
| [`validate_pilot_dataset.py`](../src/preprocessing/validate_pilot_dataset.py) | Verificações sobre o dataset piloto. |

## `src/training/`

| Script | Responsabilidade |
|---|---|
| [`train_gru.py`](../src/training/train_gru.py) | Treino da GRU na fase inicial. |
| [`train_lstm.py`](../src/training/train_lstm.py) | Treino da LSTM na fase inicial. |
| [`train_transformer.py`](../src/training/train_transformer.py) | Treino do Transformer na fase inicial. |
| [`train_v2.py`](../src/training/train_v2.py) | Treino centralizado V2-B. |
| [`train_v2c.py`](../src/training/train_v2c.py) | Treino centralizado da validação interna V2-C. |
| [`train_v2c_final.py`](../src/training/train_v2c_final.py) | Treino centralizado final V2-C, segundo o protocolo congelado. |

## `src/federated/`

| Script | Responsabilidade |
|---|---|
| [`check_aggregation_order.py`](../src/federated/check_aggregation_order.py) | Verificação da ordem de agregação. |
| [`check_fedavg_aggregation.py`](../src/federated/check_fedavg_aggregation.py) | Verificação da agregação FedAvg. |
| [`check_fedavg_equivalence.py`](../src/federated/check_fedavg_equivalence.py) | Verificação de equivalência de implementações FedAvg. |
| [`check_flower_serialization.py`](../src/federated/check_flower_serialization.py) | Verificação da serialização usada pelo Flower. |
| [`check_reproducibility.py`](../src/federated/check_reproducibility.py) | Verificações de reprodutibilidade. |
| [`compare_local_updates.py`](../src/federated/compare_local_updates.py) | Comparação de atualizações locais. |
| [`create_client_partitions.py`](../src/federated/create_client_partitions.py) | Preparação das partições federadas V2-B. |
| [`create_v2c_client_partitions.py`](../src/federated/create_v2c_client_partitions.py) | Preparação das partições da validação interna V2-C. |
| [`create_v2c_final_client_partitions.py`](../src/federated/create_v2c_final_client_partitions.py) | Preparação das partições federadas finais V2-C. |
| [`flower_client.py`](../src/federated/flower_client.py) | Cliente Flower da experiência V2-B. |
| [`flower_client_v2c.py`](../src/federated/flower_client_v2c.py) | Cliente Flower da validação interna V2-C. |
| [`flower_client_v2c_final.py`](../src/federated/flower_client_v2c_final.py) | Cliente Flower do treino final V2-C. |
| [`flower_server.py`](../src/federated/flower_server.py) | Servidor Flower da experiência V2-B. |
| [`flower_server_v2c.py`](../src/federated/flower_server_v2c.py) | Servidor Flower da validação interna V2-C. |
| [`flower_server_v2c_final.py`](../src/federated/flower_server_v2c_final.py) | Servidor Flower do treino final V2-C. |
| [`run_flower_simulation.py`](../src/federated/run_flower_simulation.py) | Execução da simulação federada V2-B. |
| [`run_flower_simulation_v2c.py`](../src/federated/run_flower_simulation_v2c.py) | Execução da simulação federada interna V2-C. |
| [`run_flower_simulation_v2c_final.py`](../src/federated/run_flower_simulation_v2c_final.py) | Execução da simulação federada final V2-C. |
| [`train_fedavg_gru.py`](../src/federated/train_fedavg_gru.py) | Implementação auxiliar de treino/agregação FedAvg para GRU. |

## `src/evaluation/`

| Script | Responsabilidade |
|---|---|
| [`benchmark_v2_models.py`](../src/evaluation/benchmark_v2_models.py) | Benchmark dos modelos da experiência V2-B. |
| [`compare_centralized_federated_gru.py`](../src/evaluation/compare_centralized_federated_gru.py) | Comparação centralizado/federado para GRU. |
| [`compare_centralized_federated_models.py`](../src/evaluation/compare_centralized_federated_models.py) | Comparação centralizado/federado para as três arquiteturas. |
| [`compare_models.py`](../src/evaluation/compare_models.py) | Comparação exploratória entre modelos. |
| [`evaluate_v2_baselines.py`](../src/evaluation/evaluate_v2_baselines.py) | Avaliação de baselines da V2. |
| [`evaluate_v2_test.py`](../src/evaluation/evaluate_v2_test.py) | Avaliação do conjunto de teste da V2-B. |
| [`evaluate_v2c_final.py`](../src/evaluation/evaluate_v2c_final.py) | Avaliação final reservada da V2-C; **não repetir nem utilizar para afinação**. |
| [`evaluate_v2c_mixed_ips.py`](../src/evaluation/evaluate_v2c_mixed_ips.py) | Análise interna V2-C dos IPs com classes misturadas. |

## Outros módulos e convenções

- Os ficheiros `__init__.py` não são enumerados nas tabelas por serem inicializadores de pacotes.
- Os scripts dependem de imports como `src.models.*` e de caminhos localizados em `data/` e `results/`. Qualquer alteração de nomes ou diretórios requer nova auditoria.
- Para a metodologia, consultar [metodologia](methodology.md), [experiências](experiments/README.md) e [reprodutibilidade](reproducibility.md).
- O protocolo e os resultados finais estão em `results/v2c_final_protocol.json` e `results/v2c_final_results_consolidated.json`.

**Total:** 59 scripts funcionais/de análise listados, excluindo `__init__.py`.

## Testes automatizados

- [`test_models.py`](../tests/test_models.py): testes de
  forward e gradientes de GRU, LSTM e Transformer.
- [`test_fedavg.py`](../tests/test_fedavg.py): teste
  de agregação ponderada com parâmetros sintéticos.
- `tests/test_environment.py`: verificação manual do ambiente,
  não recolhida automaticamente pelo pytest.

Executar: `python -m pytest tests/ -v`.
