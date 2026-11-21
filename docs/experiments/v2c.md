# Experiência V2-C — Generalização para IPs de origem não vistos

[← Experiências](README.md) · [Metodologia](../methodology.md) · [Resultados](../results/README.md)

## Motivação

A validação V2-B permitia associação entre identidade de IP e classe. A V2-C introduz separação por `src_ip` para investigar o desempenho em origens de tráfego ausentes do treino.

## Preparação e conjuntos

| Subconjunto | IPs | Janelas | Normais | Ataques |
|---|---:|---:|---:|---:|
| Treino interno | 30 | 15 951 | 12 388 | 3 563 |
| Validação interna | 7 | 5 060 | 3 149 | 1 911 |
| Treino final | 37 | 21 011 | 15 537 | 5 474 |
| Avaliação reservada | 3 | 4 256 | 2 619 | 1 637 |

Os três IPs reservados são `192.168.1.30`, `192.168.1.193` e `192.168.1.195`. Não participam na validação interna nem no treino final. O `scaler` final é ajustado apenas sobre os dados disponíveis para treino.

## Seleção de hiperparâmetros e protocolo congelado

A validação interna (7 IPs separados dos 30 IPs de treino interno) é usada para determinar as durações dos treinos finais. O protocolo foi registado **antes** de consultar a avaliação reservada em [`results/v2c_final_protocol.json`](../../results/v2c_final_protocol.json).

| Arquitetura | Épocas centralizadas | Rondas FedAvg |
|---|---:|---:|
| GRU | 1 | 15 |
| LSTM | 1 | 4 |
| Transformer | 1 | 3 |

As três seeds são 42, 123 e 2026. O limiar de classificação permanece em 0,5. O treino usa Adam, learning rate 0,001, batch size 32 e dimensão oculta 32. No FedAvg participam cinco clientes lógicos, com uma época local por ronda e ponderação por número de exemplos. As partições finais não partilham IPs entre clientes e apresentam distribuições de classes heterogéneas.

## Avaliação final — desempenho global

| Arquitetura | Centralizado | Federado | Diferença (Fed − Central) |
|---|---:|---:|---:|
| GRU | 0,7750 | 0,9314 | 0,1565 |
| LSTM | 0,7758 | 0,7705 | -0,0053 |
| Transformer | 0,7998 | 0,8105 | 0,0107 |

As métricas resultam da média das três seeds. O melhor F1 global observado é o da **GRU federada**, mas a vantagem não se estende de modo uniforme às outras arquiteturas.

## Análise adicional dos dois IPs com tráfego misto

Os IPs `.193` e `.195` contêm em conjunto 2 728 janelas (2 619 normais e 109 de ataque). O IP `.30` contém apenas 1 528 janelas de ataque; a sua presença pode elevar o F1 global e não permite definir FPR nesse IP.

| Arquitetura | Centralizado | Federado | Diferença (Fed − Central) |
|---|---:|---:|---:|
| GRU | 0,1475 | 0,4212 | 0,2737 |
| LSTM | 0,1462 | 0,1471 | 0,0010 |
| Transformer | 0,1788 | 0,1887 | 0,0099 |

O F1 dos IPs mistos é calculado **agregando primeiro as matrizes de confusão de cada seed**, e não fazendo a média simples do F1 de cada IP.

No IP `.195`, que tem apenas 28 janelas de ataque, persistem falsos positivos e dificuldades de deteção de ataques raros mesmo para a GRU federada.

## Ficheiros e reprodutibilidade

- Treino centralizado: `src/training/train_v2c.py` (interno), `src/training/train_v2c_final.py` (final).
- Treino federado: `src/federated/run_flower_simulation_v2c.py` (interno), `src/federated/run_flower_simulation_v2c_final.py` (final).
- Avaliação final já concluída: `src/evaluation/evaluate_v2c_final.py`.
- Resultados congelados: [`results/v2c_final_results_consolidated.json`](../../results/v2c_final_results_consolidated.json), [`results/tables/`](../../results/tables/).

A avaliação reservada **não deve ser reexecutada para procurar melhores configurações**, nem utilizada para reajustar modelos ou limiares.

## Limitações

A seleção dos três IPs reservados foi deliberada após inspeção das distribuições; as janelas sobrepostas são dependentes; um IP não representa necessariamente um dispositivo; e a generalização observada é **por IP de origem**, não temporal. FedAvg não garante privacidade sem mecanismos adicionais.

[→ Comparação centralizada/federada](../results/centralized-vs-federated.md)
