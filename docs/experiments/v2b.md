# Experiência V2-B — Divisão temporal

[← Experiências](README.md) · [V2-C](v2c.md) · [Limitações](../results/limitations.md)

## Objetivo

Comparar GRU, LSTM e Transformer nos regimes centralizado e federado e estudar o comportamento de sequências de tráfego IoT em dias posteriores aos utilizados no treino.

## Dados e divisão

| Subconjunto | Datas | Janelas | IPs distintos | Normais | Ataques |
|---|---|---:|---:|---:|---:|
| Treino | Até 26/04/2019 | 25 267 | 40 | 18 156 | 7 111 |
| Validação | 27/04/2019 | 4 031 | 15 | 2 774 | 1 257 |
| Teste | 28–29/04/2019 | 9 759 | 35 | 7 984 | 1 775 |

Cada sequência contém cinco minutos consecutivos de um `src_ip`, com seis características por minuto. A etiqueta binária corresponde à presença de ataques no último minuto.

## Treino

- Treino centralizado: `src/training/train_v2.py`.
- Treino federado: `src/federated/run_flower_simulation.py` (cliente/servidor respetivos).
- Três seeds: 42, 123 e 2026; arquiteturas GRU, LSTM e Transformer.
- Federado: cinco clientes lógicos com FedAvg e até 20 rondas, com seleção por validação.

## Resultados da validação

| Arquitetura | F1 médio centralizado | F1 médio federado |
|---|---:|---:|
| GRU | 0,9988 | 0,9955 |
| LSTM | 0,9995 | 0,9960 |
| Transformer | 1,0000 | 0,9963 |

Os resultados acima descrevem **a validação**, não o desempenho em IPs de origem novos. As tabelas disponíveis em `results/tables/v2b_*.csv` documentam outras métricas e os resultados de teste já realizados.

## Ameaças à validade

Todos os 15 IPs de validação também apareciam no treino e, nessa validação, os IPs com tráfego normal e os IPs com tráfego malicioso estavam segregados. Um baseline de atribuição da classe maioritária de treino por IP alcançava F1 perfeito nessa validação. Embora os modelos não recebam explicitamente o IP como característica, podem aprender padrões de tráfego correlacionados com a origem.

Assim, o F1 quase perfeito da V2-B **não deve ser interpretado como prova de generalização para IPs não vistos**. O protocolo V2-C foi concebido para investigar essa limitação de forma mais direta.

[→ Experiência V2-C](v2c.md)
