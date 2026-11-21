# SRSI — Deteção de Anomalias em Redes IoT

**Comparação de aprendizagem centralizada e federada com GRU, LSTM e Transformer, utilizando dados de tráfego de rede do ToN_IoT.**

Este repositório reúne o código, os protocolos e os resultados de uma investigação experimental sobre **deteção binária de tráfego anómalo em redes IoT**. Comparamos treino centralizado em PyTorch com treino federado em **Flower/FedAvg**, dando especial atenção à generalização para **IPs de origem não utilizados no treino**.

> **Estado do projeto:** as experiências V2-B e V2-C estão concluídas. Na V2-C, os **18 modelos finais** (3 arquiteturas × 2 abordagens × 3 sementes) foram avaliados uma vez num conjunto reservado de 3 IPs. O protocolo foi fixado antes dessa avaliação; os seus resultados não são utilizados para reajustar modelos.

## Navegação

| Documentação | Conteúdo |
|---|---|
| [Documentação completa](docs/README.md) | Índice de todas as páginas técnicas |
| [Arquitetura](docs/architecture.md) | Componentes, modelos e interdependências |
| [Inventário do código](docs/code-inventory.md) | Lista e responsabilidades dos 59 scripts Python |
| [Dataset](docs/dataset.md) | ToN_IoT, características e preparação |
| [Metodologia](docs/methodology.md) | Desenho experimental e métricas |
| [Experiências](docs/experiments/README.md) | Percurso experimental e versões |
| [Experiência V2-B](docs/experiments/v2b.md) | Divisão temporal e fatores de confusão |
| [Experiência V2-C](docs/experiments/v2c.md) | Validação por IP e avaliação reservada |
| [Resultados](docs/results/README.md) | Índice das análises |
| [Centralizado vs. federado](docs/results/centralized-vs-federated.md) | Comparação quantitativa |
| [Limitações](docs/results/limitations.md) | Validade e alcance das conclusões |
| [Reprodutibilidade](docs/reproducibility.md) | Ambiente, scripts e precauções |

## Objetivos

1. Explorar o tráfego normal e malicioso no ToN_IoT.
2. Criar sequências de características agregadas de rede.
3. Comparar **GRU, LSTM e Transformer** sob treino centralizado e federado.
4. Estudar a heterogeneidade das distribuições dos clientes federados (*non-IID*).
5. Avaliar os modelos em IPs de origem ausentes do treino, analisando tanto as métricas globais como os resultados por IP.

## Dataset e representação dos dados

A fonte utilizada é a componente de **tráfego de rede do ToN_IoT**. Os ficheiros de origem são armazenados localmente em `data/raw/ton_iot/` (por exemplo, `Network_dataset_1.csv`) e **não devem ser considerados incluídos num clone do GitHub**. O acesso aos dados de origem depende da respetiva disponibilização e condições de utilização.

As experiências V2-B e V2-C utilizam agregações por minuto e por `src_ip`, com seis características:

| Característica | Significado |
|---|---|
| `connections` | Número de ligações |
| `src_bytes_sum` | Soma dos bytes da origem |
| `dst_bytes_sum` | Soma dos bytes do destino |
| `src_pkts_sum` | Soma dos pacotes da origem |
| `dst_pkts_sum` | Soma dos pacotes do destino |
| `duration_mean` | Duração média das ligações |

Cada amostra contém **5 minutos consecutivos**, com dimensão `(5, 6)`. O rótulo binário corresponde à presença de ligações classificadas como ataque no **último minuto**. As características são transformadas com `log1p` e normalizadas com parâmetros ajustados apenas sobre os dados de treino de cada experiência.

As sequências deslizantes podem sobrepor-se e, por isso, **não representam observações estatisticamente independentes**.

## Desenho experimental

### V2-B — Divisão temporal

| Conjunto | Período | Janelas | IPs distintos |
|---|---|---:|---:|
| Treino | Até 26/04/2019 | 25 267 | 40 |
| Validação | 27/04/2019 | 4 031 | 15 |
| Teste | 28–29/04/2019 | 9 759 | 35 |

Foram comparadas as três arquiteturas com as sementes **42, 123 e 2026**, nas abordagens centralizada e federada. O treino federado utilizou **cinco clientes lógicos** e FedAvg.

**Limitação relevante:** os 15 IPs da validação também ocorrem no treino. Além disso, a distribuição de classes por IP permite que uma regra trivial baseada na classe maioritária por IP obtenha F1 de 1 nessa validação. Assim, os valores elevados da V2-B **não constituem prova de generalização para IPs desconhecidos**. A divisão por datas estuda um cenário temporal específico, mas não elimina fatores de confusão associados aos IPs.

[Detalhes da V2-B](docs/experiments/v2b.md)

### V2-C — IPs de origem não vistos no treino

A V2-C isola **3 IPs de origem** para a avaliação final, separando-os dos **37 IPs utilizados no treino final**.

| Conjunto | Janelas | IPs | Normal | Ataque |
|---|---:|---:|---:|---:|
| Treino final | 21 011 | 37 | 15 537 | 5 474 |
| Avaliação reservada | 4 256 | 3 | 2 619 | 1 637 |

Antes do treino final, os 37 IPs disponíveis foram divididos numa **validação interna sem sobreposição de IPs**: 30 IPs para treino interno (15 951 janelas) e 7 para validação interna (5 060 janelas). As durações do treino definitivo foram fixadas com base na mediana das melhores épocas/rondas da validação interna.

| Configuração final | Centralizado | Federado |
|---|---|---|
| Modelos | GRU, LSTM, Transformer | GRU, LSTM, Transformer |
| Sementes | 42, 123, 2026 | 42, 123, 2026 |
| GRU | 1 época | 15 rondas |
| LSTM | 1 época | 4 rondas |
| Transformer | 1 época | 3 rondas |
| Clientes | — | 5 clientes lógicos |
| Épocas locais | — | 1 por ronda |
| Batch / learning rate | 32 / 0,001 | 32 / 0,001 |
| Dimensão oculta / threshold | 32 / 0,5 | 32 / 0,5 |

Os clientes federados têm números de janelas semelhantes, mas percentagens de ataque entre **2,90% e 53,89%**, caracterizando heterogeneidade entre clientes. Cada IP de treino foi atribuído a apenas um cliente.

O protocolo final está preservado em [`results/v2c_final_protocol.json`](results/v2c_final_protocol.json). Os IPs reservados foram `192.168.1.30`, `192.168.1.193` e `192.168.1.195`. **Foram escolhidos deliberadamente após inspeção das distribuições**, pelo que não representam uma amostra aleatória nem um teste inteiramente cego.

[Detalhes da V2-C](docs/experiments/v2c.md)

## Resultados finais V2-C

Médias e desvios-padrão entre as três sementes, nos **4 256 exemplos reservados**:

| Arquitetura | Abordagem | F1 médio ± desvio-padrão | Recall médio | FPR médio |
|---|---|---:|---:|---:|
| GRU | Centralizada | 0,7750 ± 0,0010 | 0,9821 | 0,3453 |
| **GRU** | **Federada** | **0,9314 ± 0,0183** | **0,9839** | **0,0808** |
| LSTM | Centralizada | 0,7758 ± 0,0064 | 0,9813 | 0,3429 |
| LSTM | Federada | 0,7705 ± 0,0359 | 0,9813 | 0,3561 |
| Transformer | Centralizada | 0,7998 ± 0,0245 | 0,9859 | 0,3006 |
| Transformer | Federada | 0,8105 ± 0,0233 | 0,9851 | 0,2795 |

A **GRU federada** obteve o melhor F1 médio nesta avaliação e reduziu substancialmente os falsos positivos face à GRU centralizada. A vantagem da federação **não é universal**: a LSTM federada apresentou F1 global ligeiramente inferior ao da centralizada, e a diferença do Transformer foi pequena.

### Porque é importante analisar os IPs separadamente?

O IP `192.168.1.30` contém **1 528 janelas de ataque e nenhuma janela normal**. Consequentemente, a FPR nesse IP é indefinida e o seu desempenho quase perfeito influencia o F1 global.

Nos outros dois IPs, que contêm **2 728 janelas (109 ataques e 2 619 normais)**, o F1 foi calculado agregando primeiro as matrizes de confusão por modelo e semente:

| Arquitetura | F1 centralizado | F1 federado |
|---|---:|---:|
| GRU | 0,1475 | **0,4212** |
| LSTM | 0,1462 | 0,1471 |
| Transformer | 0,1788 | 0,1887 |

A GRU federada mantém a vantagem, mas o seu F1 desce de **0,9314 globalmente para 0,4212 nos IPs mistos**. O IP `.195` é particularmente difícil: contém apenas 28 ataques entre 1 581 janelas. Por isso, **o F1 global não deve ser apresentado isoladamente**.

[Comparação detalhada](docs/results/centralized-vs-federated.md) · [Limitações](docs/results/limitations.md)

## Organização do repositório

```text
SRSI-IoT-Anomaly-Detection/
├── configs/               # Configuração experimental
├── data/                  # Dados brutos e preparados (locais)
├── docs/                  # Documentação técnica e científica
│   ├── experiments/       # V2-B e V2-C
│   └── results/           # Análise dos resultados e limitações
├── results/
│   ├── federated/         # Relatórios e checkpoints federados
│   ├── figures/           # Figuras
│   ├── models/            # Relatórios e checkpoints centralizados
│   ├── tables/            # Métricas e comparações em CSV
│   ├── v2c_final_protocol.json
│   └── v2c_final_results_consolidated.json
├── src/
│   ├── evaluation/        # Avaliação
│   ├── federated/         # Flower e FedAvg
│   ├── models/            # GRU, LSTM e Transformer
│   ├── preprocessing/     # Preparação dos dados
│   └── training/          # Treino centralizado
├── tests/
├── requirements.txt
└── README.md
```


## Instalação

São necessários Python compatível com [`requirements.txt`](requirements.txt), PyTorch, Flower e acesso aos dados ToN_IoT. Exemplo em **PowerShell**:

```powershell
git clone https://github.com/LEI-122678/SRSI-IoT-Anomaly-Detection.git
cd SRSI-IoT-Anomaly-Detection
git switch goncalo

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Os dados originais devem ser disponibilizados em `data/raw/ton_iot/`. Os ficheiros de dados preparados (`.npz`, `.npy`) e os checkpoints (`.pt`) podem não estar incluídos no GitHub.

## Execução e reprodutibilidade

> **Não repetir a avaliação final V2-C como rotina de instalação.** Os resultados reservados já foram consultados uma vez e não devem ser utilizados para afinar hiperparâmetros. Um novo estudo independente exigiria uma nova divisão e um protocolo previamente definido.

Principais módulos de pré-processamento, executados a partir da raiz do projeto:

```powershell
python -m src.preprocessing.create_v2_minute_features
python -m src.preprocessing.prepare_v2_temporal_data
python -m src.preprocessing.prepare_v2_ip_generalization
python -m src.preprocessing.prepare_v2c_internal_validation
```

**Atenção:** estes comandos produzem ou recriam artefactos. A ordem de execução e a existência de saídas devem ser verificadas antes de os utilizar sobre dados já preparados.

Os principais módulos do treino definitivo V2-C são `src/training/train_v2c_final.py`, `src/federated/create_v2c_final_client_partitions.py` e `src/federated/run_flower_simulation_v2c_final.py`. Não devem ser executados indiscriminadamente sobre experiências concluídas.

Para consultar os resultados **sem treinar ou avaliar novamente**:

- [Protocolo V2-C](results/v2c_final_protocol.json)
- [Resultados consolidados](results/v2c_final_results_consolidated.json)
- [Métricas por modelo e semente](results/tables/v2c_final_evaluation_models.csv)
- [Resumo global](results/tables/v2c_final_evaluation_summary.csv)
- [Resultados por IP](results/tables/v2c_final_evaluation_per_ip.csv)
- [Resumo dos IPs mistos](results/tables/v2c_final_mixed_ips_summary.csv)

Para procedimentos adicionais, consultar a página de [reprodutibilidade](docs/reproducibility.md).

## Limitações científicas

- **IPs de origem não são necessariamente dispositivos físicos.**
- Os três IPs finais foram escolhidos após análise exploratória; não são uma amostra aleatória representativa.
- Janelas deslizantes sobrepostas criam dependência entre observações.
- A experiência V2-C avalia IPs desconhecidos no treino, **não** generalização temporal, entre instalações ou para novos datasets.
- **FedAvg, por si só, não assegura privacidade ou segurança**: atualizações de modelos podem revelar informação, mesmo quando os dados brutos permanecem nos clientes.
- As durações centralizadas e federadas foram fixadas segundo critérios internos distintos, sem pressupor custos computacionais ou de comunicação equivalentes.

O desempenho observado é específico dos IPs e do protocolo utilizados; não demonstra uma superioridade estatística universal da aprendizagem federada.

## Artefactos e histórico

O protocolo e os resultados finais V2-C estão preservados no commit [`aebb592`](https://github.com/LEI-122678/SRSI-IoT-Anomaly-Detection/commit/aebb592). As páginas técnicas encontram-se em [`docs/`](docs/README.md).

Os CSV/JSON versionados são a referência para as tabelas e gráficos científicos. Dados brutos, partições e checkpoints locais devem ser arquivados separadamente quando não fizerem parte do controlo de versões.

---

**SRSI — Deteção de Anomalias em Redes IoT** · [Consultar a documentação](docs/README.md)
