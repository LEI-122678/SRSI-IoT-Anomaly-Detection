# SRSI — Deteção de Anomalias em Redes IoT

**Comparação de aprendizagem centralizada e federada com GRU, LSTM e Transformer sobre o conjunto de dados ToN_IoT.**

Este repositório contém uma investigação experimental sobre a identificação binária de tráfego anómalo em redes *Internet of Things* (IoT), com particular atenção à **generalização para endereços IP de origem não utilizados no treino**. São comparadas três arquiteturas de aprendizagem profunda — **GRU**, **LSTM** e **Transformer** — em configurações **centralizada** e **federada (Flower/FedAvg)**.

> **Estado:** os 18 modelos da experiência final V2-C (3 arquiteturas × 2 abordagens × 3 sementes) já foram treinados e avaliados. O protocolo da avaliação final foi fixado antes da consulta aos IPs reservados. Os resultados finais não devem ser utilizados para reajustar os modelos.

## Índice

- [Objetivos](#objetivos)
- [Dados e representação](#dados-e-representação)
- [Desenho experimental](#desenho-experimental)
- [Resultados finais V2-C](#resultados-finais-v2-c)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Instalação](#instalação)
- [Execução e reprodutibilidade](#execução-e-reprodutibilidade)
- [Limitações](#limitações)
- [Artefactos e documentação](#artefactos-e-documentação)

## Objetivos

1. Explorar padrões de tráfego normal e malicioso no ToN_IoT.
2. Construir sequências temporais de características agregadas de rede.
3. Comparar GRU, LSTM e Transformer sob treino centralizado e federado.
4. Investigar o impacto de clientes federados com distribuições heterogéneas (*non-IID*).
5. Avaliar o desempenho em IPs de origem ausentes do treino, incluindo resultados separados por IP para evitar conclusões baseadas apenas em métricas agregadas.

## Dados e representação

A fonte utilizada é o **ToN_IoT**, na componente de tráfego de rede. Os ficheiros originais encontram-se localmente em `data/raw/ton_iot/`, com nomes como `Network_dataset_1.csv`. **Os dados brutos não são distribuídos neste README nem devem ser assumidos como presentes num clone do GitHub.** Consulte a fonte oficial do dataset e as respetivas condições de utilização antes de o descarregar.

A experiência V2 utiliza agregações por minuto e por `src_ip`, com seis características numéricas:

| Característica | Descrição |
|---|---|
| `connections` | Número de ligações |
| `src_bytes_sum` | Soma dos bytes enviados |
| `dst_bytes_sum` | Soma dos bytes recebidos |
| `src_pkts_sum` | Soma dos pacotes de origem |
| `dst_pkts_sum` | Soma dos pacotes de destino |
| `duration_mean` | Duração média das ligações |

Cada exemplo é constituído por uma sequência de **5 minutos consecutivos**, representada por um tensor de dimensão `(5, 6)`. O rótulo binário indica a presença de ligações maliciosas **no último minuto** da sequência. O pré-processamento inclui transformação `log1p` e normalização, com ajuste dos parâmetros apenas sobre os dados de treino da respetiva experiência.

As janelas consecutivas podem sobrepor-se; por isso, não devem ser consideradas observações estatisticamente independentes.

## Desenho experimental

### V2-B — Divisão temporal

Na experiência V2-B, os dados foram separados por data:

| Conjunto | Janelas | IPs distintos |
|---|---:|---:|
| Treino (até 26/04/2019) | 25 267 | 40 |
| Validação (27/04/2019) | 4 031 | 15 |
| Teste (28–29/04/2019) | 9 759 | 35 |

Foram estudadas as três arquiteturas e três sementes (`42`, `123` e `2026`) em ambas as abordagens. Na federação, foram utilizados **cinco clientes lógicos** e FedAvg.

**Limitação metodológica importante:** os 15 IPs da validação também aparecem no treino, e a distribuição de classes por IP permite uma regra trivial baseada na classe maioritária de cada IP obter F1 de 1 nessa validação. Portanto, as métricas elevadas da V2-B não constituem evidência suficiente de generalização para IPs desconhecidos.

### V2-C — Separação por IP de origem

Para investigar a generalização entre IPs, a V2-C separa os dados da fase de treino temporal por `src_ip`, evitando que o mesmo IP esteja simultaneamente nos conjuntos de treino e avaliação.

| Conjunto | Janelas | IPs | Normal | Ataque |
|---|---:|---:|---:|---:|
| Treino final | 21 011 | 37 | 15 537 | 5 474 |
| Avaliação reservada | 4 256 | 3 | 2 619 | 1 637 |

A seleção de épocas/rondas foi feita utilizando uma **validação interna separada por IP** dentro dos 37 IPs disponíveis: **30 IPs para treino interno (15 951 janelas)** e **7 IPs para validação interna (5 060 janelas)**. Após essa etapa, o protocolo foi congelado e os modelos definitivos foram treinados com os 37 IPs.

Os três IPs da avaliação reservada foram `192.168.1.30`, `192.168.1.193` e `192.168.1.195`. Esta escolha foi **deliberada após inspeção das distribuições**; não equivale a uma amostra aleatória nem a um teste totalmente cego.

#### Protocolo final fixado

| Configuração | Centralizado | Federado |
|---|---|---|
| Arquiteturas | GRU, LSTM, Transformer | GRU, LSTM, Transformer |
| Sementes | 42, 123, 2026 | 42, 123, 2026 |
| Duração — GRU | 1 época | 15 rondas |
| Duração — LSTM | 1 época | 4 rondas |
| Duração — Transformer | 1 época | 3 rondas |
| Clientes | — | 5 clientes lógicos |
| Épocas locais | — | 1 por ronda |
| Tamanho do batch | 32 | 32 |
| Taxa de aprendizagem | 0,001 | 0,001 |
| Dimensão oculta | 32 | 32 |
| Limiar de decisão | 0,5 | 0,5 |
| Seleção do checkpoint | Duração fixa | Última agregação global |

As durações foram definidas com base na mediana da melhor época/ronda obtida na validação interna. Na fase final, **não houve validação, *early stopping* nem seleção de modelos com recurso aos IPs reservados**.

Os cinco clientes federados finais contêm conjuntos de IPs sem sobreposição. Têm aproximadamente o mesmo número de janelas, mas proporções de ataques substancialmente diferentes (de **2,90%** a **53,89%**), constituindo uma distribuição heterogénea.

## Resultados finais V2-C

### Métricas globais

Resultados na avaliação reservada de **4 256 janelas**, reportados como **média ± desvio-padrão das três sementes**:

| Modelo | Abordagem | F1 | Recall | FPR |
|---|---|---:|---:|---:|
| GRU | Centralizada | 0,7750 ± 0,0010 | 0,9821 | 0,3453 |
| **GRU** | **Federada** | **0,9314 ± 0,0183** | **0,9839** | **0,0808** |
| LSTM | Centralizada | 0,7758 ± 0,0064 | 0,9813 | 0,3429 |
| LSTM | Federada | 0,7705 ± 0,0359 | 0,9813 | 0,3561 |
| Transformer | Centralizada | 0,7998 ± 0,0245 | 0,9859 | 0,3006 |
| Transformer | Federada | 0,8105 ± 0,0233 | 0,9851 | 0,2795 |

A **GRU federada** alcançou o melhor F1 médio nesta avaliação, com uma redução relevante da taxa de falsos positivos face à GRU centralizada. O mesmo padrão de vantagem **não é universal**: as diferenças são pequenas na LSTM e no Transformer, e dependem da arquitetura e da semente.

### Análise dos IPs com tráfego misto

O IP `192.168.1.30` tem **1 528 janelas de ataque e nenhuma janela normal**, o que favorece métricas agregadas elevadas e torna a FPR desse IP indefinida. Para interpretar o comportamento em contextos com as duas classes, foi feita uma análise complementar dos IPs `.193` e `.195` (**2 728 janelas, das quais 109 ataques**).

O F1 abaixo foi calculado **após agregar as matrizes de confusão dos dois IPs, para cada modelo/semente**, e só depois calcular a média entre sementes:

| Arquitetura | Centralizado — F1 | Federado — F1 |
|---|---:|---:|
| GRU | 0,1475 | **0,4212** |
| LSTM | 0,1462 | 0,1471 |
| Transformer | 0,1788 | 0,1887 |

A GRU federada mantém a vantagem nesta análise, mas o seu F1 é substancialmente menor do que o F1 global. No IP `.195`, com apenas **28 ataques em 1 581 janelas**, a deteção permanece particularmente difícil. Assim, **não se deve apresentar o F1 global isoladamente**.

Estes resultados são descritivos para os IPs escolhidos; não demonstram superioridade estatística geral em redes IoT independentes.

## Estrutura do repositório

```text
SRSI-IoT-Anomaly-Detection/
├── configs/                    # Configuração de experiências
├── data/
│   ├── raw/                    # Dados ToN_IoT originais (locais)
│   └── processed/              # Dados e partições produzidos pelo pré-processamento
├── results/
│   ├── federated/              # Relatórios e checkpoints federados
│   ├── figures/                # Figuras experimentais
│   ├── models/                 # Relatórios e checkpoints centralizados
│   ├── tables/                 # Métricas e comparações em CSV
│   ├── v2c_final_protocol.json
│   └── v2c_final_results_consolidated.json
├── src/
│   ├── attacks/                # Módulo reservado para ataques
│   ├── evaluation/             # Avaliação e comparação dos modelos
│   ├── federated/              # Flower, FedAvg e partições dos clientes
│   ├── models/                 # GRU, LSTM e Transformer
│   ├── preprocessing/          # Exploração, agregação e preparação dos dados
│   ├── privacy/                # Módulo reservado para privacidade
│   └── training/               # Treino centralizado
├── tests/                      # Testes de ambiente
├── requirements.txt
└── README.md
```

As pastas `attacks/` e `privacy/` existem no projeto, mas a sua presença **não implica** a implementação de ataques federados nem de mecanismos de proteção de privacidade.

## Instalação

Requisitos: **Python compatível com `requirements.txt`**, ambiente virtual e acesso aos dados ToN_IoT. A implementação foi executada num ambiente local com PyTorch e Flower, incluindo simulações federadas através de Ray. As versões concretas das dependências devem ser consultadas no ficheiro `requirements.txt`.

No **PowerShell**, a partir da raiz do repositório:

```powershell
git clone https://github.com/LEI-122678/SRSI-IoT-Anomaly-Detection.git
cd SRSI-IoT-Anomaly-Detection
git switch goncalo

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Para disponibilizar os dados, coloque os ficheiros originais ToN_IoT em `data/raw/ton_iot/` e execute as etapas de preparação apropriadas. Os ficheiros `.npz` processados e os checkpoints `.pt` podem não estar versionados no GitHub.

## Execução e reprodutibilidade

> **Atenção:** os comandos abaixo documentam o fluxo experimental. **Não volte a executar a avaliação final V2-C como parte de um procedimento normal de instalação ou demonstração.** O protocolo determina uma avaliação final única, sem reajuste posterior. Um novo estudo deve usar uma nova divisão de dados e um novo protocolo previamente estabelecido.

### 1. Pré-processamento

Os módulos relevantes, executados a partir da raiz do repositório com `python -m`, incluem:

```powershell
python -m src.preprocessing.create_v2_minute_features
python -m src.preprocessing.prepare_v2_temporal_data
python -m src.preprocessing.prepare_v2_ip_generalization
python -m src.preprocessing.prepare_v2c_internal_validation
```

A ordem e os ficheiros necessários devem ser verificados antes de reconstruir os conjuntos de dados, pois as etapas dependem das saídas anteriores. O ajustamento dos *scalers* deve continuar limitado aos IPs de treino correspondentes.

### 2. Experiência interna V2-C

Os principais módulos são:

```text
src/training/train_v2c.py
src/federated/create_v2c_client_partitions.py
src/federated/run_flower_simulation_v2c.py
```

A configuração das arquiteturas e sementes é definida pelos argumentos ou variáveis de ambiente de cada script. A validação interna é utilizada apenas para selecionar durações de treino — nunca para consultar os três IPs reservados.

### 3. Treino final V2-C

O protocolo a respeitar encontra-se em [`results/v2c_final_protocol.json`](results/v2c_final_protocol.json). Os módulos utilizados foram:

```text
src/training/train_v2c_final.py
src/federated/create_v2c_final_client_partitions.py
src/federated/run_flower_simulation_v2c_final.py
```

Exemplo de configuração de uma experiência federada **num estudo independente**, após preparar os dados e confirmar que os caminhos de saída não contêm resultados existentes:

```powershell
$env:SRSI_MODEL = "gru"
$env:SRSI_SEED = "42"
python -m src.federated.run_flower_simulation_v2c_final
Remove-Item Env:SRSI_MODEL, Env:SRSI_SEED
```

Este comando inicia um treino, **não** uma simples verificação. Os scripts de treino final foram concebidos para evitar a substituição de checkpoints já existentes.

### 4. Consulta dos resultados já produzidos

Não é necessário executar modelos para consultar os resultados:

- [Protocolo experimental final](results/v2c_final_protocol.json)
- [Resultados finais consolidados](results/v2c_final_results_consolidated.json)
- [Métricas por modelo e semente](results/tables/v2c_final_evaluation_models.csv)
- [Resumo global](results/tables/v2c_final_evaluation_summary.csv)
- [Resultados por IP](results/tables/v2c_final_evaluation_per_ip.csv)
- [Resumo dos IPs mistos](results/tables/v2c_final_mixed_ips_summary.csv)

O módulo `src/evaluation/evaluate_v2c_final.py` documenta o procedimento da avaliação reservada já realizada. **Não deve ser usado para selecionar hiperparâmetros nem repetido como novo teste independente.**

## Limitações

- **IPs de origem não são necessariamente dispositivos físicos:** vários equipamentos podem partilhar um endereço e o mesmo equipamento pode utilizar vários endereços.
- **Escolha dos IPs de avaliação:** os três IPs foram escolhidos após análise exploratória das distribuições, pelo que a avaliação não é inteiramente cega nem representativa de uma amostra aleatória.
- **Dependência temporal:** as janelas deslizantes sobrepostas produzem exemplos correlacionados.
- **Classes heterogéneas:** um dos IPs reservados contém apenas ataques e outro contém muito poucos ataques, influenciando fortemente as métricas globais.
- **Generalização limitada:** a V2-C testa IPs de origem não vistos, **não** generalização temporal, entre instalações, sensores físicos ou novos datasets.
- **FedAvg não garante privacidade:** deixar os dados brutos nos clientes não impede, por si só, inferência ou fuga de informação através das atualizações do modelo. Não foram demonstradas garantias como privacidade diferencial ou agregação segura.
- **Comparação experimental específica:** diferenças entre o número de épocas centralizadas e as rondas federadas seguem o protocolo de validação interna; não implicam igualdade de custo computacional ou de comunicação.

## Artefactos e documentação

Os scripts e relatórios finais V2-C foram preservados no commit [`aebb592`](https://github.com/LEI-122678/SRSI-IoT-Anomaly-Detection/commit/aebb592), na branch [`goncalo`](https://github.com/LEI-122678/SRSI-IoT-Anomaly-Detection/tree/goncalo).

Os resultados em CSV/JSON são a fonte de referência para tabelas e gráficos. Os checkpoints dos modelos e os dados de origem devem ser arquivados separadamente quando não estiverem incluídos no controlo de versões.

---

**Projeto SRSI — Deteção de Anomalias em Redes IoT.** Documento técnico do estado experimental atual; sujeito a atualização da documentação, sem alteração retroativa dos resultados da avaliação final.
