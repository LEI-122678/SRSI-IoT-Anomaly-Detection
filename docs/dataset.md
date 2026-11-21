<!-- Gerado por gerar_documentacao_srsi.py; pode ser atualizado pelo script. -->

# Dataset e representação dos dados

[Índice](README.md) · [Arquitetura](architecture.md) · [Dataset](dataset.md) · [Metodologia](methodology.md) · [Reprodutibilidade](reproducibility.md)

## Fonte

O projeto utiliza ficheiros de tráfego de rede do **ToN_IoT**, colocados localmente em `data/raw/ton_iot/` com nomes como `Network_dataset_1.csv`. Os dados originais não são incluídos nesta documentação nem devem ser publicados inadvertidamente no GitHub.

O pré-processamento principal está em `src/preprocessing/`. O ficheiro `src/preprocessing/create_v2_minute_features.py` prepara as características temporais usadas nas experiências V2-B/V2-C.

## Representação das sequências

Cada amostra contém **5 minutos consecutivos** do mesmo `src_ip`, representados por **6 características por minuto**: número de ligações; bytes de origem; bytes de destino; pacotes de origem; pacotes de destino; duração média.

A dimensão de entrada é `(N, 5, 6)`. A etiqueta binária corresponde à presença de ligações de ataque **no último minuto** da sequência. A preparação inclui `log1p` e normalização ajustada sem consultar as amostras da avaliação reservada.

As janelas deslizantes sobrepostas partilham observações e **não são amostras estatisticamente independentes**.

## V2-B — Divisão temporal

A experiência V2-B usa sequências preparadas em `data/processed/ton_iot_v2/temporal_generalization/`:

| Divisão | Janelas | IPs de origem |
|---|---:|---:|
| Treino | 25 267 | 40 |
| Validação | 4 031 | 15 |
| Teste | 9 759 | 35 |

A divisão usa treino até 26/04/2019, validação em 27/04/2019 e teste em 28–29/04/2019. Os 15 IPs da validação também aparecem no treino; além disso, a validação está segregada por classe em função do IP. Este facto compromete a interpretação dos resultados extremamente elevados de validação como prova de generalização.

## V2-C — Separação por IP de origem

A preparação encontra-se em `src/preprocessing/prepare_v2_ip_generalization.py` e `src/preprocessing/prepare_v2c_internal_validation.py`.

| Fase | Janelas | IPs de origem | Utilização |
|---|---:|---:|---|
| Treino interno | 15 951 | 30 | Escolha da duração de treino |
| Validação interna | 5 060 | 7 | Comparação e seleção interna |
| Treino final | 21 011 | 37 | Ajuste dos 18 modelos definitivos |
| Avaliação final | 4 256 | 3 | Avaliação única reservada |

O treino interno e a validação interna têm conjuntos de IPs disjuntos. A normalização interna é ajustada só aos IPs de treino interno. No treino final, a normalização é ajustada aos 37 IPs de treino.

### Composição da avaliação final

| IP de origem | Janelas | Normal | Ataque |
| --- | --- | --- | --- |
| `192.168.1.193` | 1 147 | 1066 | 81 |
| `192.168.1.195` | 1 581 | 1553 | 28 |
| `192.168.1.30` | 1 528 | 0 | 1528 |

Os IPs foram escolhidos após inspecionar distribuições, não por amostragem aleatória. A experiência demonstra desempenho nestes três IPs reservados, **não** garante generalização a quaisquer IPs ou a um período temporal futuro.

## Ficheiros locais

- `data/processed/ton_iot_v2/ip_generalization/train.npz`: treino final.
- `data/processed/ton_iot_v2/ip_generalization/evaluation.npz`: avaliação final reservada; não voltar a usar para afinação.
- `data/processed/ton_iot_v2/ip_generalization/internal_validation/`: treino e validação internos.
- `data/processed/ton_iot_v2/ip_generalization/federated_internal/`: índices dos clientes internos.
- `data/processed/ton_iot_v2/ip_generalization/federated_final/`: índices dos clientes finais.

Os dados e os índices locais são artefactos de execução, não devem ser confundidos com os relatórios JSON e CSV publicados.
