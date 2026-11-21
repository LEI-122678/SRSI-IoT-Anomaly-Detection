<!-- Gerado por gerar_documentacao_srsi.py; pode ser atualizado pelo script. -->

# Comparação entre aprendizagem centralizada e federada

[← Resultados](README.md) · [Experiência V2-C](../experiments/v2c.md)

## V2-C — Métricas globais da avaliação final

O protocolo fixado antecipadamente utiliza 18 checkpoints, três seeds e limiar 0,5. As médias seguintes foram calculadas sobre os mesmos três IPs reservados.

| Modelo | F1 centralizado | F1 federado | Δ F1 (federado − centralizado) |
| --- | --- | --- | --- |
| GRU | 0,7750 | 0,9314 | 0,1565 |
| LSTM | 0,7758 | 0,7705 | -0,0053 |
| Transformer | 0,7998 | 0,8105 | 0,0107 |

### FPR e recall

| Modelo | FPR centralizado | FPR federado | Recall centralizado | Recall federado |
| --- | --- | --- | --- | --- |
| GRU | 0,3453 | 0,0808 | 0,9821 | 0,9839 |
| LSTM | 0,3429 | 0,3561 | 0,9813 | 0,9813 |
| Transformer | 0,3006 | 0,2795 | 0,9859 | 0,9851 |

A **GRU federada** apresenta a maior melhoria de F1 e uma redução expressiva da FPR. Para a **LSTM**, a abordagem federada não oferece melhoria global consistente; para o **Transformer**, a diferença média é modesta.

## IPs com tráfego misto

Os dois IPs mistos somam 2 728 janelas, 109 ataques e 2 619 janelas normais. O F1 foi calculado agregando as contagens `TP`, `FP` e `FN` antes de aplicar a fórmula, e só depois calculando a média entre seeds.

| Modelo | F1 centralizado | F1 federado | Δ F1 |
| --- | --- | --- | --- |
| GRU | 0,1475 | 0,4212 | 0,2737 |
| LSTM | 0,1462 | 0,1471 | 0,0010 |
| Transformer | 0,1788 | 0,1887 | 0,0099 |

A diferença entre o F1 global da GRU federada (**0,9314**) e o F1 nos IPs mistos (**0,4212**) é essencial para uma leitura correta: o IP que contém só ataques facilita uma pontuação agregada elevada, mesmo quando existem dificuldades nos cenários raros/mistos.

## Interpretação dos resultados

- Os resultados mostram desempenho nos **três IPs selecionados** e não demonstram superioridade universal da aprendizagem federada.
- O treino federado distribui os dados pelos clientes lógicos, mas **FedAvg não assegura privacidade criptográfica** nem impede por si só inferência a partir de atualizações.
- Os clientes possuem percentagens de ataque heterogéneas (*non-IID*) e a distribuição não representa necessariamente cinco dispositivos físicos.
- A seleção dos IPs reservados não foi aleatória; os resultados não autorizam intervalos populacionais de confiança.

As métricas originais e suas contagens estão em [`v2c_final_evaluation_models.csv`](../../results/tables/v2c_final_evaluation_models.csv), [`v2c_final_evaluation_per_ip.csv`](../../results/tables/v2c_final_evaluation_per_ip.csv) e no [JSON consolidado](../../results/v2c_final_results_consolidated.json).
