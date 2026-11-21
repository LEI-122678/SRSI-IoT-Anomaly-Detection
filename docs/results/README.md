<!-- Gerado por gerar_documentacao_srsi.py; pode ser atualizado pelo script. -->

# Resultados experimentais

[← Índice da documentação](../README.md)

A análise distingue as experiências **V2-B (divisão temporal)** e **V2-C (separação por IP)**. O conjunto de avaliação V2-C já foi utilizado uma vez no protocolo congelado; não é fonte para novas escolhas de hiperparâmetros.

## Páginas

- [Comparação centralizada vs. federada](centralized-vs-federated.md)
- [Limitações e validade das conclusões](limitations.md)
- [Experiência V2-B](../experiments/v2b.md)
- [Experiência V2-C](../experiments/v2c.md)

## Dados publicados

Os CSV em [`results/tables/`](../../results/tables/) e o consolidado [`results/v2c_final_results_consolidated.json`](../../results/v2c_final_results_consolidated.json) contêm as métricas reais, enquanto os checkpoints `.pt` se mantêm locais.

- `v2c_final_evaluation_models.csv`: 18 linhas, uma por modelo, abordagem e seed.
- `v2c_final_evaluation_summary.csv`: 6 linhas, médias e desvios-padrão.
- `v2c_final_evaluation_per_ip.csv`: 54 linhas, métricas por IP.
- `v2c_final_mixed_ips_summary.csv`: 6 linhas, análise derivada dos dois IPs mistos.

Não interpretar resultados de validação interna como se fossem resultados de avaliação final.
