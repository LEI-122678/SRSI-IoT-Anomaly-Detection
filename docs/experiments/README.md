# Experiências SRSI

[← Documentação](../README.md) · [Metodologia](../methodology.md) · [Resultados](../results/README.md)

O projeto contém vários estudos exploratórios, além de dois protocolos relevantes para a comparação centralizada/federada.

## V2-B — Divisão temporal

[Documentação da experiência V2-B](v2b.md)

As janelas são separadas por datas de treino, validação e teste. O desenho permite observar comportamento em dias distintos, mas **não isola IPs de origem**: parte dos IPs da validação também ocorre no treino. Por isso, valores elevados de F1 nessa validação não demonstram generalização para dispositivos/IPs novos.

## V2-C — Separação por IP de origem

[Documentação da experiência V2-C](v2c.md)

A validação interna utiliza IPs diferentes dos do treino interno. Depois de congeladas as durações, os modelos finais são treinados com 37 IPs e avaliados uma única vez em três IPs reservados.

**Nota:** os três IPs reservados foram escolhidos deliberadamente após análise das distribuições; não são uma amostra aleatória. IP de origem não é necessariamente um dispositivo físico.

## Leitura recomendada

Começa por [metodologia](../methodology.md), segue para [V2-B](v2b.md) e [V2-C](v2c.md) e termina em [comparação centralizada/federada](../results/centralized-vs-federated.md) e [limitações](../results/limitations.md).
