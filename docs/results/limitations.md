# Limitações e validade das conclusões

[← Resultados](README.md) · [Experiência V2-B](../experiments/v2b.md) · [Experiência V2-C](../experiments/v2c.md)

## 1. Sobreposição entre origens na V2-B

A validação V2-B partilhava todos os seus 15 IPs com o treino. A distribuição das etiquetas estava segregada por IP: um baseline que atribuía a classe maioritária de treino a cada IP obtinha F1 perfeito. Os modelos não recebiam o identificador IP diretamente, mas podiam beneficiar de padrões correlacionados com a origem. Por isso, os valores de validação V2-B próximos de 1 não provam generalização para IPs novos.

## 2. Seleção dos IPs reservados na V2-C

Os três IPs da avaliação final foram selecionados deliberadamente após observar distribuições de dados, e não amostrados aleatoriamente. A avaliação é separada do treino e da validação interna, mas **não é uma amostra inteiramente cega nem representativa de todos os ambientes IoT**.

## 3. IP de origem não equivale a dispositivo físico

A partição por `src_ip` testa generalização entre identificadores de origem observados. Tradução de endereços, IPs partilhados, dinâmicos ou outras particularidades da rede impedem concluir que cada IP equivale a um dispositivo independente.

## 4. Janelas temporalmente sobrepostas

As amostras são sequências deslizantes de cinco minutos. Janelas adjacentes partilham observações e não podem ser tratadas como medições estatisticamente independentes. Desvios-padrão entre apenas três seeds refletem variabilidade das execuções, **não** intervalos de confiança para novas redes ou novos IPs.

## 5. Desempenho agregado pode ser enganador

Um dos IPs reservados contém exclusivamente ataques (1 528 janelas). O F1 global da GRU federada é 0,9314, enquanto nos dois IPs com tráfego normal e malicioso é 0,4212. Estes valores não são contraditórios: mudam a composição e a prevalência das classes. Deve apresentar-se também a precisão, o recall, a FPR e as matrizes de confusão por IP.

A FPR é indefinida para IPs sem exemplos normais e não deve ser representada artificialmente como zero.

## 6. Ataques raros e falsos positivos

O IP `.195` contém 28 ataques em 1 581 janelas. Mesmo modelos com recall elevado a nível global podem apresentar precisão muito baixa em origens onde as anomalias são raras. Estes erros são especialmente relevantes para um possível sistema real de alertas.

## 7. Validade temporal e externa

A V2-C testa exclusivamente a separação por IP dentro do período de dados utilizado. **Não demonstra generalização para datas futuras**, redes diferentes, novos tipos de ataque ou sistemas em produção. A experiência V2-B usa divisão temporal, mas tem os fatores de confusão já referidos.

## 8. Segurança e privacidade no FedAvg

FedAvg permite treinar sem centralizar diretamente os exemplos brutos em cada ronda, mas as atualizações do modelo podem revelar informação. A simulação não demonstra, por si só, segurança, confidencialidade, resistência a ataques adversariais ou garantias formais de privacidade.

## 9. Comparação e seleção experimental

As durações finais foram selecionadas por validação interna e congeladas antes da avaliação. As abordagens possuem dinâmicas de otimização distintas (épocas centralizadas vs. rondas federadas), pelo que o estudo compara **configurações experimentais específicas**, e não custos computacionais ou orçamentos de otimização perfeitamente equivalentes.

## 10. Utilização responsável dos resultados

Os resultados reservados já foram consultados e devem permanecer finais. Uma extensão experimental que procure melhorar a precisão ou reduzir falsos positivos exige um **novo conjunto externo ou novo protocolo pré-registado**, sem reutilizar os três IPs reservados para escolher hiperparâmetros.

[← Documentação](../README.md)
