
# Demonstração federada e ataque de model poisoning

[← Índice da documentação](README.md)

## 1. Objetivo

Esta demonstração utiliza Flower e FedAvg para observar
o treino federado de uma GRU em dois cenários:

- **Benigno:** cinco clientes executam treino local e
  enviam atualizações legítimas ao servidor.
- **Adversarial:** quatro clientes são legítimos e
  um cliente adultera as atualizações antes da agregação.

O objetivo é avaliar o impacto de um participante
malicioso na integridade da aprendizagem federada.

Esta experiência é complementar à avaliação final V2-C
e não modifica os seus 18 modelos ou resultados.

## 2. Configuração experimental

| Parâmetro | Valor |
|---|---|
| Arquitetura | GRU |
| Algoritmo | FedAvg |
| Plataforma | Flower |
| Clientes lógicos | 5 |
| Clientes maliciosos | 1 (cliente 4) |
| Rondas federadas | 20 |
| Épocas locais | 1 por cliente e ronda |
| Batch size | 32 |
| Learning rate | 0.001 |
| Semente | 42 |
| Janelas de treino | 15 951 |
| Janelas de validação | 5 060 |
| Fator de ataque | -10 |

A experiência utiliza apenas as partições internas de
desenvolvimento. Os três IPs reservados para a
avaliação final V2-C não são utilizados.

## 3. Funcionamento do ataque

O cliente malicioso recebe os parâmetros globais,
executa o treino local normalmente e, antes de enviar
os novos parâmetros ao servidor, aplica:

θ_mal = θ_global + α × (θ_local - θ_global)

com α = -10.

Esta operação inverte a direção da atualização local
e amplifica a sua magnitude dez vezes.

O ataque corresponde a **model update poisoning**:
altera os parâmetros enviados, não os dados de treino
nem as etiquetas.

O servidor utiliza FedAvg convencional, sem mecanismos
de deteção ou rejeição das atualizações maliciosas.

## 4. Scripts principais

| Script | Responsabilidade |
|---|---|
| `src/federated/demo_20rounds_benign.py` | Iniciar a simulação benigna |
| `src/federated/demo_20rounds_benign_server.py` | Configuração do servidor benigno |
| `src/federated/demo_flower_client.py` | Cliente legítimo |
| `src/federated/demo_20rounds_malicious.py` | Iniciar a simulação adversarial |
| `src/federated/demo_20rounds_malicious_server.py` | Configuração do servidor adversarial |
| `src/federated/demo_malicious_client.py` | Implementação do cliente atacante |
| `src/federated/demo_compare.py` | Comparação dos históricos |

## 5. Execução

A partir da raiz do projeto, num ambiente Python
com as dependências instaladas:

### Cenário benigno

```powershell
python -m src.federated.demo_20rounds_benign
```

### Cenário adversarial

```powershell
python -m src.federated.demo_20rounds_malicious
```

**Atenção:** as simulações executam novamente o treino.
Não são necessárias para consultar os resultados já
obtidos e exigem os dados de desenvolvimento preparados.

Antes de repetir uma experiência, deve confirmar-se o
diretório de saída definido nos scripts, evitando
substituir resultados existentes.

## 6. Resultados observados

Métricas na ronda 20, calculadas sobre a validação
interna de 5 060 janelas:

| Métrica | Benigno | Malicioso verificado |
|---|---:|---:|
| F1-score | 0.8705 | 0.0000 |
| Recall | 0.8493 | 0.0000 |
| FPR | 0.0619 | 0.2575 |
| Perda de validação | 0.2191 | 34.5546 |

A execução adversarial verificada guardou também
o modelo global da ronda 20.

A avaliação independente desse checkpoint produziu:

| | Previsto normal | Previsto ataque |
|---|---:|---:|
| Real normal | 2338 | 811 |
| Real ataque | 1911 | 0 |

O modelo malicioso não identificou corretamente
nenhuma das 1 911 janelas de ataque.

## 7. Artefactos experimentais

Os resultados originais foram preservados localmente em:

- `results/demonstration/flower_20rounds_benign/`
- `results/demonstration/flower_20rounds_malicious_neg10/`

A execução adversarial utilizada para verificar
independentemente o checkpoint da ronda 20 está em:

`results/demonstration/flower_20rounds_malicious_neg10_verified/`

Os artefactos incluem históricos de validação,
relatórios JSON e checkpoints PyTorch.

O melhor checkpoint é selecionado pela menor perda
de validação e pode não corresponder à ronda 20.

Os históricos de validação em CSV, os relatórios JSON
e os quatro gráficos comparativos encontram-se
disponíveis no repositório GitHub, em
`results/demonstration/`.

Os checkpoints PyTorch (`.pt`) permanecem armazenados
localmente e não foram publicados no GitHub.

Os gráficos comparativos podem ser produzidos
a partir dos históricos correspondentes, mantendo
a identificação das execuções utilizadas.

## 8. Limitações

- Foi testada apenas a arquitetura GRU.
- Foi utilizada uma única semente.
- O fator de ataque foi fixado em -10.
- O ataque esteve ativo desde a primeira ronda.
- Não foram implementadas defesas contra poisoning.
- Não foram avaliados dispositivos IoT físicos.
- Não foram testados ataques à confidencialidade.
- Não foi utilizada a avaliação final reservada V2-C.

Os resultados demonstram degradação grave do modelo
global nesta configuração, não uma vulnerabilidade
universal de todos os sistemas federados.

## 9. Relação com a avaliação V2-C

A experiência principal V2-C compara GRU, LSTM e
Transformer nas modalidades centralizada e federada,
em três sementes, com 18 checkpoints finais.

Esta demonstração de segurança é independente dessa
avaliação e utiliza exclusivamente dados internos.

[← Reprodutibilidade](reproducibility.md)