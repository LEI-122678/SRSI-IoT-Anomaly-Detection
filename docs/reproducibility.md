# Reprodutibilidade e execução

[← Índice da documentação](README.md) · [Metodologia](methodology.md)

## Ambiente

Os módulos Python devem ser executados a partir da raiz do repositório com um ambiente virtual e as dependências de `requirements.txt` instaladas.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Organização dos artefactos

Os dados de origem ToN_IoT são mantidos em `data/raw/ton_iot/`. Os dados processados situam-se em `data/processed/`. Os relatórios dos treinos estão em `results/models/` e `results/federated/` e as tabelas em `results/tables/`.

Os checkpoints binários `.pt`, os conjuntos `.npz` e os índices `.npy` podem estar excluídos do Git; a disponibilidade desses ficheiros locais deve ser assegurada antes de tentar reproduzir o treino.

## Preparação — referência a scripts

```powershell
python -m src.preprocessing.create_v2_minute_features
python -m src.preprocessing.prepare_v2_temporal_data
python -m src.preprocessing.prepare_v2_ip_generalization
python -m src.preprocessing.prepare_v2c_internal_validation
python -m src.federated.create_v2c_client_partitions
python -m src.federated.create_v2c_final_client_partitions
```

**Atenção:** estes comandos podem gerar ou substituir ficheiros processados. Executá-los apenas num ambiente de reprodução separado, com cópia de segurança; **não** são necessários para consultar os resultados atuais.

## Treino e protocolo

- Treino centralizado V2-B: `src/training/train_v2.py`.
- Treino centralizado interno V2-C: `src/training/train_v2c.py`.
- Simulação federada V2-C interna: `src/federated/run_flower_simulation_v2c.py`.
- Treino centralizado V2-C final: `src/training/train_v2c_final.py`.
- Simulação federada V2-C final: `src/federated/run_flower_simulation_v2c_final.py`.

As simulações finais usam variáveis de ambiente `SRSI_MODEL` (`gru`, `lstm`, `transformer`) e `SRSI_SEED` (42, 123, 2026). O número de rondas é lido do [protocolo congelado](../results/v2c_final_protocol.json).

Os scripts finais têm proteção contra substituir checkpoints e relatórios já existentes. **Não voltar a treinar sobre diretórios finais sem um plano de reprodução independente.**

## Avaliação final: concluída

A avaliação oficial foi executada uma vez através de `src/evaluation/evaluate_v2c_final.py`, com três IPs reservados, threshold 0,5 e 18 checkpoints. **Não repetir para seleção de hiperparâmetros.** Os resultados oficiais são:

- [`v2c_final_evaluation_models.csv`](../results/tables/v2c_final_evaluation_models.csv)
- [`v2c_final_evaluation_summary.csv`](../results/tables/v2c_final_evaluation_summary.csv)
- [`v2c_final_evaluation_per_ip.csv`](../results/tables/v2c_final_evaluation_per_ip.csv)
- [`v2c_final_mixed_ips_summary.csv`](../results/tables/v2c_final_mixed_ips_summary.csv)
- [`v2c_final_results_consolidated.json`](../results/v2c_final_results_consolidated.json)

## Verificações não destrutivas

```powershell
python -m compileall -q src tests
python -m pytest -q
```

A execução do `pytest` depende das ferramentas efetivamente instaladas. `compileall` verifica sintaxe, mas não substitui testes de integração; o projeto contém um conjunto limitado de testes automatizados.

## Preservação do histórico

O commit `aebb592` na branch `goncalo` guarda o código V2-C final, os relatórios de treino, as tabelas de avaliação e o JSON consolidado. Os pesos finais permanecem locais e não fazem parte desse commit.

## Testes automáticos e validação técnica

Executar a partir da raiz do repositório:

`python -m pytest tests/ -v`

- `tests/test_models.py`: forward e gradientes dos três modelos.
- `tests/test_fedavg.py`: agregação ponderada Flower/FedAvg
  utilizando parâmetros sintéticos.

Resultado observado: 4 testes aprovados.

Adicionalmente, os 18 checkpoints finais foram carregados
com `strict=True` e os seis artefactos finais JSON/CSV
foram verificados.


Foram também executadas simulações federadas completas
com Flower, envolvendo cinco clientes lógicos e 20 rondas
de treino, tanto num cenário benigno como num cenário
com um cliente malicioso.

No cenário adversarial, o cliente 4 adulterou as
atualizações do modelo GRU com um fator de -10.
A verificação independente do checkpoint global
da ronda 20 confirmou um F1-score de 0 e uma
taxa de falsos positivos de aproximadamente 25,75%.

Estas simulações complementam os testes automáticos,
mas não constituem uma bateria exaustiva de testes
automatizados de integração, comunicação ou segurança.

As experiências utilizaram dados de desenvolvimento.
A avaliação final reservada V2-C não foi repetida.