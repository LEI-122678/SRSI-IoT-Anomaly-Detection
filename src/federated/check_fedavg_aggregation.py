
import copy

import numpy as np
import torch

from flwr.app import ArrayRecord, MetricRecord, RecordDict
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
)

from src.federated.train_fedavg_gru import (
    SEED,
    NUM_CLIENTS,
    PARTITION_DIR,
    build_model,
    load_data,
    train_client,
    fedavg,
    evaluate,
)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, y_train = load_data("train")
    X_val, y_val = load_data("validation")

    initial_model = build_model()
    initial_state = copy.deepcopy(
        initial_model.state_dict()
    )

    manual_updates = []
    flower_records = []

    print("=== COMPARAÇÃO FEDAVG ===")

    for client_id in range(NUM_CLIENTS):
        indices = np.load(
            PARTITION_DIR
            / f"client_{client_id}_indices.npy"
        )

        X_local = X_train[indices]
        y_local = y_train[indices]

        local_seed = SEED + 100 + client_id

        torch.manual_seed(local_seed)
        np.random.seed(local_seed)

        state, count, loss = train_client(
            initial_state,
            X_local,
            y_local,
            client_id,
            round_id=1
        )

        manual_updates.append(
            (state, count, loss)
        )

        flower_records.append(
            RecordDict({
                "arrays": ArrayRecord(state),
                "metrics": MetricRecord({
                    "num-examples": count,
                    "train_loss": float(loss)
                })
            })
        )

        print(
            f"Cliente {client_id}: "
            f"{count} janelas | "
            f"Loss={loss:.6f}"
        )

    # Agregação da nossa implementação.
    manual_state = fedavg(manual_updates)

    # Agregação usada internamente pelo
    # FedAvg da versão instalada do Flower.
    flower_arrays = aggregate_arrayrecords(
        flower_records,
        "num-examples"
    )

    flower_state = (
        flower_arrays.to_torch_state_dict()
    )

    if manual_state.keys() != flower_state.keys():
        raise ValueError(
            "As chaves dos modelos agregados diferem."
        )

    differences = {}

    for key in manual_state:
        a = manual_state[key]
        b = flower_state[key]

        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(
                f"Shape ou dtype diferente: {key}"
            )

        differences[key] = (
            (a - b).abs().max().item()
        )

    max_difference = max(differences.values())

    print("\n=== COMPARAÇÃO DOS PESOS ===")
    print("Diferença máxima:", max_difference)

    # A implementação manual utiliza float64
    # na acumulação; Flower pode ter outra
    # ordem ou precisão de operações.
    tolerance = 1e-6

    equivalent = all(
        torch.allclose(
            manual_state[key],
            flower_state[key],
            rtol=0,
            atol=tolerance
        )
        for key in manual_state
    )

    print("Tolerância absoluta:", tolerance)
    print(
        "Resultado:",
        "AGREGAÇÃO EQUIVALENTE"
        if equivalent
        else "DIFERENÇAS ACIMA DA TOLERÂNCIA"
    )

    manual_model = build_model()
    manual_model.load_state_dict(manual_state)

    flower_model = build_model()
    flower_model.load_state_dict(flower_state)

    manual_metrics = evaluate(
        manual_model, X_val, y_val
    )

    flower_metrics = evaluate(
        flower_model, X_val, y_val
    )

    print("\n=== VALIDAÇÃO APÓS A RONDA 1 ===")
    print(
        f"Manual | Loss={manual_metrics['loss']:.8f} "
        f"| F1={manual_metrics['f1']:.8f}"
    )
    print(
        f"Flower | Loss={flower_metrics['loss']:.8f} "
        f"| F1={flower_metrics['f1']:.8f}"
    )

    if not equivalent:
        print("\nDiferenças por tensor:")
        for key, diff in differences.items():
            print(f"{key}: {diff}")


if __name__ == "__main__":
    main()
