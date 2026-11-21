
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
)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, y_train = load_data("train")

    model = build_model()
    initial_state = copy.deepcopy(model.state_dict())

    records = []

    for client_id in range(NUM_CLIENTS):
        indices = np.load(
            PARTITION_DIR / f"client_{client_id}_indices.npy"
        )

        local_seed = SEED + 100 + client_id
        torch.manual_seed(local_seed)
        np.random.seed(local_seed)

        state, count, loss = train_client(
            initial_state,
            X_train[indices],
            y_train[indices],
            client_id,
            round_id=1
        )

        records.append(
            RecordDict({
                "arrays": ArrayRecord(state),
                "metrics": MetricRecord({
                    "num-examples": count
                })
            })
        )

    # Agregar os mesmos cinco resultados
    # em duas ordens diferentes.
    forward = aggregate_arrayrecords(
        records,
        "num-examples"
    ).to_torch_state_dict()

    reverse = aggregate_arrayrecords(
        list(reversed(records)),
        "num-examples"
    ).to_torch_state_dict()

    differences = {
        key: (
            forward[key] - reverse[key]
        ).abs().max().item()
        for key in forward
    }

    max_difference = max(differences.values())

    exactly_equal = all(
        torch.equal(forward[key], reverse[key])
        for key in forward
    )

    print("=== ORDEM DE AGREGAÇÃO FEDAVG ===")
    print("Clientes:", NUM_CLIENTS)
    print("Ordem A: 0, 1, 2, 3, 4")
    print("Ordem B: 4, 3, 2, 1, 0")
    print("Diferença máxima:", max_difference)
    print("Pesos exatamente iguais:", exactly_equal)
    print(
        "Equivalentes com tolerância 1e-6:",
        all(
            torch.allclose(
                forward[key],
                reverse[key],
                atol=1e-6,
                rtol=0
            )
            for key in forward
        )
    )


if __name__ == "__main__":
    main()
