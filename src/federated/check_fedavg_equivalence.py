
import copy

import numpy as np
import torch

from src.federated.train_fedavg_gru import (
    build_model,
    fedavg,
    load_data,
    train_client,
    evaluate,
    PARTITION_DIR,
    NUM_CLIENTS,
    SEED,
)


ROUNDS = 2


def main():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, y_train = load_data("train")
    X_val, y_val = load_data("validation")

    clients = []

    for client_id in range(NUM_CLIENTS):
        indices = np.load(
            PARTITION_DIR
            / f"client_{client_id}_indices.npy"
        )

        clients.append((
            X_train[indices],
            y_train[indices]
        ))

    model = build_model()
    global_state = copy.deepcopy(
        model.state_dict()
    )

    print("=== FEDAVG MANUAL — CONTROLO DE SEEDS ===")

    for round_id in range(1, ROUNDS + 1):
        updates = []

        for client_id, (X, y) in enumerate(clients):
            local_seed = (
                SEED + round_id * 100 + client_id
            )

            torch.manual_seed(local_seed)
            np.random.seed(local_seed)

            update = train_client(
                global_state,
                X,
                y,
                client_id,
                round_id
            )

            updates.append(update)

        global_state = fedavg(updates)
        model.load_state_dict(global_state)

        metrics = evaluate(
            model,
            X_val,
            y_val
        )

        print(
            f"Ronda {round_id:02d} | "
            f"Val Loss={metrics['loss']:.6f} | "
            f"F1={metrics['f1']:.6f} | "
            f"Recall={metrics['recall']:.6f} | "
            f"FPR={metrics['fpr']:.6f}"
        )


if __name__ == "__main__":
    main()
