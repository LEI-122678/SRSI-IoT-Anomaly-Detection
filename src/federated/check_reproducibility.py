
import copy

import numpy as np
import torch

from src.federated.train_fedavg_gru import (
    build_model,
    load_data,
    train_client,
    PARTITION_DIR,
    SEED,
)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, y_train = load_data("train")

    indices = np.load(
        PARTITION_DIR / "client_0_indices.npy"
    )

    X_local = X_train[indices]
    y_local = y_train[indices]

    # Mesmos pesos iniciais nas duas execuções.
    global_model = build_model()
    initial_state = copy.deepcopy(
        global_model.state_dict()
    )

    print("=== TESTE DE REPRODUTIBILIDADE ===")
    print("Cliente: 0")
    print("Janelas:", len(y_local))
    print("Ronda: 1")

    results = []

    for execution in range(1, 3):
        # Reiniciar o estado aleatório antes
        # de cada execução do treino local.
        local_seed = SEED + 1 * 100 + 0

        torch.manual_seed(local_seed)
        np.random.seed(local_seed)

        state, count, loss = train_client(
            initial_state,
            X_local,
            y_local,
            client_id=0,
            round_id=1
        )

        results.append((state, loss))

        print(
            f"Execução {execution} | "
            f"Loss: {loss:.12f}"
        )

    state_1, loss_1 = results[0]
    state_2, loss_2 = results[1]

    max_difference = max(
        (
            state_1[key] - state_2[key]
        ).abs().max().item()
        for key in state_1
    )

    print("\n=== COMPARAÇÃO ===")
    print(
        "Diferença absoluta da loss:",
        abs(loss_1 - loss_2)
    )
    print(
        "Diferença máxima dos pesos:",
        max_difference
    )

    if max_difference == 0 and loss_1 == loss_2:
        print("RESULTADO: REPRODUTÍVEL")
    else:
        print("RESULTADO: DIFERENÇAS DETETADAS")


if __name__ == "__main__":
    main()
