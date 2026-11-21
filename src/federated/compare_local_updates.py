
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.federated.train_fedavg_gru import (
    build_model,
    load_data,
    train_client,
    PARTITION_DIR,
    SEED,
)

from src.federated.flower_client import (
    load_client_data,
)


CLIENT_ID = 0
ROUND_ID = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001


def train_flower_style(initial_state, X, y):
    local_seed = (
        SEED + ROUND_ID * 100 + CLIENT_ID
    )

    torch.manual_seed(local_seed)

    generator = torch.Generator().manual_seed(
        local_seed
    )

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    model = build_model()
    model.load_state_dict(initial_state)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_samples = 0

    for batch_X, batch_y in loader:
        optimizer.zero_grad(set_to_none=True)

        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        n = len(batch_y)
        total_loss += loss.item() * n
        total_samples += n

    return (
        copy.deepcopy(model.state_dict()),
        total_loss / total_samples
    )


def main():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Os mesmos pesos iniciais.
    initial_model = build_model()
    initial_state = copy.deepcopy(
        initial_model.state_dict()
    )

    # Dados carregados pela implementação manual.
    X_train, y_train = load_data("train")

    indices = np.load(
        PARTITION_DIR /
        f"client_{CLIENT_ID}_indices.npy"
    )

    X_manual = X_train[indices]
    y_manual = y_train[indices]

    # Dados carregados pelo cliente Flower.
    X_flower, y_flower = load_client_data(
        CLIENT_ID
    )

    assert torch.equal(X_manual, X_flower)
    assert torch.equal(y_manual, y_flower)

    local_seed = (
        SEED + ROUND_ID * 100 + CLIENT_ID
    )

    torch.manual_seed(local_seed)

    manual_state, _, manual_loss = train_client(
        initial_state,
        X_manual,
        y_manual,
        CLIENT_ID,
        ROUND_ID
    )

    flower_state, flower_loss = train_flower_style(
        initial_state,
        X_flower,
        y_flower
    )

    differences = {
        key: (
            manual_state[key] - flower_state[key]
        ).abs().max().item()
        for key in manual_state
    }

    max_diff = max(differences.values())

    print("=== COMPARAÇÃO DAS ATUALIZAÇÕES ===")
    print("Cliente:", CLIENT_ID)
    print("Ronda:", ROUND_ID)
    print("Janelas:", len(y_manual))
    print(
        "Loss manual:",
        format(manual_loss, ".12f")
    )
    print(
        "Loss Flower-style:",
        format(flower_loss, ".12f")
    )
    print(
        "Diferença absoluta da loss:",
        abs(manual_loss - flower_loss)
    )
    print(
        "Diferença máxima dos pesos:",
        max_diff
    )

    if max_diff == 0 and manual_loss == flower_loss:
        print("RESULTADO: ATUALIZAÇÕES IDÊNTICAS")
    else:
        print("RESULTADO: ATUALIZAÇÕES DIFERENTES")

        print("\nMaiores diferenças:")
        for key, diff in sorted(
            differences.items(),
            key=lambda item: item[1],
            reverse=True
        )[:5]:
            print(key, diff)


if __name__ == "__main__":
    main()
