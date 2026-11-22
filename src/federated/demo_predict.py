
from pathlib import Path

import numpy as np
import torch

from src.models.gru import GRUClassifier


ROOT = Path(__file__).resolve().parents[2]


# Modelo federado original: 20 rondas de treino,
# checkpoint selecionado na ronda 16.
# Este ficheiro sera apenas lido, nunca modificado.
CHECKPOINT = (
    ROOT / "results" / "federated"
    / "flower_v2c_gru_seed42" / "gru_best.pt"
)


VALIDATION = (
    ROOT / "data" / "processed"
    / "ton_iot_v2" / "ip_generalization"
    / "internal_validation" / "internal_validation.npz"
)


def main():
    # 1. Carregar o modelo global produzido pelo FedAvg
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    model = GRUClassifier(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )
    model.eval()

    print("\n=== MODELO GLOBAL FEDERADO ===")
    print("Modelo:", checkpoint["model_name"].upper())
    print("Clientes:", checkpoint["num_clients"])
    print("Ronda:", checkpoint["best_round"])
    print("Checkpoint:", CHECKPOINT.name)

    # 2. Carregar exclusivamente a validação interna
    with np.load(VALIDATION) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)

    print("\nJanelas de validação:", len(y))

    # 3. Selecionar alguns exemplos de cada classe
    normal_indices = np.flatnonzero(y == 0)[:5]
    attack_indices = np.flatnonzero(y == 1)[:5]

    selected = np.concatenate([
        normal_indices,
        attack_indices
    ])

    # 4. Produzir previsões
    inputs = torch.from_numpy(X[selected])

    with torch.no_grad():
        logits = model(inputs).reshape(-1)
        probabilities = torch.sigmoid(logits).cpu().numpy()

    predictions = (probabilities >= 0.5).astype(int)

    print("\n=== PREVISOES DO MODELO GLOBAL ===")
    print("0 = NORMAL | 1 = ATAQUE\n")

    for index, probability, predicted in zip(
        selected, probabilities, predictions
    ):
        actual = int(y[index])
        result = "CORRETO" if predicted == actual else "ERRADO"

        print(
            f"Janela {index:5d} | "
            f"Real: {actual} | "
            f"Previsto: {predicted} | "
            f"P(ataque): {probability:.4f} | "
            f"{result}"
        )

    print("\nDemonstracao de inferencia concluida.")


if __name__ == "__main__":
    main()
