
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from src.models.lstm import LSTMClassifier
from src.models.gru import GRUClassifier
from src.models.transformer import TransformerClassifier


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "processed" / "ton_iot_v2"
    / "temporal_generalization"
)

MODEL_DIR = ROOT / "results" / "models" / "v2b"
OUTPUT_DIR = ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2026]
BATCH_SIZE = 128

MODELS = {
    "lstm": LSTMClassifier,
    "gru": GRUClassifier,
    "transformer": TransformerClassifier,
}

ATTACK_TYPES = [
    "backdoor", "ransomware", "mitm"
]


def load_split(name):
    with np.load(DATA_DIR / f"{name}.npz") as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)

    return X, y


def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "fpr": fp / (tn + fp) if tn + fp else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def predict_network(model, X):
    predictions = []

    model.eval()

    with torch.inference_mode():
        for start in range(0, len(X), BATCH_SIZE):
            batch = torch.from_numpy(
                X[start:start + BATCH_SIZE]
            )

            logits = model(batch)
            probs = torch.sigmoid(logits)

            predictions.append(
                (probs >= 0.5)
                .to(torch.int64)
                .numpy()
            )

    return np.concatenate(predictions)


def load_network(name, seed):
    path = MODEL_DIR / (
        f"{name}_seed{seed}_best.pt"
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    if checkpoint.get("model_name") != name:
        raise ValueError(
            f"Modelo incorreto em {path}"
        )

    model = MODELS[name](
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model


def main():
    torch.set_num_threads(1)

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    metadata = pd.read_csv(
        DATA_DIR / "test_metadata.csv"
    )

    if len(metadata) != len(y_test):
        raise ValueError(
            "Metadata e sequências não coincidem."
        )

    if not np.array_equal(
        metadata["label"].to_numpy(),
        y_test
    ):
        raise ValueError(
            "Os rótulos da metadata não coincidem."
        )

    print("=== TESTE FINAL V2-B ===")
    print("Treino:", X_train.shape)
    print("Teste:", X_test.shape)
    print("Normal:", int((y_test == 0).sum()))
    print("Ataque:", int((y_test == 1).sum()))

    results = []
    attack_results = []

    def register(name, seed, predictions):
        result = {
            "model": name,
            "seed": seed,
            **metrics(y_test, predictions)
        }
        results.append(result)

        print(
            f"{name} | seed={seed} | "
            f"F1={result['f1']:.4f} | "
            f"Recall={result['recall']:.4f} | "
            f"FPR={result['fpr']:.4f}"
        )

        print(
            f"TN={result['tn']} "
            f"FP={result['fp']} "
            f"FN={result['fn']} "
            f"TP={result['tp']}"
        )

        for attack in ATTACK_TYPES:
            mask = (
                metadata[attack].to_numpy() > 0
            )

            n = int(mask.sum())

            if n == 0:
                continue

            detected = int(
                (predictions[mask] == 1).sum()
            )

            attack_results.append({
                "model": name,
                "seed": seed,
                "attack": attack,
                "windows": n,
                "detected": detected,
                "missed": n - detected,
                "detection_rate": detected / n
            })

    # Baseline sem aprendizagem
    register(
        "Always Normal",
        None,
        np.zeros_like(y_test)
    )

    # Baselines treinadas somente no treino
    variants = {
        "LogReg Last Minute": (
            X_train[:, -1, :],
            X_test[:, -1, :]
        ),
        "LogReg Full Window": (
            X_train.reshape(len(X_train), -1),
            X_test.reshape(len(X_test), -1)
        ),
        "LogReg Connections": (
            X_train[:, -1, 0:1],
            X_test[:, -1, 0:1]
        ),
    }

    for name, (train_X, test_X) in variants.items():
        baseline = LogisticRegression(
            max_iter=2000,
            random_state=42
        )

        baseline.fit(train_X, y_train)

        register(
            name,
            None,
            baseline.predict(test_X)
        )

    # Nove checkpoints já selecionados na validação
    for name in MODELS:
        for seed in SEEDS:
            model = load_network(name, seed)

            predictions = predict_network(
                model, X_test
            )

            register(
                name.upper(),
                seed,
                predictions
            )

    results_df = pd.DataFrame(results)
    attacks_df = pd.DataFrame(attack_results)

    results_df.to_csv(
        OUTPUT_DIR / "v2b_test_results.csv",
        index=False
    )

    attacks_df.to_csv(
        OUTPUT_DIR / "v2b_test_by_attack.csv",
        index=False
    )

    neural = results_df[
        results_df["seed"].notna()
    ]

    summary = (
        neural.groupby("model")
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            recall_mean=("recall", "mean"),
            fpr_mean=("fpr", "mean"),
            balanced_accuracy_mean=(
                "balanced_accuracy", "mean"
            )
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "v2b_test_summary.csv",
        index=False
    )

    print("\n=== MÉDIA DAS TRÊS SEEDS ===")
    print(summary.to_string(index=False))

    print("\n=== DETEÇÃO POR TIPO DE ATAQUE ===")
    print(
        attacks_df[
            ["model", "seed", "attack",
             "windows", "detected",
             "detection_rate"]
        ].to_string(index=False)
    )

    print("\nFicheiros guardados em:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
