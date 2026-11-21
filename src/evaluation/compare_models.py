
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)

from src.models.lstm import LSTMClassifier
from src.models.gru import GRUClassifier
from src.models.transformer import TransformerClassifier


ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT / "data" / "processed"
    / "temporal_scanning" / "model_data" / "test.npz"
)

MODEL_DIR = ROOT / "results" / "models"
OUTPUT_DIR = ROOT / "results" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
BATCH_SIZE = 32
THRESHOLD = 0.5

MODELS = {
    "LSTM": LSTMClassifier,
    "GRU": GRUClassifier,
    "Transformer": TransformerClassifier
}


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(
            y_true, y_pred, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, y_pred, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, y_pred, zero_division=0
        )),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "fpr": float(
            fp / (fp + tn) if (fp + tn) else 0
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def load_model(name, model_class):
    path = MODEL_DIR / f"{name.lower()}_best.pt"

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True
    )

    model = model_class(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    return model


def predict(model, X):
    outputs = []

    with torch.inference_mode():
        for start in range(0, len(X), BATCH_SIZE):
            batch = torch.from_numpy(
                X[start:start + BATCH_SIZE]
            ).to(DEVICE)

            logits = model(batch)
            probs = torch.sigmoid(logits)

            outputs.append(
                (probs >= THRESHOLD)
                .cpu()
                .numpy()
                .astype(np.int64)
            )

    return np.concatenate(outputs)


def main():
    torch.set_num_threads(1)

    with np.load(DATA_FILE) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)

    if X.shape[1:] != (5, 6):
        raise ValueError(f"Shape inválido: {X.shape}")

    print("\nTESTE TEMPORAL — 24/04/2019")
    print(f"Sequências: {len(y)}")
    print(f"Normal: {(y == 0).sum()}")
    print(f"Scanning: {(y == 1).sum()}")

    results = []

    baseline_predictions = np.zeros_like(y)

    results.append({
        "model": "Always Normal",
        "parameters": 0,
        "inference_ms_per_window": None,
        **calculate_metrics(y, baseline_predictions)
    })

    for name, model_class in MODELS.items():
        print(f"\nA avaliar {name}...")

        model = load_model(name, model_class)

        # Warm-up
        predict(model, X[:32])

        start_time = time.perf_counter()
        predictions = predict(model, X)
        elapsed = time.perf_counter() - start_time

        metrics = calculate_metrics(y, predictions)

        parameters = sum(
            p.numel() for p in model.parameters()
        )

        latency = elapsed * 1000 / len(X)

        results.append({
            "model": name,
            "parameters": int(parameters),
            "inference_ms_per_window": float(latency),
            **metrics
        })

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"FPR: {metrics['fpr']:.4f}")
        print(
            f"TN={metrics['tn']} FP={metrics['fp']} "
            f"FN={metrics['fn']} TP={metrics['tp']}"
        )
        print(f"Inferência: {latency:.4f} ms/janela")

    df = pd.DataFrame(results)

    csv_path = OUTPUT_DIR / "model_test_comparison.csv"
    json_path = OUTPUT_DIR / "model_test_comparison.json"

    df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nCOMPARAÇÃO FINAL DO PROTÓTIPO")
    print(
        df[[
            "model", "accuracy", "precision",
            "recall", "f1", "fpr", "parameters"
        ]].to_string(index=False)
    )

    print(f"\nResultados guardados em: {csv_path}")


if __name__ == "__main__":
    main()
