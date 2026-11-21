
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "processed"
    / "ton_iot_v2" / "temporal_generalization"
)

OUTPUT_DIR = ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(name):
    with np.load(DATA_DIR / f"{name}.npz") as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
    return X, y


def evaluate(name, y_true, predictions):
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "model": name,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def train_baseline(name, X_train, y_train, X_val, y_val):
    model = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    return evaluate(name, y_val, predictions)


def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")

    print("=== BASELINES V2-B ===")
    print("Treino:", X_train.shape)
    print("Validação:", X_val.shape)

    results = []

    # 1. Baseline sem aprendizagem
    results.append(evaluate(
        "Always Normal",
        y_val,
        np.zeros_like(y_val)
    ))

    # 2. Apenas o último minuto:
    # seis características, sem sequência.
    results.append(train_baseline(
        "LogReg - Last Minute",
        X_train[:, -1, :],
        y_train,
        X_val[:, -1, :],
        y_val
    ))

    # 3. Cinco minutos completos:
    # 30 características numa representação plana.
    results.append(train_baseline(
        "LogReg - Full Window",
        X_train.reshape(len(X_train), -1),
        y_train,
        X_val.reshape(len(X_val), -1),
        y_val
    ))

    # 4. Apenas o número de ligações
    # no último minuto.
    results.append(train_baseline(
        "LogReg - Connections Only",
        X_train[:, -1, 0:1],
        y_train,
        X_val[:, -1, 0:1],
        y_val
    ))

    df = pd.DataFrame(results)

    print("\n=== RESULTADOS NA VALIDAÇÃO ===")

    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    output = (
        OUTPUT_DIR / "v2b_baselines_validation.csv"
    )

    df.to_csv(output, index=False)

    print("\nResultados guardados em:")
    print(output)


if __name__ == "__main__":
    main()
