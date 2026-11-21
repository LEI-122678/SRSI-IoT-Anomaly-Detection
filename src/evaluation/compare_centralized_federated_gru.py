
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.gru import GRUClassifier
from src.federated.train_fedavg_gru import evaluate


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "processed"
    / "ton_iot_v2" / "temporal_generalization"
)

CENTRALIZED_DIR = (
    ROOT / "results" / "models" / "v2b"
)

FEDERATED_DIR = (
    ROOT / "results" / "federated"
)

OUTPUT_DIR = ROOT / "results" / "tables"

SEEDS = [42, 123, 2026]

METRICS = [
    "loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "fpr",
]


def load_validation():
    with np.load(DATA_DIR / "validation.npz") as data:
        X = torch.from_numpy(
            data["X"].astype(np.float32)
        )

        y = torch.from_numpy(
            data["y"].astype(np.float32)
        )

    return X, y


def build_model():
    return GRUClassifier(
        input_size=6,
        hidden_size=32
    )


def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint sem model_state_dict: {path}"
        )

    return checkpoint


def evaluate_checkpoint(path, X_val, y_val):
    checkpoint = load_checkpoint(path)

    model = build_model()

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    with torch.no_grad():
        metrics = evaluate(
            model,
            X_val,
            y_val
        )

    return metrics


def main():
    torch.set_num_threads(1)

    X_val, y_val = load_validation()

    print(
        "=== COMPARAÇÃO GRU CENTRALIZADA VS FEDERADA ==="
    )
    print("Validação:", len(y_val), "janelas")
    print("Seeds:", SEEDS)
    print("Conjunto de teste: NÃO UTILIZADO")
    print()

    results = []

    for seed in SEEDS:
        centralized_path = (
            CENTRALIZED_DIR
            / f"gru_seed{seed}_best.pt"
        )

        federated_path = (
            FEDERATED_DIR
            / f"flower_v2b_gru_seed{seed}"
            / "gru_best.pt"
        )

        for method, path in [
            ("centralized", centralized_path),
            ("federated", federated_path),
        ]:
            metrics = evaluate_checkpoint(
                path,
                X_val,
                y_val
            )

            record = {
                "method": method,
                "model": "gru",
                "seed": seed,
                "validation_windows": len(y_val),
                **metrics
            }

            results.append(record)

            print(
                f"Seed {seed} | "
                f"{method:12s} | "
                f"Loss={metrics['loss']:.6f} | "
                f"F1={metrics['f1']:.6f} | "
                f"Recall={metrics['recall']:.6f} | "
                f"FPR={metrics['fpr']:.6f}"
            )

    df = pd.DataFrame(results)

    summary = (
        df.groupby("method")[METRICS]
        .agg(["mean", "std"])
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary = summary.reset_index()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    detail_path = (
        OUTPUT_DIR
        / "v2b_gru_centralized_vs_federated.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "v2b_gru_centralized_vs_federated_summary.csv"
    )

    df.to_csv(
        detail_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\n=== MÉDIA ± DESVIO-PADRÃO ===")

    for _, row in summary.iterrows():
        print(f"\nMétodo: {row['method']}")

        for metric in ["loss", "f1", "recall", "fpr"]:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]

            print(
                f"{metric:7s}: "
                f"{mean:.6f} ± {std:.6f}"
            )

    print("\n=== FICHEIROS GUARDADOS ===")
    print(detail_path)
    print(summary_path)

    print("\nCOMPARAÇÃO CONCLUÍDA")


if __name__ == "__main__":
    main()
