
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.gru import GRUClassifier
from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerClassifier
from src.federated.train_fedavg_gru import evaluate


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "processed"
    / "ton_iot_v2" / "temporal_generalization"
)

CENTRALIZED_DIR = ROOT / "results" / "models" / "v2b"
FEDERATED_DIR = ROOT / "results" / "federated"
OUTPUT_DIR = ROOT / "results" / "tables"

MODELS = ["gru", "lstm", "transformer"]
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


def build_model(model_name):
    if model_name == "gru":
        return GRUClassifier(
            input_size=6,
            hidden_size=32
        )

    if model_name == "lstm":
        return LSTMClassifier(
            input_size=6,
            hidden_size=32,
            num_layers=1,
            dropout=0.2
        )
    if model_name == "transformer":
        return TransformerClassifier(
            input_size=6,
            hidden_size=32,
            num_layers=1,
            dropout=0.2,
            nhead=4,
            max_seq_len=5
        )

    raise ValueError(
        f"Modelo não suportado: {model_name}"
    )


def evaluate_checkpoint(path, model_name, X_val, y_val):
    if not path.is_file():
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

    model = build_model(model_name)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    with torch.no_grad():
        return evaluate(model, X_val, y_val)


def main():
    torch.set_num_threads(1)

    X_val, y_val = load_validation()

    print("=== COMPARAÇÃO CENTRALIZADA VS FEDERADA ===")
    print("Modelos:", MODELS)
    print("Seeds:", SEEDS)
    print("Janelas de validação:", len(y_val))
    print("Conjunto de teste: NÃO UTILIZADO")
    print()

    results = []

    for model_name in MODELS:
        for seed in SEEDS:
            checkpoints = {
                "centralized": (
                    CENTRALIZED_DIR
                    / f"{model_name}_seed{seed}_best.pt"
                ),
                "federated": (
                    FEDERATED_DIR
                    / f"flower_v2b_{model_name}_seed{seed}"
                    / f"{model_name}_best.pt"
                ),
            }

            for method, path in checkpoints.items():
                metrics = evaluate_checkpoint(
                    path,
                    model_name,
                    X_val,
                    y_val
                )

                results.append({
                    "model": model_name,
                    "method": method,
                    "seed": seed,
                    "validation_windows": len(y_val),
                    **metrics,
                })

                print(
                    f"{model_name.upper():4s} | "
                    f"Seed {seed:4d} | "
                    f"{method:12s} | "
                    f"Loss={metrics['loss']:.6f} | "
                    f"F1={metrics['f1']:.6f} | "
                    f"Recall={metrics['recall']:.6f} | "
                    f"FPR={metrics['fpr']:.6f}"
                )

    df = pd.DataFrame(results)

    summary = (
        df.groupby(["model", "method"])[METRICS]
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
        / "v2b_centralized_vs_federated_models.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "v2b_centralized_vs_federated_models_summary.csv"
    )

    df.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n=== MÉDIA ± DESVIO-PADRÃO ===")

    for _, row in summary.iterrows():
        print(
            f"\n{row['model'].upper()} — "
            f"{row['method']}"
        )

        for metric in ["loss", "f1", "recall", "fpr"]:
            print(
                f"{metric:7s}: "
                f"{row[f'{metric}_mean']:.6f} ± "
                f"{row[f'{metric}_std']:.6f}"
            )

    print("\n=== FICHEIROS GUARDADOS ===")
    print(detail_path)
    print(summary_path)
    print("\nCOMPARAÇÃO CONCLUÍDA")


if __name__ == "__main__":
    main()
