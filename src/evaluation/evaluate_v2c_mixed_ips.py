
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from src.models.gru import GRUClassifier
from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerClassifier
from src.federated.train_fedavg_gru import evaluate


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "processed" / "ton_iot_v2"
    / "ip_generalization" / "internal_validation"
)

MODEL_DIR = ROOT / "results" / "models" / "v2c"
OUTPUT_DIR = ROOT / "results" / "tables"

MODELS = ["gru", "lstm", "transformer"]
SEEDS = [42, 123, 2026]

MIXED_IPS = {
    "192.168.1.79",
    "192.168.1.180",
}


def build_model(name):
    if name == "gru":
        return GRUClassifier(
            input_size=6,
            hidden_size=32
        )

    if name == "lstm":
        return LSTMClassifier(
            input_size=6,
            hidden_size=32,
            num_layers=1,
            dropout=0.2
        )

    if name == "transformer":
        return TransformerClassifier(
            input_size=6,
            hidden_size=32,
            num_layers=1,
            dropout=0.2,
            nhead=4,
            max_seq_len=5
        )

    raise ValueError(f"Modelo desconhecido: {name}")


def main():
    torch.set_num_threads(1)

    with np.load(
        DATA_DIR / "internal_validation.npz"
    ) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)

    metadata = pd.read_csv(
        DATA_DIR / "internal_validation_metadata.csv",
        dtype={"src_ip": "string"},
    )

    if len(metadata) != len(y):
        raise ValueError(
            "Dimensoes diferentes entre dados e metadata."
        )

    if not np.array_equal(
        metadata["label"].to_numpy(),
        y.astype(np.int64)
    ):
        raise ValueError("Rotulos inconsistentes.")

    mask = metadata["src_ip"].isin(MIXED_IPS).to_numpy()

    X_mixed = torch.from_numpy(X[mask])
    y_mixed = torch.from_numpy(y[mask])

    if len(y_mixed) == 0:
        raise ValueError("Sem janelas de IPs mistos.")

    if set(np.unique(y[mask])) != {0, 1}:
        raise ValueError(
            "Os IPs mistos nao contem ambas as classes."
        )

    print("=== AVALIACAO V2-C: IPS MISTOS ===")
    print("Janelas:", len(y_mixed))
    print("Normais:", int((y_mixed == 0).sum()))
    print("Ataques:", int((y_mixed == 1).sum()))

    results = []

    for name in MODELS:
        for seed in SEEDS:
            path = (
                MODEL_DIR
                / f"{name}_seed{seed}_best.pt"
            )

            if not path.exists():
                raise FileNotFoundError(path)

            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True
            )

            if checkpoint["model_name"] != name:
                raise ValueError(
                    f"Modelo incorreto no checkpoint: {path}"
                )

            model = build_model(name)
            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            metrics = evaluate(
                model,
                X_mixed,
                y_mixed
            )

            results.append({
                "model": name,
                "seed": seed,
                "best_epoch": checkpoint["best_epoch"],
                "subset": "mixed_ips",
                "windows": len(y_mixed),
                "normal": int((y_mixed == 0).sum()),
                "attacks": int((y_mixed == 1).sum()),
                **metrics,
            })

            print(
                f"{name.upper():12s} "
                f"seed={seed:4d} "
                f"F1={metrics['f1']:.4f} "
                f"Recall={metrics['recall']:.4f} "
                f"FPR={metrics['fpr']:.4f} "
                f"FN={metrics['fn']}"
            )

    df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        OUTPUT_DIR / "v2c_centralized_mixed_ips.csv"
    )

    df.to_csv(output, index=False)

    print("\n=== MEDIAS POR ARQUITETURA ===")
    print(
        df.groupby("model")[
            ["f1", "recall", "fpr"]
        ].agg(["mean", "std"]).round(4)
    )

    print("\nResultados:", output)


if __name__ == "__main__":
    main()
