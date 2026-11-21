from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from src.models.gru import GRUClassifier
from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerClassifier


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "ton_iot_v2" / "ip_generalization"
RESULTS = ROOT / "results"
PROTOCOL_PATH = RESULTS / "v2c_final_protocol.json"

EVALUATION_IPS = {
    "192.168.1.30",
    "192.168.1.193",
    "192.168.1.195",
}


def build_model(name):
    if name == "gru":
        return GRUClassifier(input_size=6, hidden_size=32)
    if name == "lstm":
        return LSTMClassifier(
            input_size=6, hidden_size=32,
            num_layers=1, dropout=0.2
        )
    if name == "transformer":
        return TransformerClassifier(
            input_size=6, hidden_size=32,
            num_layers=1, dropout=0.2,
            nhead=4, max_seq_len=5
        )
    raise ValueError(name)


def metrics_for(y, pred):
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "windows": len(y),
        "normal": tn + fp,
        "attacks": tp + fn,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tp + tn) / len(y),
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * tp / (2 * tp + fp + fn)
            if 2 * tp + fp + fn else 0.0
        ),
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
    }


def main():
    torch.set_num_threads(1)

    protocol = json.loads(
        PROTOCOL_PATH.read_text(encoding="utf-8")
    )

    assert protocol["status"] == "frozen_before_final_evaluation"
    assert protocol["classification_threshold"] == 0.5

    # Confirmar separacao por IP antes de ler o conjunto reservado.
    train_meta = pd.read_csv(
        DATA_DIR / "train_metadata.csv",
        dtype={"src_ip": str}
    )
    eval_meta = pd.read_csv(
        DATA_DIR / "evaluation_metadata.csv",
        dtype={"src_ip": str}
    )

    train_ips = set(train_meta["src_ip"])
    eval_ips = set(eval_meta["src_ip"])

    assert len(train_ips) == 37
    assert eval_ips == EVALUATION_IPS
    assert train_ips.isdisjoint(eval_ips)

    # Todos os checkpoints devem existir antes da avaliacao.
    paths = []

    for name in protocol["models"]:
        for seed in protocol["seeds"]:
            paths.extend([
                (
                    "centralized", name, seed,
                    RESULTS / "models" / "v2c_final"
                    / f"{name}_seed{seed}_final.pt"
                ),
                (
                    "federated", name, seed,
                    RESULTS / "federated"
                    / f"flower_v2c_final_{name}_seed{seed}"
                    / f"{name}_final.pt"
                )
            ])

    assert len(paths) == 18

    for _, _, _, path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = RESULTS / "tables"

    outputs = [
        output_dir / "v2c_final_evaluation_models.csv",
        output_dir / "v2c_final_evaluation_summary.csv",
        output_dir / "v2c_final_evaluation_per_ip.csv",
    ]

    if any(path.exists() for path in outputs):
        raise FileExistsError(
            "Ja existem ficheiros da avaliacao final. "
            "Nao serao substituidos."
        )

    with np.load(DATA_DIR / "evaluation.npz") as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64).reshape(-1)

    assert X.shape == (4256, 5, 6)
    assert len(eval_meta) == len(y)
    assert np.array_equal(
        eval_meta["label"].to_numpy(),
        y
    )
    assert np.isfinite(X).all()
    assert np.isin(y, [0, 1]).all()

    X_tensor = torch.from_numpy(X)

    global_rows = []
    ip_rows = []

    for approach, name, seed, path in paths:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=True
        )

        assert checkpoint["model_name"] == name
        assert checkpoint["seed"] == seed
        assert checkpoint["input_size"] == 6

        if approach == "centralized":
            assert checkpoint["fixed_epochs"] == (
                protocol["centralized_epochs"][name]
            )
        else:
            assert checkpoint["fixed_rounds"] == (
                protocol["federated_rounds"][name]
            )

        model = build_model(name)
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )
        model.eval()

        predictions = []

        with torch.inference_mode():
            for batch in X_tensor.split(128):
                logits = model(batch).reshape(-1)
                predictions.append(
                    (
                        torch.sigmoid(logits)
                        >= protocol["classification_threshold"]
                    ).to(torch.int64).numpy()
                )

        pred = np.concatenate(predictions)
        common = {
            "approach": approach,
            "model": name,
            "seed": seed,
        }

        global_rows.append({
            **common,
            **metrics_for(y, pred)
        })

        for ip in sorted(EVALUATION_IPS):
            mask = eval_meta["src_ip"].eq(ip).to_numpy()
            ip_rows.append({
                **common,
                "src_ip": ip,
                **metrics_for(y[mask], pred[mask])
            })

    global_df = pd.DataFrame(global_rows)
    ip_df = pd.DataFrame(ip_rows)

    summary = (
        global_df.groupby(["model", "approach"])[
            ["accuracy", "precision", "recall", "f1", "fpr"]
        ]
        .agg(["mean", "std"])
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]
    summary = summary.reset_index()

    output_dir.mkdir(parents=True, exist_ok=True)

    global_df.to_csv(outputs[0], index=False)
    summary.to_csv(outputs[1], index=False)
    ip_df.to_csv(outputs[2], index=False)

    print("=== AVALIACAO FINAL V2-C ===")
    print("Modelos avaliados:", len(global_df))
    print("Janelas reservadas:", len(y))
    print("IPs reservados:", len(eval_ips))

    print("\n=== RESULTADOS MEDIOS ===")
    print(
        summary[
            ["model", "approach",
             "f1_mean", "f1_std",
             "recall_mean", "fpr_mean"]
        ].round(4).to_string(index=False)
    )

    print("\nFicheiros:")
    for path in outputs:
        print(path)

    print("\nAvaliacao concluida. Nao reajustar modelos.")


if __name__ == "__main__":
    main()
