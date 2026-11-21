
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    MetricRecord,
)
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from src.models.gru import GRUClassifier
from src.federated.train_fedavg_gru import evaluate


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "processed"
    / "ton_iot_v2" / "temporal_generalization"
)

OUTPUT_DIR = (
    ROOT / "results" / "federated"
    / "flower_v2b_smoke"
)

NUM_CLIENTS = 5
NUM_ROUNDS = 2
LEARNING_RATE = 0.001
SEED = 42

app = ServerApp()


def build_model():
    return GRUClassifier(
        input_size=6,
        hidden_size=32
    )


def load_validation_data():
    with np.load(DATA_DIR / "validation.npz") as data:
        X = torch.from_numpy(
            data["X"].astype(np.float32)
        )
        y = torch.from_numpy(
            data["y"].astype(np.float32)
        )

    return X, y


@app.main()
def main(grid: Grid, context: Context):
    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X_val, y_val = load_validation_data()

    model = build_model()

    initial_arrays = ArrayRecord(
        model.state_dict()
    )

    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=NUM_CLIENTS,
        min_available_nodes=NUM_CLIENTS,
        weighted_by_key="num-examples"
    )

    history = []

    best_loss = float("inf")
    best_round = None

    def evaluate_global(
        server_round: int,
        arrays: ArrayRecord
    ) -> MetricRecord:
        nonlocal best_loss, best_round

        model = build_model()

        model.load_state_dict(
            arrays.to_torch_state_dict()
        )

        metrics = evaluate(
            model,
            X_val,
            y_val
        )

        record = {
            "round": int(server_round),
            **metrics
        }

        history.append(record)

        print(
            f"[SERVIDOR] Ronda {server_round:02d} | "
            f"Val Loss={metrics['loss']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"FPR={metrics['fpr']:.4f}",
            flush=True
        )

        # Seleção pela menor validation loss,
        # mantendo a regra do FedAvg manual.
        if (
            server_round > 0
            and metrics["loss"] < best_loss
        ):
            best_loss = metrics["loss"]
            best_round = server_round

            torch.save({
                "model_state_dict":
                    arrays.to_torch_state_dict(),
                "model_name": "gru",
                "experiment": "flower_v2b_smoke",
                "seed": SEED,
                "input_size": 6,
                "hidden_size": 32,
                "best_round": best_round,
                "num_clients": NUM_CLIENTS
            }, OUTPUT_DIR / "gru_best.pt")

        return MetricRecord({
            "val_loss": float(metrics["loss"]),
            "val_f1": float(metrics["f1"]),
            "val_recall": float(metrics["recall"]),
            "val_fpr": float(metrics["fpr"]),
            "num-examples": len(y_val)
        })

    print("=== FLOWER — TESTE DE INTEGRAÇÃO ===")
    print("Clientes:", NUM_CLIENTS)
    print("Rondas:", NUM_ROUNDS)
    print("Validação:", len(y_val))
    print("Agregação: FedAvg")
    print("Seed:", SEED)

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=NUM_ROUNDS,
        train_config=ConfigRecord({
            "lr": LEARNING_RATE
        }),
        evaluate_fn=evaluate_global
    )

    if result.arrays is None:
        raise RuntimeError(
            "O Flower não devolveu o modelo global."
        )

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "validation_history.csv",
        index=False
    )

    report = {
        "experiment": "flower_v2b_smoke",
        "num_clients": NUM_CLIENTS,
        "rounds": NUM_ROUNDS,
        "seed": SEED,
        "best_round": best_round,
        "best_validation_loss": best_loss,
        "validation_windows": len(y_val),
        "evaluation": "centralized_validation",
        "test_used": False
    }

    with open(
        OUTPUT_DIR / "report.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print("\n=== FLOWER TERMINADO ===")
    print("Melhor ronda:", best_round)
    print("Menor validation loss:", best_loss)
    print("Resultados:", OUTPUT_DIR)
