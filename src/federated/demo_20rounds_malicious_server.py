
from pathlib import Path
import json

import numpy as np
import os
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
from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerClassifier

from src.federated.train_fedavg_gru import evaluate


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "processed"
    / "ton_iot_v2" / "ip_generalization" / "internal_validation"
)

# Configuração exclusiva da demonstração
NUM_CLIENTS = 5
NUM_ROUNDS = 20
LEARNING_RATE = 0.001
SEED = 42
MODEL_NAME = "gru"


# Pasta exclusiva da experiencia com cliente malicioso.
# Nunca substituir os resultados da demonstracao benigna.
# Resultados exclusivos do ataque com fator -10

# Nova execucao para verificar o checkpoint da ronda 20
OUTPUT_DIR = (
    ROOT / "results" / "demonstration"
    / "flower_20rounds_malicious_neg10_verified"
)




# Early stopping
PATIENCE = 8
MIN_DELTA = 1e-4

EXPERIMENT_NAME = "flower_demo_20rounds_malicious_neg10"

app = ServerApp()


def build_model():
    if MODEL_NAME == "gru":
        return GRUClassifier(
            input_size=6,
            hidden_size=32
        )

    if MODEL_NAME == "lstm":
        return LSTMClassifier(
            input_size=6,
            hidden_size=32,
            num_layers=1,
            dropout=0.2
        )

    if MODEL_NAME == "transformer":
        return TransformerClassifier(
            input_size=6,
            hidden_size=32,
            num_layers=1,
            dropout=0.2,
            nhead=4,
            max_seq_len=5
        )

    raise ValueError(
        f"Arquitetura não suportada: {MODEL_NAME}"
    )


def load_validation_data():
    with np.load(DATA_DIR / "internal_validation.npz") as data:
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
    np.random.seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X_val, y_val = load_validation_data()

    # Modelo global inicial
    model = build_model()

    initial_arrays = ArrayRecord(
        model.state_dict()
    )

    # Estratégia FedAvg
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
    best_state = None

    stopped_early = False

    def evaluate_global(
        server_round: int,
        arrays: ArrayRecord
    ) -> MetricRecord:
        nonlocal best_loss, best_round, best_state

        # Carregar o modelo global da ronda atual
        model = build_model()

        state_dict = arrays.to_torch_state_dict()

        model.load_state_dict(state_dict)

        # Validação centralizada
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
            f"Val Loss={metrics['loss']:.6f} | "
            f"F1={metrics['f1']:.6f} | "
            f"Recall={metrics['recall']:.6f} | "
            f"FPR={metrics['fpr']:.6f}",
            flush=True
        )

        # A ronda 0 corresponde ao modelo inicial.
        # Não é considerada para seleção do checkpoint.
        if server_round > 0:

            # Guardar o melhor checkpoint
            # usando a menor validation loss.
            if (
                best_round is None
                or metrics["loss"] < best_loss - MIN_DELTA
            ):
                best_loss = float(metrics["loss"])
                best_round = int(server_round)

                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in state_dict.items()
                }

                torch.save(
                    {
                        "model_state_dict": best_state,
                        "model_name": MODEL_NAME,
                        "experiment": EXPERIMENT_NAME,
                        "seed": SEED,
                        "input_size": 6,
                        "hidden_size": 32,
                        "best_round": best_round,
                        "best_validation_loss": best_loss,
                        "num_clients": NUM_CLIENTS,
                        "learning_rate": LEARNING_RATE,
                        "local_epochs": 1,
                        "batch_size": 32,
                        "patience": PATIENCE,
                        "min_delta": MIN_DELTA,
                    },
                    OUTPUT_DIR / f"{MODEL_NAME}_best.pt"
                )

                print(
                    f"[CHECKPOINT] Melhor modelo "
                    f"na ronda {best_round} | "
                    f"Val Loss={best_loss:.6f}",
                    flush=True
                )

        return MetricRecord({
            "val_loss": float(metrics["loss"]),
            "val_f1": float(metrics["f1"]),
            "val_recall": float(metrics["recall"]),
            "val_fpr": float(metrics["fpr"]),
            "num-examples": len(y_val)
        })

    def should_stop(server_round, result):
        nonlocal stopped_early

        if best_round is None:
            return False

        rounds_without_improvement = (
            server_round - best_round
        )

        if rounds_without_improvement >= PATIENCE:
            stopped_early = True

            print(
                f"\n[EARLY STOPPING] "
                f"Treino interrompido na ronda "
                f"{server_round}.",
                flush=True
            )

            print(
                f"Sem melhoria significativa "
                f"durante {PATIENCE} rondas.",
                flush=True
            )

            print(
                f"Melhor ronda: {best_round} | "
                f"Validation loss: {best_loss:.6f}",
                flush=True
            )

            return True

        return False

    print(f"\n=== FLOWER — TREINO FEDERADO {MODEL_NAME.upper()} ===")
    print("Experiência:", EXPERIMENT_NAME)
    print("Clientes:", NUM_CLIENTS)
    print("Máximo de rondas:", NUM_ROUNDS)
    print("Validação:", len(y_val))
    print("Agregação: FedAvg")
    print("Seed:", SEED)
    print("Learning rate:", LEARNING_RATE)
    print("Early stopping patience:", PATIENCE)
    print("Early stopping min_delta:", MIN_DELTA)
    print("Dispositivo: CPU")
    print("===================================\n")

    # Executar treino federado
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

    # Guardar o modelo global da ultima ronda
    final_state = result.arrays.to_torch_state_dict()

    torch.save(
        {
            "model_state_dict": final_state,
            "model_name": MODEL_NAME,
            "experiment": EXPERIMENT_NAME,
            "seed": SEED,
            "input_size": 6,
            "hidden_size": 32,
            "last_round": NUM_ROUNDS,
            "num_clients": NUM_CLIENTS,
        },
        OUTPUT_DIR / "gru_last.pt"
    )

    if best_round is None or best_state is None:
        raise RuntimeError(
            "Nenhum checkpoint válido foi guardado."
        )

    # Guardar histórico completo de validação
    history_df = pd.DataFrame(history)

    history_df.to_csv(
        OUTPUT_DIR / "validation_history.csv",
        index=False
    )

    rounds_executed = max(
        int(record["round"])
        for record in history
    )

    # Guardar relatório da experiência
    report = {
        "experiment": EXPERIMENT_NAME,
        "model": MODEL_NAME,
        "algorithm": "FedAvg",
        "seed": SEED,
        "num_clients": NUM_CLIENTS,
        "max_rounds": NUM_ROUNDS,
        "rounds_executed": rounds_executed,
        "stopped_early": stopped_early,
        "best_round": best_round,
        "best_validation_loss": best_loss,
        "learning_rate": LEARNING_RATE,
        "local_epochs": 1,
        "batch_size": 32,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "validation_windows": len(y_val),
        "evaluation": "centralized_validation",
        "checkpoint_selection": "validation_loss",
        "test_used": False,
        "device": "cpu",
        "backend": "ray",
        "framework": "flower"
    }

    with open(
        OUTPUT_DIR / "report.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n=== FLOWER TERMINADO ===")
    print("Rondas executadas:", rounds_executed)
    print("Early stopping:", stopped_early)
    print("Melhor ronda:", best_round)
    print("Menor validation loss:", best_loss)
    print("Resultados:", OUTPUT_DIR)
    print("========================\n")
