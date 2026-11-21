
from pathlib import Path
import copy
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.models.gru import GRUClassifier


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "processed"
    / "ton_iot_v2" / "temporal_generalization"
)

PARTITION_DIR = (
    ROOT / "data" / "processed"
    / "ton_iot_v2" / "federated"
)

OUTPUT_DIR = ROOT / "results" / "federated" / "v2b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLIENTS = 5
ROUNDS = 20
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SEED = 42
PATIENCE = 8


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def build_model():
    return GRUClassifier(
        input_size=6,
        hidden_size=32
    )


def load_data(name):
    with np.load(DATA_DIR / f"{name}.npz") as data:
        X = torch.from_numpy(
            data["X"].astype(np.float32)
        )
        y = torch.from_numpy(
            data["y"].astype(np.float32)
        )

    return X, y


def train_client(global_state, X, y, client_id, round_id):
    model = build_model()
    model.load_state_dict(global_state)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.BCEWithLogitsLoss()

    generator = torch.Generator().manual_seed(
        SEED + round_id * 100 + client_id
    )

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    total_loss = 0.0
    total_samples = 0

    for _ in range(LOCAL_EPOCHS):
        for batch_X, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            n = len(batch_y)
            total_loss += loss.item() * n
            total_samples += n

    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    return (
        state,
        len(y),
        total_loss / total_samples
    )


def fedavg(client_updates):
    total_samples = sum(
        count for _, count, _ in client_updates
    )

    first_state = client_updates[0][0]
    aggregated = {}

    for key in first_state:
        reference = first_state[key]

        if not torch.is_floating_point(reference):
            raise TypeError(
                f"Estado não flutuante inesperado: {key}"
            )

        value = torch.zeros_like(
            reference, dtype=torch.float64
        )

        for state, count, _ in client_updates:
            weight = count / total_samples

            value += (
                state[key].to(torch.float64)
                * weight
            )

        aggregated[key] = value.to(
            dtype=reference.dtype
        )

    return aggregated


def evaluate(model, X, y):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    logits_list = []

    with torch.inference_mode():
        for batch in torch.split(X, 128):
            logits_list.append(model(batch))

        logits = torch.cat(logits_list)
        loss = criterion(logits, y).item()

        predictions = (
            torch.sigmoid(logits) >= 0.5
        ).int().numpy()

    actual = y.int().numpy()

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "loss": float(loss),
        "accuracy": float(
            accuracy_score(actual, predictions)
        ),
        "precision": float(
            precision_score(
                actual, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                actual, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                actual, predictions, zero_division=0
            )
        ),
        "fpr": float(
            fp / (tn + fp) if tn + fp else 0
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def main():
    set_seed(SEED)

    X_train, y_train = load_data("train")
    X_val, y_val = load_data("validation")

    if tuple(X_train.shape[1:]) != (5, 6):
        raise ValueError(
            f"Dimensões inesperadas: {X_train.shape}"
        )

    client_data = []
    all_indices = []

    for client_id in range(NUM_CLIENTS):
        path = PARTITION_DIR / (
            f"client_{client_id}_indices.npy"
        )

        indices = np.load(path).astype(np.int64)

        if not len(indices):
            raise ValueError(
                f"Cliente {client_id} vazio"
            )

        if (
            indices.min() < 0
            or indices.max() >= len(y_train)
        ):
            raise ValueError(
                "Índice fora dos limites"
            )

        all_indices.append(indices)

        client_data.append((
            X_train[indices],
            y_train[indices]
        ))

    combined = np.concatenate(all_indices)

    if not np.array_equal(
        np.sort(combined),
        np.arange(len(y_train))
    ):
        raise ValueError(
            "Partições duplicadas ou incompletas"
        )

    global_model = build_model()
    global_state = copy.deepcopy(
        global_model.state_dict()
    )

    best_loss = float("inf")
    best_round = 0
    no_improvement = 0
    history = []

    checkpoint_path = (
        OUTPUT_DIR / "gru_fedavg_seed42_best.pt"
    )

    print("=== GRU FEDERADA — FEDAVG ===")
    print("Clientes:", NUM_CLIENTS)
    print("Janelas de treino:", len(y_train))
    print("Janelas de validação:", len(y_val))
    print("Rondas máximas:", ROUNDS)
    print("Épocas locais:", LOCAL_EPOCHS)

    start = time.perf_counter()

    for round_id in range(1, ROUNDS + 1):
        updates = []

        for client_id, (X_local, y_local) in enumerate(
            client_data
        ):
            update = train_client(
                global_state,
                X_local,
                y_local,
                client_id,
                round_id
            )

            updates.append(update)

        global_state = fedavg(updates)

        global_model.load_state_dict(
            global_state
        )

        val_metrics = evaluate(
            global_model, X_val, y_val
        )

        train_loss = sum(
            loss * count
            for _, count, loss in updates
        ) / len(y_train)

        record = {
            "round": round_id,
            "local_train_loss": float(train_loss),
            **{
                f"validation_{key}": value
                for key, value in val_metrics.items()
            }
        }

        history.append(record)

        print(
            f"Ronda {round_id:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f} | "
            f"FPR: {val_metrics['fpr']:.4f}"
        )

        if val_metrics["loss"] < best_loss - 1e-4:
            best_loss = val_metrics["loss"]
            best_round = round_id
            no_improvement = 0

            torch.save({
                "model_state_dict": global_state,
                "model_name": "gru",
                "experiment": "v2b_fedavg",
                "seed": SEED,
                "input_size": 6,
                "hidden_size": 32,
                "best_round": best_round,
                "num_clients": NUM_CLIENTS,
                "local_epochs": LOCAL_EPOCHS
            }, checkpoint_path)

        else:
            no_improvement += 1

        if no_improvement >= PATIENCE:
            print("Early stopping.")
            break

    duration = time.perf_counter() - start

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    global_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    best_metrics = evaluate(
        global_model, X_val, y_val
    )

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "gru_fedavg_seed42_history.csv",
        index=False
    )

    report = {
        "experiment": "v2b_fedavg",
        "model": "gru",
        "seed": SEED,
        "num_clients": NUM_CLIENTS,
        "rounds_completed": len(history),
        "best_round": best_round,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "selection_metric": "validation_loss",
        "training_seconds": duration,
        "validation": best_metrics,
        "simulation": (
            "Single-process FedAvg simulation; "
            "not a deployed distributed system"
        )
    }

    with open(
        OUTPUT_DIR / "gru_fedavg_seed42_report.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print("\n=== RESULTADO FINAL FEDAVG ===")
    print("Melhor ronda:", best_round)
    print("Rondas executadas:", len(history))
    print("Tempo:", round(duration, 2), "s")
    print("F1:", round(best_metrics["f1"], 4))
    print("Recall:", round(best_metrics["recall"], 4))
    print("FPR:", round(best_metrics["fpr"], 4))
    print(
        f"TN={best_metrics['tn']} "
        f"FP={best_metrics['fp']} "
        f"FN={best_metrics['fn']} "
        f"TP={best_metrics['tp']}"
    )
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
