
from pathlib import Path
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.models.lstm import LSTMClassifier


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "processed"
    / "temporal_scanning" / "model_data"
)

OUTPUT_DIR = ROOT / "results" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 8
HIDDEN_SIZE = 32


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(name):
    path = DATA_DIR / f"{name}.npz"

    with np.load(path) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)

    if X.ndim != 3 or X.shape[1:] != (5, 6):
        raise ValueError(
            f"Formato inesperado em {name}: {X.shape}"
        )

    return TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y)
    )


def evaluate(model, loader, criterion, device):
    model.eval()

    losses = []
    labels = []
    predictions = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = criterion(logits, y)

            losses.append(
                (loss.item(), len(y))
            )

            probs = torch.sigmoid(logits)
            pred = (probs >= 0.5).long()

            labels.extend(
                y.cpu().numpy().astype(int).tolist()
            )

            predictions.extend(
                pred.cpu().numpy().tolist()
            )

    total = sum(n for _, n in losses)

    avg_loss = (
        sum(loss * n for loss, n in losses) / total
    )

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if fp + tn else 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, zero_division=0
        ),
        "f1": f1_score(
            labels, predictions, zero_division=0
        ),
        "fpr": fpr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Dispositivo:", device)

    train_data = load_data("train")
    val_data = load_data("validation")

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED)
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = LSTMClassifier(
        input_size=6,
        hidden_size=HIDDEN_SIZE
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print(
        "Parâmetros treináveis:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        seen = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(X)
            loss = criterion(logits, y)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            running_loss += loss.item() * len(y)
            seen += len(y)

        train_loss = running_loss / seen

        metrics = evaluate(
            model, val_loader, criterion, device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": metrics
        })

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {metrics['loss']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"FPR: {metrics['fpr']:.4f}"
        )

        if metrics["loss"] < best_val_loss - 1e-4:
            best_val_loss = metrics["loss"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": 6,
                    "hidden_size": HIDDEN_SIZE,
                    "seed": SEED,
                    "best_epoch": best_epoch,
                },
                OUTPUT_DIR / "lstm_best.pt"
            )
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\nEarly stopping.")
            break

    duration = time.perf_counter() - start_time

    # Recuperar o melhor modelo para validação final
    checkpoint = torch.load(
        OUTPUT_DIR / "lstm_best.pt",
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    final_metrics = evaluate(
        model, val_loader, criterion, device
    )

    report = {
        "model": "LSTM",
        "seed": SEED,
        "device": str(device),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "training_seconds": duration,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "hidden_size": HIDDEN_SIZE,
        "parameters": sum(
            p.numel() for p in model.parameters()
        ),
        "validation": final_metrics,
        "history": history
    }

    with open(
        OUTPUT_DIR / "lstm_training_report.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 55)
    print("RESULTADOS DA MELHOR LSTM — VALIDAÇÃO")
    print("=" * 55)

    print(f"Melhor época: {best_epoch}")
    print(f"Tempo de treino: {duration:.2f} s")

    for metric in [
        "accuracy", "precision", "recall",
        "f1", "fpr"
    ]:
        print(
            f"{metric.upper()}: "
            f"{final_metrics[metric]:.4f}"
        )

    print("\nMatriz de confusão:")
    print(
        f"TN={final_metrics['tn']} "
        f"FP={final_metrics['fp']}"
    )
    print(
        f"FN={final_metrics['fn']} "
        f"TP={final_metrics['tp']}"
    )

    print("\nModelo guardado em:")
    print(OUTPUT_DIR / "lstm_best.pt")


if __name__ == "__main__":
    main()
