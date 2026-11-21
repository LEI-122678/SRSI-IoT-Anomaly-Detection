
from pathlib import Path
import argparse
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
from src.models.gru import GRUClassifier
from src.models.transformer import TransformerClassifier


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "processed" / "ton_iot_v2"
    / "ip_generalization" / "internal_validation"
)

OUTPUT_DIR = ROOT / "results" / "models" / "v2c"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CLASSES = {
    "lstm": LSTMClassifier,
    "gru": GRUClassifier,
    "transformer": TransformerClassifier
}

EPOCHS = 50
BATCH_SIZE = 32
HIDDEN_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 8


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(split):
    path = DATA_DIR / f"{split}.npz"

    with np.load(path) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)

    if X.ndim != 3 or X.shape[1:] != (5, 6):
        raise ValueError(
            f"Dimensões inesperadas em {split}: {X.shape}"
        )

    return TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y)
    )


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total = 0
    actual = []
    predicted = []

    with torch.inference_mode():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = criterion(logits, y)

            total_loss += loss.item() * len(y)
            total += len(y)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            actual.extend(
                y.cpu().numpy().astype(int).tolist()
            )

            predicted.extend(
                preds.cpu().numpy().tolist()
            )

    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    return {
        "loss": float(total_loss / total),
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "precision": float(
            precision_score(
                actual, predicted, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                actual, predicted, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                actual, predicted, zero_division=0
            )
        ),
        "fpr": float(
            fp / (fp + tn) if fp + tn else 0
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def train_model(name, seed, device):
    set_seed(seed)

    train_dataset = load_data("internal_train")
    validation_dataset = load_data("internal_validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = MODEL_CLASSES[name](
        input_size=6,
        hidden_size=HIDDEN_SIZE
    ).to(device)

    parameters = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    checkpoint_path = (
        OUTPUT_DIR / f"{name}_seed{seed}_best.pt"
    )

    print(f"\n{'=' * 60}")
    print(f"V2-C | {name.upper()} | Seed {seed}")
    print(f"Dispositivo: {device}")
    print(f"Parâmetros: {parameters}")
    print(f"{'=' * 60}")

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
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            running_loss += loss.item() * len(y)
            seen += len(y)

        train_loss = running_loss / seen

        metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
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

            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": name,
                "experiment": "v2c_internal_validation",
                "input_size": 6,
                "hidden_size": HIDDEN_SIZE,
                "seed": seed,
                "best_epoch": best_epoch
            }, checkpoint_path)

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break

    duration = time.perf_counter() - start_time

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    final_metrics = evaluate(
        model,
        validation_loader,
        criterion,
        device
    )

    report = {
        "experiment": "v2c_ip_generalization_internal",
        "model": name,
        "seed": seed,
        "device": str(device),
        "parameters": parameters,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "training_seconds": duration,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "hidden_size": HIDDEN_SIZE,
        "selection_metric": "validation_loss",
        "validation": final_metrics,
        "history": history
    }

    report_path = (
        OUTPUT_DIR / f"{name}_seed{seed}_report.json"
    )

    with open(
        report_path, "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print(f"\nRESULTADO {name.upper()}")
    print("Melhor época:", best_epoch)
    print("Tempo:", round(duration, 2), "s")

    for key in [
        "accuracy", "precision", "recall", "f1", "fpr"
    ]:
        print(f"{key.upper()}: {final_metrics[key]:.4f}")

    print(
        f"TN={final_metrics['tn']} "
        f"FP={final_metrics['fp']} "
        f"FN={final_metrics['fn']} "
        f"TP={final_metrics['tp']}"
    )

    print("Checkpoint:", checkpoint_path)
    print("Relatório:", report_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["lstm", "gru", "transformer", "all"],
        default="all"
    )

    parser.add_argument(
        "--seed", type=int, default=42
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    names = (
        list(MODEL_CLASSES)
        if args.model == "all"
        else [args.model]
    )

    for name in names:
        train_model(name, args.seed, device)


if __name__ == "__main__":
    main()
