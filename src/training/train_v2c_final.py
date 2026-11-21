from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.gru import GRUClassifier
from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerClassifier


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "processed" / "ton_iot_v2"
    / "ip_generalization"
)

OUTPUT_DIR = ROOT / "results" / "models" / "v2c_final"

PROTOCOL_PATH = ROOT / "results" / "v2c_final_protocol.json"


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

    raise ValueError(f"Modelo desconhecido: {name}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_final(name, seed, protocol):
    epochs = int(protocol["centralized_epochs"][name])
    batch_size = int(protocol["batch_size"])
    lr = float(protocol["learning_rate"])

    if epochs != 1:
        raise ValueError("Protocolo inesperado: epochs != 1")

    set_seed(seed)
    torch.set_num_threads(1)

    # Apenas treino final. Nenhum ficheiro de avaliacao
    # ou validacao e carregado neste script.
    with np.load(DATA_DIR / "train.npz") as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32).reshape(-1)

    if X.shape != (21011, 5, 6):
        raise ValueError(f"Dimensoes inesperadas: {X.shape}")

    if not np.isfinite(X).all():
        raise ValueError("Existem valores invalidos em X")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Rotulos invalidos")

    generator = torch.Generator().manual_seed(seed)

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y)
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_model(name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\n=== V2-C FINAL | {name.upper()} | SEED {seed} ===")
    print("Janelas de treino:", len(y))
    print("Epocas fixadas:", epochs)
    print("Dispositivo:", device)

    start = time.perf_counter()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X).reshape(-1)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            count = len(batch_y)
            total_loss += loss.item() * count
            total_samples += count

        mean_loss = total_loss / total_samples

        history.append({
            "epoch": epoch,
            "train_loss": mean_loss
        })

        print(
            f"Epoch {epoch:02d} | Train Loss: {mean_loss:.6f}"
        )

    duration = time.perf_counter() - start

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        OUTPUT_DIR / f"{name}_seed{seed}_final.pt"
    )

    report_path = (
        OUTPUT_DIR / f"{name}_seed{seed}_final_report.json"
    )

    # Impedir substituicao acidental de resultados.
    if checkpoint_path.exists() or report_path.exists():
        raise FileExistsError(
            f"Ja existem resultados para {name}, seed {seed}"
        )

    checkpoint = {
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "model_name": name,
        "experiment": "v2c_final_ip_generalization",
        "seed": seed,
        "input_size": 6,
        "hidden_size": 32,
        "fixed_epochs": epochs,
    }

    torch.save(checkpoint, checkpoint_path)

    report = {
        "experiment": "v2c_final_ip_generalization",
        "approach": "centralized",
        "model": name,
        "seed": seed,
        "training_windows": len(y),
        "training_ips": 37,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "train_loss": history[-1]["train_loss"],
        "training_seconds": duration,
        "history": history,
        "checkpoint_selection": "fixed_training_duration",
        "validation_used": False,
        "evaluation_used": False,
        "checkpoint": str(checkpoint_path),
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("Checkpoint:", checkpoint_path)
    print("Relatorio:", report_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["gru", "lstm", "transformer", "all"],
        default="all"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    protocol = json.loads(
        PROTOCOL_PATH.read_text(encoding="utf-8")
    )

    if protocol["status"] != "frozen_before_final_evaluation":
        raise ValueError("Protocolo final nao esta fixado")

    if args.seed not in protocol["seeds"]:
        raise ValueError("Seed nao prevista no protocolo")

    models = (
        protocol["models"]
        if args.model == "all"
        else [args.model]
    )

    for name in models:
        train_final(name, args.seed, protocol)


if __name__ == "__main__":
    main()
