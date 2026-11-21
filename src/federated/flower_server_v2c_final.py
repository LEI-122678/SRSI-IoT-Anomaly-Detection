from pathlib import Path
import json
import os
import random
import time

import numpy as np
import torch

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from src.models.gru import GRUClassifier
from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerClassifier


ROOT = Path(__file__).resolve().parents[2]

PROTOCOL_PATH = ROOT / "results" / "v2c_final_protocol.json"

SEED = int(os.environ.get("SRSI_SEED", "42"))
MODEL_NAME = os.environ.get("SRSI_MODEL", "gru").lower()

PROTOCOL = json.loads(
    PROTOCOL_PATH.read_text(encoding="utf-8")
)

if PROTOCOL["status"] != "frozen_before_final_evaluation":
    raise ValueError("Protocolo final nao esta fixado")

if MODEL_NAME not in PROTOCOL["models"]:
    raise ValueError("Modelo nao previsto no protocolo")

if SEED not in PROTOCOL["seeds"]:
    raise ValueError("Seed nao prevista no protocolo")

NUM_CLIENTS = int(PROTOCOL["federated_clients"])
NUM_ROUNDS = int(PROTOCOL["federated_rounds"][MODEL_NAME])
LEARNING_RATE = float(PROTOCOL["learning_rate"])

OUTPUT_DIR = (
    ROOT / "results" / "federated"
    / f"flower_v2c_final_{MODEL_NAME}_seed{SEED}"
)

EXPERIMENT_NAME = "flower_v2c_final"

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

    raise ValueError(MODEL_NAME)


@app.main()
def main(grid: Grid, context: Context):
    torch.set_num_threads(1)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    checkpoint_path = (
        OUTPUT_DIR / f"{MODEL_NAME}_final.pt"
    )
    report_path = OUTPUT_DIR / "report.json"

    # Evitar substituir checkpoints existentes.
    if checkpoint_path.exists() or report_path.exists():
        raise FileExistsError(
            f"Experiencia final ja existe: {OUTPUT_DIR}"
        )

    model = build_model()
    initial_arrays = ArrayRecord(model.state_dict())

    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=NUM_CLIENTS,
        min_available_nodes=NUM_CLIENTS,
        weighted_by_key="num-examples"
    )

    print("\n=== FLOWER V2-C FINAL ===", flush=True)
    print("Modelo:", MODEL_NAME.upper(), flush=True)
    print("Seed:", SEED, flush=True)
    print("Clientes:", NUM_CLIENTS, flush=True)
    print("Rondas fixas:", NUM_ROUNDS, flush=True)
    print("Learning rate:", LEARNING_RATE, flush=True)
    print("Validacao: NAO UTILIZADA", flush=True)
    print("Avaliacao final: NAO UTILIZADA", flush=True)

    start = time.perf_counter()

    # Treino federado com duracao fixa.
    # Sem evaluate_fn e sem should_stop.
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=NUM_ROUNDS,
        train_config=ConfigRecord({
            "lr": LEARNING_RATE
        })
    )

    duration = time.perf_counter() - start

    if result.arrays is None:
        raise RuntimeError(
            "O Flower nao devolveu parametros globais."
        )

    final_state = result.arrays.to_torch_state_dict()

    # Confirmar compatibilidade do modelo recebido.
    final_model = build_model()
    final_model.load_state_dict(final_state, strict=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in final_model.state_dict().items()
        },
        "model_name": MODEL_NAME,
        "experiment": EXPERIMENT_NAME,
        "seed": SEED,
        "input_size": 6,
        "hidden_size": 32,
        "fixed_rounds": NUM_ROUNDS,
        "num_clients": NUM_CLIENTS,
    }

    torch.save(checkpoint, checkpoint_path)

    report = {
        "experiment": EXPERIMENT_NAME,
        "approach": "federated",
        "algorithm": "FedAvg",
        "model": MODEL_NAME,
        "seed": SEED,
        "training_windows": 21011,
        "training_ips": 37,
        "num_clients": NUM_CLIENTS,
        "fixed_rounds": NUM_ROUNDS,
        "local_epochs": int(
            PROTOCOL["federated_local_epochs"]
        ),
        "batch_size": int(PROTOCOL["batch_size"]),
        "learning_rate": LEARNING_RATE,
        "training_seconds": duration,
        "checkpoint_selection": "fixed_training_duration",
        "validation_used": False,
        "evaluation_used": False,
        "device": "cpu",
        "backend": "ray",
        "framework": "flower",
        "checkpoint": str(checkpoint_path)
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n=== FLOWER FINAL CONCLUIDO ===", flush=True)
    print("Rondas:", NUM_ROUNDS, flush=True)
    print("Checkpoint:", checkpoint_path, flush=True)
    print("Relatorio:", report_path, flush=True)
