
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flwr.app import (
    ArrayRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp

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

NUM_CLIENTS = 5
BATCH_SIZE = 32
LOCAL_EPOCHS = 1
SEED = 42

app = ClientApp()


def build_model():
    return GRUClassifier(
        input_size=6,
        hidden_size=32
    )


def load_client_data(client_id):
    if not 0 <= client_id < NUM_CLIENTS:
        raise ValueError(
            f"Cliente inválido: {client_id}"
        )

    indices = np.load(
        PARTITION_DIR /
        f"client_{client_id}_indices.npy"
    ).astype(np.int64)

    with np.load(DATA_DIR / "train.npz") as data:
        X = torch.from_numpy(
            data["X"][indices].astype(np.float32)
        )
        y = torch.from_numpy(
            data["y"][indices].astype(np.float32)
        )

    if len(X) == 0:
        raise ValueError(
            f"Cliente {client_id} sem dados"
        )

    return X, y


@app.train()
def train(msg: Message, context: Context):
    # Durante a simulação, o Flower atribui
    # um identificador de partição a cada cliente.
    client_id = int(
        context.node_config["partition-id"]
    )

    round_id = int(
        msg.content["config"]["server-round"]
    )

    lr = float(
        msg.content["config"]["lr"]
    )

    torch.set_num_threads(1)

    # Reprodutibilidade da ordem dos batches.
    generator = torch.Generator().manual_seed(
        SEED + round_id * 100 + client_id
    )

    X, y = load_client_data(client_id)

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    model = build_model()

    # Receber os parâmetros globais enviados
    # pelo servidor Flower.
    model.load_state_dict(
        msg.content["arrays"].to_torch_state_dict()
    )

    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_samples = 0

    for _ in range(LOCAL_EPOCHS):
        for batch_X, batch_y in loader:
            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(batch_X)
            loss = criterion(
                logits,
                batch_y
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            count = len(batch_y)

            total_loss += (
                loss.item() * count
            )

            total_samples += count

    mean_loss = (
        total_loss / total_samples
    )

    # Enviar os parâmetros locais atualizados.
    arrays = ArrayRecord(
        model.state_dict()
    )

    # O FedAvg utiliza "num-examples"
    # para ponderar as contribuições dos clientes.
    metrics = MetricRecord({
        "num-examples": len(y),
        "train_loss": float(mean_loss)
    })

    print(
        f"[Flower] Ronda {round_id:02d} | "
        f"Cliente {client_id} | "
        f"Janelas: {len(y)} | "
        f"Loss: {mean_loss:.4f}",
        flush=True
    )

    return Message(
        content=RecordDict({
            "arrays": arrays,
            "metrics": metrics
        }),
        reply_to=msg
    )
