
from pathlib import Path

import numpy as np
import os
import hashlib
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
from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerClassifier


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "processed"
    / "ton_iot_v2" / "ip_generalization"
)

PARTITION_DIR = (
    ROOT / "data" / "processed"
    / "ton_iot_v2" / "ip_generalization" / "federated_final"
)

NUM_CLIENTS = 5
BATCH_SIZE = 32
LOCAL_EPOCHS = 1
SEED = int(os.environ.get("SRSI_SEED", "42"))
MODEL_NAME = os.environ.get("SRSI_MODEL", "gru").lower()

app = ClientApp()


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

    # Solicitar operações determinísticas.
    torch.use_deterministic_algorithms(True)

    # Seed específica para cada cliente e ronda.
    local_seed = SEED + round_id * 100 + client_id

    torch.manual_seed(local_seed)
    np.random.seed(local_seed)

    # Reprodutibilidade da ordem dos batches.
    generator = torch.Generator().manual_seed(
        local_seed
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

    def fingerprint(state):
        digest = hashlib.sha256()

        for name, tensor in sorted(state.items()):
            digest.update(name.encode("utf-8"))
            digest.update(
                tensor.detach()
                .cpu()
                .contiguous()
                .numpy()
                .tobytes()
            )

        return digest.hexdigest()[:16]

    initial_hash = fingerprint(model.state_dict())

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

    final_hash = fingerprint(model.state_dict())

    print(
        f"[DIAGNOSTICO] ronda={round_id} "
        f"cliente={client_id} "
        f"initial={initial_hash} "
        f"final={final_hash}",
        flush=True
    )

    return Message(
        content=RecordDict({
            "arrays": arrays,
            "metrics": metrics
        }),
        reply_to=msg
    )
