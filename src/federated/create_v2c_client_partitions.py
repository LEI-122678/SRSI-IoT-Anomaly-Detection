
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "processed"
    / "ton_iot_v2" / "ip_generalization" / "internal_validation"
)

OUTPUT_DIR = (
    ROOT / "data" / "processed"
    / "ton_iot_v2" / "ip_generalization" / "federated_internal"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLIENTS = 5


def main():
    metadata = pd.read_csv(
        DATA_DIR / "internal_train_metadata.csv"
    )

    with np.load(DATA_DIR / "internal_train.npz") as data:
        y = data["y"].astype(np.int64)

    if len(metadata) != len(y):
        raise ValueError(
            "Metadata e treino têm dimensões diferentes."
        )

    if not np.array_equal(
        metadata["label"].to_numpy(), y
    ):
        raise ValueError(
            "Os rótulos não coincidem."
        )

    # Número de janelas por IP e por classe
    ip_stats = (
        metadata.groupby("src_ip")["label"]
        .agg(["size", "sum"])
        .reset_index()
    )

    ip_stats.columns = [
        "src_ip", "windows", "attacks"
    ]

    ip_stats["normal"] = (
        ip_stats["windows"] - ip_stats["attacks"]
    )

    # Ordenação determinística: maiores IPs primeiro
    ip_stats = ip_stats.sort_values(
        ["windows", "src_ip"],
        ascending=[False, True]
    )

    # Distribuir IPs inteiros pelo cliente
    # que tiver menos janelas acumuladas.
    clients = {
        i: {
            "ips": [],
            "windows": 0,
            "attacks": 0,
            "normal": 0
        }
        for i in range(NUM_CLIENTS)
    }

    ip_to_client = {}

    for row in ip_stats.itertuples(index=False):
        client_id = min(
            clients,
            key=lambda i: (
                clients[i]["windows"], i
            )
        )

        clients[client_id]["ips"].append(
            row.src_ip
        )

        clients[client_id]["windows"] += int(
            row.windows
        )

        clients[client_id]["attacks"] += int(
            row.attacks
        )

        clients[client_id]["normal"] += int(
            row.normal
        )

        ip_to_client[row.src_ip] = client_id

    metadata["client_id"] = (
        metadata["src_ip"].map(ip_to_client)
    )

    if metadata["client_id"].isna().any():
        raise ValueError(
            "Existem IPs sem cliente atribuído."
        )

    summary = []

    all_indices = []

    for client_id in range(NUM_CLIENTS):
        indices = np.flatnonzero(
            metadata["client_id"].to_numpy()
            == client_id
        ).astype(np.int64)

        if len(indices) == 0:
            raise ValueError(
                f"Cliente {client_id} sem dados."
            )

        np.save(
            OUTPUT_DIR / f"client_{client_id}_indices.npy",
            indices
        )

        all_indices.append(indices)

        info = clients[client_id]

        row = {
            "client_id": client_id,
            "ips": len(info["ips"]),
            "windows": info["windows"],
            "normal": info["normal"],
            "attacks": info["attacks"],
            "attack_pct": round(
                100 * info["attacks"]
                / info["windows"], 2
            )
        }

        summary.append(row)

    # Confirmar que nenhuma janela foi
    # duplicada ou perdida.
    combined = np.concatenate(all_indices)

    if len(combined) != len(y):
        raise ValueError(
            "Número incorreto de janelas atribuídas."
        )

    if not np.array_equal(
        np.sort(combined),
        np.arange(len(y))
    ):
        raise ValueError(
            "Partições com duplicações ou omissões."
        )

    # Cada IP pertence a apenas um cliente.
    if (
        metadata.groupby("src_ip")[
            "client_id"
        ].nunique() > 1
    ).any():
        raise ValueError(
            "Um IP foi atribuído a vários clientes."
        )

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        OUTPUT_DIR / "client_summary.csv",
        index=False
    )

    metadata[
        ["src_ip", "client_id"]
    ].drop_duplicates().sort_values(
        ["client_id", "src_ip"]
    ).to_csv(
        OUTPUT_DIR / "ip_client_mapping.csv",
        index=False
    )

    manifest = {
        "experiment": "v2c_federated_internal",
        "num_clients": NUM_CLIENTS,
        "strategy": "greedy_by_ip_window_count",
        "train_windows": len(y),
        "client_ids": list(range(NUM_CLIENTS)),
        "data_source": str(DATA_DIR),
        "notes": (
            "Logical clients grouped by source IP. "
            "No source IP shared across clients. "
            "Internal validation and reserved evaluation excluded."
        )
    }

    with open(
        OUTPUT_DIR / "partition_manifest.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=2)

    print("=== CLIENTES FEDERADOS V2-B ===")
    print(
        summary_df.to_string(index=False)
    )

    print("\nTotal de janelas:", len(y))
    print("Total de IPs:", len(ip_stats))
    print(
        "Verificação das partições: OK"
    )

    print("\nResultados:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
