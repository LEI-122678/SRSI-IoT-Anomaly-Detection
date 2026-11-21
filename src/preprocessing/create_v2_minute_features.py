
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "ton_iot"
OUTPUT_DIR = ROOT / "data" / "processed" / "ton_iot_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARTIAL_FILE = OUTPUT_DIR / "_minute_partials.csv"
OUTPUT_FILE = OUTPUT_DIR / "minute_features.csv"

CHUNK_SIZE = 100_000

ATTACK_TYPES = [
    "scanning", "dos", "ddos", "injection",
    "password", "xss", "backdoor",
    "ransomware", "mitm"
]

COLUMNS = [
    "ts", "src_ip", "label", "type",
    "src_bytes", "dst_bytes",
    "src_pkts", "dst_pkts", "duration"
]

FEATURES = [
    "src_bytes", "dst_bytes",
    "src_pkts", "dst_pkts"
]

COUNT_COLUMNS = [
    "type_" + name for name in
    ["normal"] + ATTACK_TYPES
]

SUM_COLUMNS = [
    "connections",
    "src_bytes_sum", "dst_bytes_sum",
    "src_pkts_sum", "dst_pkts_sum",
    "duration_sum", "duration_count",
    "attack_connections"
] + COUNT_COLUMNS


def process_chunk(chunk):
    chunk = chunk.copy()

    chunk["ts"] = pd.to_numeric(
        chunk["ts"], errors="coerce"
    )

    chunk["label"] = pd.to_numeric(
        chunk["label"], errors="coerce"
    )

    chunk["type"] = (
        chunk["type"].astype("string")
        .str.strip().str.lower()
    )

    valid_types = ["normal"] + ATTACK_TYPES

    valid = (
        chunk["ts"].notna()
        & chunk["src_ip"].notna()
        & chunk["label"].isin([0, 1])
        & chunk["type"].isin(valid_types)
    )

    chunk = chunk.loc[valid].copy()

    if chunk.empty:
        return None

    inconsistent = (
        ((chunk["type"] == "normal")
         & (chunk["label"] != 0))
        | ((chunk["type"] != "normal")
           & (chunk["label"] != 1))
    )

    if inconsistent.any():
        raise ValueError(
            "Encontrados rótulos inconsistentes."
        )

    chunk["minute"] = (
        np.floor(chunk["ts"] / 60)
        .astype("int64") * 60
    )

    for feature in FEATURES + ["duration"]:
        chunk[feature] = pd.to_numeric(
            chunk[feature], errors="coerce"
        )

    # Guardar contagem de durações válidas
    chunk["duration_count"] = (
        chunk["duration"].notna().astype(int)
    )

    # Não assumir que valores negativos são válidos
    for feature in FEATURES + ["duration"]:
        if (chunk[feature].dropna() < 0).any():
            raise ValueError(
                f"Valor negativo em {feature}"
            )

        chunk[feature] = (
            chunk[feature].fillna(0)
        )

    chunk["connections"] = 1
    chunk["attack_connections"] = (
        chunk["label"] == 1
    ).astype(int)

    for kind in ["normal"] + ATTACK_TYPES:
        chunk["type_" + kind] = (
            chunk["type"] == kind
        ).astype(int)

    grouped = chunk.groupby(
        ["src_ip", "minute"], as_index=False
    ).agg({
        "connections": "sum",
        "src_bytes": "sum",
        "dst_bytes": "sum",
        "src_pkts": "sum",
        "dst_pkts": "sum",
        "duration": "sum",
        "duration_count": "sum",
        "attack_connections": "sum",
        **{col: "sum" for col in COUNT_COLUMNS}
    })

    grouped = grouped.rename(columns={
        "src_bytes": "src_bytes_sum",
        "dst_bytes": "dst_bytes_sum",
        "src_pkts": "src_pkts_sum",
        "dst_pkts": "dst_pkts_sum",
        "duration": "duration_sum"
    })

    return grouped


def main():
    if PARTIAL_FILE.exists():
        PARTIAL_FILE.unlink()

    files = sorted(
        RAW_DIR.glob("Network_dataset_*.csv"),
        key=lambda p: int(p.stem.split("_")[-1])
    )

    if not files:
        raise FileNotFoundError(
            f"Sem CSV em {RAW_DIR}"
        )

    print(f"Ficheiros encontrados: {len(files)}")

    first_write = True

    for file in files:
        rows_read = 0

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            usecols=COLUMNS,
            low_memory=False
        ):
            rows_read += len(chunk)
            grouped = process_chunk(chunk)

            if grouped is None:
                continue

            grouped.to_csv(
                PARTIAL_FILE,
                mode="w" if first_write else "a",
                header=first_write,
                index=False
            )

            first_write = False

        print(
            f"{file.name}: "
            f"{rows_read:,} registos lidos"
        )

    if first_write:
        raise ValueError(
            "Nenhum registo válido encontrado."
        )

    print("\nA consolidar agregações...")

    partials = pd.read_csv(PARTIAL_FILE)

    final = (
        partials.groupby(
            ["src_ip", "minute"],
            as_index=False
        )[SUM_COLUMNS].sum()
    )

    final["duration_mean"] = np.where(
        final["duration_count"] > 0,
        final["duration_sum"]
        / final["duration_count"].clip(lower=1),
        0.0
    )

    # Um minuto é ataque se tiver pelo menos
    # uma ligação com label=1.
    final["label"] = (
        final["attack_connections"] > 0
    ).astype(int)

    final["date"] = (
        pd.to_datetime(
            final["minute"], unit="s", utc=True
        ).dt.strftime("%Y-%m-%d")
    )

    final = final.drop(
        columns=["duration_sum", "duration_count"]
    )

    final = final.sort_values(
        ["src_ip", "minute"]
    ).reset_index(drop=True)

    final.to_csv(OUTPUT_FILE, index=False)

    print("\n=== DATASET TEMPORAL V2 ===")
    print("Linhas:", len(final))
    print("IPs:", final["src_ip"].nunique())
    print("Dias:", final["date"].nunique())

    print("\nClasses:")
    print(final["label"].value_counts())

    print("\nMinutos com cada tipo de ataque:")
    for kind in ATTACK_TYPES:
        count = (
            final["type_" + kind] > 0
        ).sum()
        print(f"{kind}: {count}")

    print("\nFicheiro:", OUTPUT_FILE)

    # Remover apenas o ficheiro temporário
    PARTIAL_FILE.unlink()


if __name__ == "__main__":
    main()

