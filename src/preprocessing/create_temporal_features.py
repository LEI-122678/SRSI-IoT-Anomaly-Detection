
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "ton_iot"
OUT_DIR = ROOT / "data" / "processed" / "temporal_scanning"

OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000

NUMERIC = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "src_pkts",
    "dst_pkts",
]

COLUMNS = [
    "ts", "src_ip", "label", "type",
    *NUMERIC
]

PARTIAL_FILE = OUT_DIR / "_partial_minutes.csv"
OUTPUT_FILE = OUT_DIR / "minute_features.csv"


def aggregate(df):
    df = df.copy()

    df["minute"] = (
        (df["ts"].astype("int64") // 60) * 60
    )

    for col in NUMERIC:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        ).fillna(0)

    grouped = df.groupby(
        ["src_ip", "minute"], as_index=False
    ).agg(
        connections=("label", "size"),
        duration_sum=("duration", "sum"),
        src_bytes_sum=("src_bytes", "sum"),
        dst_bytes_sum=("dst_bytes", "sum"),
        src_pkts_sum=("src_pkts", "sum"),
        dst_pkts_sum=("dst_pkts", "sum"),
        label=("label", "max"),
    )

    return grouped


def main():
    files = sorted(
        RAW_DIR.glob("Network_dataset_*.csv"),
        key=lambda p: int(p.stem.split("_")[-1])
    )

    if len(files) != 23:
        raise RuntimeError(
            f"Esperados 23 ficheiros; encontrados {len(files)}"
        )

    if PARTIAL_FILE.exists():
        PARTIAL_FILE.unlink()

    first_write = True

    for i, file in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {file.name}")

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            usecols=lambda c: c in COLUMNS,
            low_memory=False
        ):
            chunk["ts"] = pd.to_numeric(
                chunk["ts"], errors="coerce"
            )

            chunk = chunk.dropna(
                subset=["ts", "src_ip"]
            )

            kind = (
                chunk["type"]
                .astype(str)
                .str.lower()
                .str.strip()
            )

            label = pd.to_numeric(
                chunk["label"], errors="coerce"
            )

            valid = (
                ((kind == "normal") & (label == 0))
                | ((kind == "scanning") & (label == 1))
            )

            chunk = chunk.loc[valid].copy()

            if chunk.empty:
                continue

            dates = pd.to_datetime(
                chunk["ts"], unit="s", utc=True
            ).dt.strftime("%Y-%m-%d")

            chunk = chunk[
                dates.isin([
                    "2019-04-23",
                    "2019-04-24"
                ])
            ]

            if chunk.empty:
                continue

            part = aggregate(chunk)

            part.to_csv(
                PARTIAL_FILE,
                mode="w" if first_write else "a",
                header=first_write,
                index=False
            )

            first_write = False

    if first_write:
        raise RuntimeError("Não foram encontrados dados.")

    print("\nA consolidar minutos...")

    df = pd.read_csv(PARTIAL_FILE)

    df = df.groupby(
        ["src_ip", "minute"],
        as_index=False
    ).agg({
        "connections": "sum",
        "duration_sum": "sum",
        "src_bytes_sum": "sum",
        "dst_bytes_sum": "sum",
        "src_pkts_sum": "sum",
        "dst_pkts_sum": "sum",
        "label": "max"
    })

    df["duration_mean"] = (
        df["duration_sum"] /
        df["connections"].clip(lower=1)
    )

    df = df.drop(columns=["duration_sum"])

    df = df.sort_values(
        ["src_ip", "minute"]
    ).reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSÉRIES TEMPORAIS CRIADAS")
    print(f"Intervalos IP/minuto: {len(df):,}")
    print(f"IPs de origem: {df['src_ip'].nunique():,}")

    print("\nDistribuição das classes:")
    print(df["label"].value_counts())

    print("\nIntervalo temporal:")
    print(pd.to_datetime(
        df["minute"].min(), unit="s", utc=True
    ))
    print(pd.to_datetime(
        df["minute"].max(), unit="s", utc=True
    ))

    print(f"\nGuardado em: {OUTPUT_FILE}")

    PARTIAL_FILE.unlink()


if __name__ == "__main__":
    main()
