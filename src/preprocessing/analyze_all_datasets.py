
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "ton_iot"
RESULTS_DIR = ROOT / "results" / "tables"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000


def file_order(path):
    match = re.search(r"Network_dataset_(\d+)\.csv$", path.name)
    return int(match.group(1)) if match else 9999


def format_date(timestamp):
    if timestamp is None:
        return "N/A"

    return datetime.fromtimestamp(
        timestamp, timezone.utc
    ).isoformat()


def analyze_file(file):
    print(f"\nA analisar: {file.name}")

    total = 0
    labels = Counter()
    attack_types = Counter()
    daily_counts = Counter()

    min_ts = None
    max_ts = None
    columns = None
    previous_last_ts = None
    timestamp_decreases = 0

    for chunk in pd.read_csv(
        file,
        chunksize=CHUNK_SIZE,
        usecols=lambda col: col in ("ts", "label", "type"),
        low_memory=False,
    ):
        if columns is None:
            columns = list(chunk.columns)

        if not {"ts", "label", "type"}.issubset(chunk.columns):
            raise ValueError(
                f"Colunas obrigatórias em falta: {file.name}"
            )

        total += len(chunk)

        labels.update(
            chunk["label"].value_counts(
                dropna=False
            ).to_dict()
        )

        attack_types.update(
            chunk["type"].value_counts(
                dropna=False
            ).to_dict()
        )

        timestamps = pd.to_numeric(
            chunk["ts"], errors="coerce"
        )

        valid_ts = timestamps.dropna()

        if not valid_ts.empty:
            current_min = valid_ts.min()
            current_max = valid_ts.max()

            min_ts = (
                current_min if min_ts is None
                else min(min_ts, current_min)
            )

            max_ts = (
                current_max if max_ts is None
                else max(max_ts, current_max)
            )

            # Identifica diminuições na ordem temporal
            timestamp_decreases += int(
                (valid_ts.diff() < 0).sum()
            )

            if previous_last_ts is not None:
                if valid_ts.iloc[0] < previous_last_ts:
                    timestamp_decreases += 1

            previous_last_ts = valid_ts.iloc[-1]

        dates = pd.to_datetime(
            timestamps, unit="s",
            utc=True, errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        daily = pd.DataFrame({
            "date": dates,
            "label": chunk["label"]
        })

        counts = (
            daily.dropna(subset=["date"])
            .groupby(["date", "label"])
            .size()
        )

        for key, count in counts.items():
            daily_counts[key] += int(count)

    normal = int(labels.get(0, 0))
    attacks = int(labels.get(1, 0))

    print(f"  Total: {total:,}")
    print(f"  Normal: {normal:,}")
    print(f"  Ataques: {attacks:,}")
    print(f"  Tipos: {dict(attack_types)}")
    print(f"  Intervalo: {format_date(min_ts)}")
    print(f"          a {format_date(max_ts)}")
    print(f"  Quebras na ordem temporal: {timestamp_decreases}")

    summary = {
        "file": file.name,
        "total": total,
        "normal": normal,
        "attacks": attacks,
        "normal_pct": round(100 * normal / total, 2) if total else 0,
        "attack_pct": round(100 * attacks / total, 2) if total else 0,
        "start": format_date(min_ts),
        "end": format_date(max_ts),
        "timestamp_decreases": timestamp_decreases,
        "types": str(dict(attack_types))
    }

    type_rows = [
        {"file": file.name, "type": str(t), "count": int(n)}
        for t, n in attack_types.items()
    ]

    daily_rows = [
        {
            "file": file.name,
            "date": date,
            "label": label,
            "count": count
        }
        for (date, label), count in daily_counts.items()
    ]

    return summary, type_rows, daily_rows


def main():
    files = sorted(
        DATA_DIR.glob("Network_dataset_*.csv"),
        key=file_order
    )

    print(f"Encontrados {len(files)} ficheiros.")

    if len(files) != 23:
        print(
            "AVISO: Esperavam-se 23 ficheiros. "
            "Confirma se o download está completo."
        )

    if not files:
        return

    summaries = []
    all_types = []
    all_daily = []

    for file in files:
        summary, types, daily = analyze_file(file)

        summaries.append(summary)
        all_types.extend(types)
        all_daily.extend(daily)

    pd.DataFrame(summaries).to_csv(
        RESULTS_DIR / "dataset_summary.csv",
        index=False
    )

    pd.DataFrame(all_types).to_csv(
        RESULTS_DIR / "attack_types.csv",
        index=False
    )

    pd.DataFrame(all_daily).to_csv(
        RESULTS_DIR / "daily_distribution.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("RESUMO GLOBAL")
    print("=" * 65)

    summary_df = pd.DataFrame(summaries)

    print(
        summary_df[
            ["file", "total", "normal", "attacks",
             "attack_pct", "timestamp_decreases"]
        ].to_string(index=False)
    )

    print("\nTOTAL DE REGISTOS:",
          f"{summary_df['total'].sum():,}")

    print("TOTAL NORMAL:",
          f"{summary_df['normal'].sum():,}")

    print("TOTAL ATAQUES:",
          f"{summary_df['attacks'].sum():,}")

    print("\nTIPOS DE TRÁFEGO / ATAQUES:")

    type_df = pd.DataFrame(all_types)

    print(
        type_df.groupby("type")["count"]
        .sum()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nResultados guardados em:")
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()
