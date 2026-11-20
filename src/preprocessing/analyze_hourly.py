
from pathlib import Path
from collections import Counter
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "raw" / "ton_iot"
OUTPUT_DIR = ROOT / "results" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000


def main():
    files = sorted(
        DATA_DIR.glob("Network_dataset_*.csv"),
        key=lambda f: int(f.stem.split("_")[-1])
    )

    if len(files) != 23:
        raise RuntimeError(
            f"Esperados 23 CSV, encontrados {len(files)}"
        )

    hourly_counts = Counter()

    for i, file in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {file.name}")

        for chunk in pd.read_csv(
            file,
            usecols=["ts", "label", "type"],
            chunksize=CHUNK_SIZE
        ):
            timestamp = pd.to_numeric(
                chunk["ts"], errors="coerce"
            )

            hour = pd.to_datetime(
                timestamp,
                unit="s",
                utc=True,
                errors="coerce"
            ).dt.floor("h")

            attack_type = (
                chunk["type"]
                .fillna("unknown")
                .astype(str)
                .str.strip()
                .str.lower()
            )

            batch = pd.DataFrame({
                "hour": hour,
                "type": attack_type,
                "label": chunk["label"]
            }).dropna(subset=["hour"])

            grouped = (
                batch.groupby(["hour", "type", "label"])
                .size()
            )

            for key, count in grouped.items():
                hourly_counts[key] += int(count)

    rows = [
        {
            "hour": hour.isoformat(),
            "type": attack_type,
            "label": label,
            "count": count
        }
        for (hour, attack_type, label), count
        in hourly_counts.items()
    ]

    df = pd.DataFrame(rows).sort_values(
        ["hour", "type", "label"]
    )

    output = OUTPUT_DIR / "hourly_distribution.csv"
    df.to_csv(output, index=False)

    summary = (
        df.groupby(["hour", "label"])["count"]
        .sum()
        .unstack(fill_value=0)
        .rename(columns={0: "Normal", 1: "Ataque"})
    )

    print("\nDISTRIBUIÇÃO POR HORA:")
    print(summary.to_string())

    print("\nHORAS COM AMBAS AS CLASSES:")

    mixed = summary[
        (summary.get("Normal", 0) > 0)
        & (summary.get("Ataque", 0) > 0)
    ]

    print(mixed.to_string())

    print(f"\nTotal de horas com ambas: {len(mixed)}")
    print(f"\nResultado guardado em: {output}")


if __name__ == "__main__":
    main()
