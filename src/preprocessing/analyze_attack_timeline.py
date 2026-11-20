
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
        key=lambda p: int(p.stem.split("_")[-1])
    )

    if len(files) != 23:
        raise RuntimeError(
            f"Esperavam-se 23 CSV; encontrados {len(files)}."
        )

    counts = Counter()

    for index, file in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {file.name}")

        for chunk in pd.read_csv(
            file,
            usecols=["ts", "label", "type"],
            chunksize=CHUNK_SIZE
        ):
            dates = pd.to_datetime(
                chunk["ts"],
                unit="s",
                utc=True,
                errors="coerce"
            ).dt.strftime("%Y-%m-%d")

            attack_types = (
                chunk["type"]
                .fillna("unknown")
                .astype(str)
                .str.strip()
                .str.lower()
            )

            batch = pd.DataFrame({
                "date": dates,
                "type": attack_types,
                "label": chunk["label"]
            })

            if batch["date"].isna().any():
                raise ValueError(
                    f"Timestamps inválidos em {file.name}"
                )

            grouped = (
                batch.groupby(["date", "type", "label"])
                .size()
            )

            for key, value in grouped.items():
                counts[key] += int(value)

    rows = [
        {
            "date": date,
            "type": attack_type,
            "label": label,
            "count": count
        }
        for (date, attack_type, label), count
        in counts.items()
    ]

    detail = pd.DataFrame(rows).sort_values(
        ["date", "type", "label"]
    )

    detail.to_csv(
        OUTPUT_DIR / "attack_timeline_detail.csv",
        index=False
    )

    table = detail.pivot_table(
        index="date",
        columns="type",
        values="count",
        aggfunc="sum",
        fill_value=0
    )

    table.to_csv(
        OUTPUT_DIR / "attack_timeline.csv"
    )

    print("\nDISTRIBUIÇÃO DOS ATAQUES POR DIA:")
    print(table.to_string())

    print("\nTOTAL POR TIPO:")
    print(table.sum().sort_values(ascending=False))

    print("\nTOTAL DE REGISTOS:")
    print(int(table.to_numpy().sum()))

    print("\nResultados guardados em:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
