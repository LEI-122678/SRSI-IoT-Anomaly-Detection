
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "raw" / "ton_iot"
    / "Network_dataset_1.csv"
)

OUTPUT_DIR = ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("A analisar distribuição temporal...")

    chunks = []

    for chunk in pd.read_csv(
        DATA_PATH,
        usecols=["ts", "label", "type"],
        chunksize=100_000
    ):
        chunk["datetime"] = pd.to_datetime(
            chunk["ts"],
            unit="s",
            utc=True
        )

        chunk["date"] = chunk["datetime"].dt.date

        daily = (
            chunk.groupby(["date", "label"])
            .size()
            .reset_index(name="count")
        )

        chunks.append(daily)

    df = pd.concat(chunks, ignore_index=True)

    daily_counts = (
        df.groupby(["date", "label"])["count"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )

    daily_counts = daily_counts.rename(
        columns={0: "Normal", 1: "Ataque"}
    )

    print("\nREGISTOS POR DIA:")
    print(daily_counts.to_string())

    print("\nTOTAL POR CLASSE:")
    print(daily_counts.sum())

    ax = daily_counts.plot(
        kind="bar",
        figsize=(13, 6),
        width=0.8
    )

    ax.set_title(
        "Distribuição temporal — ToN_IoT Network Dataset 1"
    )
    ax.set_xlabel("Data")
    ax.set_ylabel("Número de registos")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output = (
        OUTPUT_DIR /
        "network_dataset_1_temporal_distribution.png"
    )

    plt.savefig(output, dpi=150)
    print(f"\nGráfico guardado em: {output}")

    plt.show()


if __name__ == "__main__":
    main()
