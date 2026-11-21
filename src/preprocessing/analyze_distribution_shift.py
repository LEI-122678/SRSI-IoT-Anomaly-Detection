
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "temporal_scanning" / "minute_features.csv"
)

OUTPUT_DIR = ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "connections",
    "src_bytes_sum",
    "dst_bytes_sum",
    "src_pkts_sum",
    "dst_pkts_sum",
    "duration_mean",
]

CUTOFF = pd.Timestamp(
    "2019-04-23T14:00:00Z"
)


def main():
    df = pd.read_csv(INPUT_FILE)

    df["datetime"] = pd.to_datetime(
        df["minute"], unit="s", utc=True
    )

    # Reproduzir a divisão temporal do experimento
    conditions = [
        df["datetime"] < CUTOFF,
        (
            (df["datetime"] >= CUTOFF)
            & (
                df["datetime"].dt.strftime("%Y-%m-%d")
                == "2019-04-23"
            )
        ),
        (
            df["datetime"].dt.strftime("%Y-%m-%d")
            == "2019-04-24"
        )
    ]

    df["split"] = np.select(
        conditions,
        ["train", "validation", "test"],
        default="other"
    )

    df = df[df["split"] != "other"].copy()

    print("\n1. DISTRIBUIÇÃO DOS REGISTOS")

    distribution = (
        df.groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            index=["train", "validation", "test"],
            columns=[0, 1],
            fill_value=0
        )
    )

    distribution.columns = ["normal", "scanning"]

    print(distribution)

    print("\n2. IPS POR CONJUNTO")

    ips = {
        name: set(
            df.loc[df["split"] == name, "src_ip"]
            .dropna()
            .astype(str)
        )
        for name in ["train", "validation", "test"]
    }

    for name, values in ips.items():
        print(f"{name}: {len(values)} IPs")

    print(
        "IPs comuns treino/teste:",
        len(ips["train"] & ips["test"])
    )

    print(
        "IPs exclusivos do teste:",
        len(ips["test"] - ips["train"])
    )

    print("\n3. CARACTERÍSTICAS POR CLASSE")

    rows = []

    for (split, label), group in df.groupby(
        ["split", "label"]
    ):
        for feature in FEATURES:
            values = pd.to_numeric(
                group[feature], errors="coerce"
            ).dropna()

            rows.append({
                "split": split,
                "label": int(label),
                "feature": feature,
                "count": len(values),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p90": float(values.quantile(0.90)),
                "p99": float(values.quantile(0.99)),
            })

    summary = pd.DataFrame(rows)

    summary.to_csv(
        OUTPUT_DIR / "distribution_shift_features.csv",
        index=False
    )

    print(
        summary[
            ["split", "label", "feature", "median", "p90"]
        ].to_string(index=False)
    )

    print("\n4. MUDANÇA ENTRE TREINO E TESTE")

    comparisons = []

    for label in [0, 1]:
        print(
            f"\nClasse: "
            f"{'normal' if label == 0 else 'scanning'}"
        )

        for feature in FEATURES:
            train_values = df.loc[
                (df["split"] == "train")
                & (df["label"] == label),
                feature
            ].astype(float)

            test_values = df.loc[
                (df["split"] == "test")
                & (df["label"] == label),
                feature
            ].astype(float)

            train_median = train_values.median()
            test_median = test_values.median()

            comparisons.append({
                "label": label,
                "feature": feature,
                "train_median": train_median,
                "test_median": test_median,
                "train_p90": train_values.quantile(0.90),
                "test_p90": test_values.quantile(0.90)
            })

            print(
                f"{feature}: "
                f"mediana treino={train_median:.2f}, "
                f"teste={test_median:.2f}"
            )

    pd.DataFrame(comparisons).to_csv(
        OUTPUT_DIR / "distribution_shift_comparison.csv",
        index=False
    )

    print("\n5. ANÁLISE CONCLUÍDA")
    print("Ficheiros guardados em:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
