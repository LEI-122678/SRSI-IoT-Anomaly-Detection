
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "ton_iot_v2" / "minute_features.csv"
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


def get_split(date):
    if date <= "2019-04-26":
        return "train"
    if date == "2019-04-27":
        return "validation"
    return "other"


def main():
    df = pd.read_csv(INPUT_FILE)

    required = [
        "src_ip", "date", "minute", "label"
    ] + FEATURES

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Colunas em falta: {sorted(missing)}"
        )

    df["split"] = df["date"].apply(get_split)
    df = df[df["split"] != "other"].copy()

    train = df[df["split"] == "train"]
    validation = df[df["split"] == "validation"]

    if train.empty or validation.empty:
        raise ValueError(
            "Treino ou validação sem dados."
        )

    print("=== DISTRIBUTION SHIFT V2-B ===")
    print("Minutos de treino:", len(train))
    print("Minutos de validação:", len(validation))

    # 1. Distribuição das características por classe
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

    feature_summary = pd.DataFrame(rows)

    feature_summary.to_csv(
        OUTPUT_DIR / "v2b_feature_distribution_shift.csv",
        index=False
    )

    # 2. Generalização entre IPs
    train_ips = set(train["src_ip"].dropna())

    validation = validation.copy()
    validation["ip_seen_in_train"] = (
        validation["src_ip"].isin(train_ips)
    )

    ip_summary = (
        validation.groupby(
            ["ip_seen_in_train", "label"],
            as_index=False
        )
        .agg(
            minutes=("minute", "size"),
            unique_ips=("src_ip", "nunique"),
        )
    )

    ip_summary.to_csv(
        OUTPUT_DIR / "v2b_ip_generalization.csv",
        index=False
    )

    # 3. Comparação das medianas por classe
    comparisons = []

    for label in [0, 1]:
        for feature in FEATURES:
            train_values = train.loc[
                train["label"] == label, feature
            ]

            val_values = validation.loc[
                validation["label"] == label, feature
            ]

            train_median = float(
                train_values.median()
            )
            val_median = float(
                val_values.median()
            )

            comparisons.append({
                "label": label,
                "feature": feature,
                "train_median": train_median,
                "validation_median": val_median,
                "absolute_difference": abs(
                    val_median - train_median
                ),
                "train_p90": float(
                    train_values.quantile(0.90)
                ),
                "validation_p90": float(
                    val_values.quantile(0.90)
                ),
            })

    comparison_df = pd.DataFrame(comparisons)

    comparison_df.to_csv(
        OUTPUT_DIR / "v2b_distribution_shift_summary.csv",
        index=False
    )

    print("\n=== IPs ===")
    print("IPs no treino:", len(train_ips))
    print(
        "IPs na validação:",
        validation["src_ip"].nunique()
    )
    print(
        "IPs novos na validação:",
        len(
            set(validation["src_ip"]) - train_ips
        )
    )

    print("\n=== GENERALIZAÇÃO POR IP ===")
    print(ip_summary.to_string(index=False))

    print("\n=== COMPARAÇÃO DAS MEDIANAS ===")
    print(
        comparison_df[
            [
                "label",
                "feature",
                "train_median",
                "validation_median",
                "absolute_difference",
            ]
        ].to_string(index=False)
    )

    print("\nAnálise concluída.")
    print("Resultados:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
