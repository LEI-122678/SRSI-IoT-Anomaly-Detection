
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "temporal_scanning" / "minute_features.csv"
)

OUTPUT_DIR = ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2019-04-23T14:00:00Z")


def main():
    df = pd.read_csv(INPUT_FILE)

    df["datetime"] = pd.to_datetime(
        df["minute"], unit="s", utc=True
    )

    df["split"] = "other"

    df.loc[
        (df["datetime"].dt.date ==
         pd.Timestamp("2019-04-23").date())
        & (df["datetime"] < CUTOFF),
        "split"
    ] = "train"

    df.loc[
        (df["datetime"].dt.date ==
         pd.Timestamp("2019-04-23").date())
        & (df["datetime"] >= CUTOFF),
        "split"
    ] = "validation"

    df.loc[
        df["datetime"].dt.date ==
        pd.Timestamp("2019-04-24").date(),
        "split"
    ] = "test"

    df = df[df["split"] != "other"].copy()

    train_ips = set(
        df.loc[df["split"] == "train", "src_ip"]
    )

    df["ip_seen_in_train"] = (
        df["src_ip"].isin(train_ips)
    )

    summary = (
        df.groupby(
            ["split", "label", "src_ip",
             "ip_seen_in_train"],
            as_index=False
        )
        .agg(
            minutes=("minute", "size"),
            median_connections=("connections", "median"),
            median_src_pkts=("src_pkts_sum", "median"),
            p90_src_pkts=(
                "src_pkts_sum",
                lambda x: x.quantile(0.90)
            ),
            median_src_bytes=("src_bytes_sum", "median"),
            median_duration=("duration_mean", "median")
        )
    )

    summary.to_csv(
        OUTPUT_DIR / "ip_distribution_audit.csv",
        index=False
    )

    print("\n1. PRINCIPAIS IPS COM TRÁFEGO NORMAL")

    for split in ["train", "test"]:
        subset = summary[
            (summary["split"] == split)
            & (summary["label"] == 0)
        ].sort_values(
            "minutes", ascending=False
        )

        print(f"\n{split.upper()}")
        print(
            subset[
                [
                    "src_ip",
                    "minutes",
                    "median_connections",
                    "median_src_pkts",
                    "median_src_bytes",
                    "ip_seen_in_train"
                ]
            ].head(15).to_string(index=False)
        )

    print("\n2. TESTE: IPS NOVOS VS CONHECIDOS")

    test = df[df["split"] == "test"]

    comparison = (
        test.groupby(
            ["ip_seen_in_train", "label"],
            as_index=False
        )
        .agg(
            minutes=("minute", "size"),
            median_connections=("connections", "median"),
            median_src_pkts=("src_pkts_sum", "median"),
            median_src_bytes=("src_bytes_sum", "median")
        )
    )

    print(comparison.to_string(index=False))

    comparison.to_csv(
        OUTPUT_DIR / "ip_seen_unseen_comparison.csv",
        index=False
    )

    print("\n3. VALORES EXTREMOS DE PACOTES")

    extremes = df.sort_values(
        "src_pkts_sum",
        ascending=False
    ).head(20)

    print(
        extremes[
            [
                "split",
                "src_ip",
                "datetime",
                "label",
                "connections",
                "src_pkts_sum",
                "src_bytes_sum"
            ]
        ].to_string(index=False)
    )

    print("\nAuditoria concluída.")


if __name__ == "__main__":
    main()
