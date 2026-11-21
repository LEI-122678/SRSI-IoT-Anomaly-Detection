
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "ton_iot_v2" / "minute_features.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "processed"
    / "ton_iot_v2" / "temporal_generalization"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 5

FEATURES = [
    "connections",
    "src_bytes_sum",
    "dst_bytes_sum",
    "src_pkts_sum",
    "dst_pkts_sum",
    "duration_mean"
]

ATTACK_TYPES = [
    "scanning", "dos", "ddos", "injection",
    "password", "xss", "backdoor",
    "ransomware", "mitm"
]


def get_split(date):
    if date <= "2019-04-26":
        return "train"
    if date == "2019-04-27":
        return "validation"
    if date in ["2019-04-28", "2019-04-29"]:
        return "test"
    return "other"


def create_windows(df, scaler):
    X, y, metadata = [], [], []

    for (ip, date), group in df.groupby(
        ["src_ip", "date"], sort=False
    ):
        group = group.sort_values("minute")

        # Dividir nas séries realmente consecutivas
        run_number = (
            group["minute"].diff().ne(60)
        ).cumsum()

        for _, run in group.groupby(run_number):
            if len(run) < WINDOW_SIZE:
                continue

            raw_features = run[FEATURES].to_numpy(
                dtype=np.float64
            )

            transformed = scaler.transform(
                np.log1p(raw_features)
            ).astype(np.float32)

            labels = run["label"].to_numpy(
                dtype=np.int64
            )

            for end in range(
                WINDOW_SIZE - 1, len(run)
            ):
                start = end - WINDOW_SIZE + 1

                X.append(
                    transformed[start:end + 1]
                )
                y.append(int(labels[end]))

                last = run.iloc[end]

                row = {
                    "src_ip": ip,
                    "date": date,
                    "end_minute": int(last["minute"]),
                    "label": int(last["label"])
                }

                for attack in ATTACK_TYPES:
                    row[attack] = int(
                        last[f"type_{attack}"] > 0
                    )

                metadata.append(row)

    if not X:
        raise ValueError(
            "Nenhuma janela criada neste split."
        )

    return (
        np.stack(X),
        np.asarray(y, dtype=np.int64),
        pd.DataFrame(metadata)
    )


def main():
    df = pd.read_csv(INPUT_FILE)

    required = (
        ["src_ip", "date", "minute", "label"]
        + FEATURES
        + [f"type_{x}" for x in ATTACK_TYPES]
    )

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Colunas em falta: {sorted(missing)}"
        )

    df["split"] = df["date"].apply(get_split)
    df = df[df["split"] != "other"].copy()

    if df[FEATURES].isna().any().any():
        raise ValueError(
            "Existem características em falta."
        )

    if (df[FEATURES] < 0).any().any():
        raise ValueError(
            "Existem características negativas."
        )

    train_df = df[df["split"] == "train"]

    if train_df.empty:
        raise ValueError("Treino vazio.")

    # Importante: o scaler aprende APENAS
    # a partir de dados de treino.
    scaler = StandardScaler()

    scaler.fit(
        np.log1p(
            train_df[FEATURES].to_numpy(
                dtype=np.float64
            )
        )
    )

    scaler_info = {
        "features": FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "window_size": WINDOW_SIZE,
        "transform": "log1p_then_standard_scaler",
        "fitted_on": "train_only"
    }

    with open(
        OUTPUT_DIR / "scaler.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(scaler_info, f, indent=2)

    report = {
        "experiment": "V2-B_temporal_generalization",
        "window_size": WINDOW_SIZE,
        "features": FEATURES,
        "splits": {}
    }

    for split in ["train", "validation", "test"]:
        subset = df[df["split"] == split].copy()

        X, y, metadata = create_windows(
            subset, scaler
        )

        np.savez_compressed(
            OUTPUT_DIR / f"{split}.npz",
            X=X,
            y=y
        )

        metadata.to_csv(
            OUTPUT_DIR / f"{split}_metadata.csv",
            index=False
        )

        attack_counts = {
            attack: int(metadata[attack].sum())
            for attack in ATTACK_TYPES
        }

        report["splits"][split] = {
            "windows": len(y),
            "normal": int((y == 0).sum()),
            "attack": int((y == 1).sum()),
            "unique_ips": int(
                metadata["src_ip"].nunique()
            ),
            "attack_types": attack_counts
        }

        print(f"\n=== {split.upper()} ===")
        print("X:", X.shape)
        print("y:", y.shape)
        print("Normal:", int((y == 0).sum()))
        print("Ataque:", int((y == 1).sum()))
        print(
            "IPs:", metadata["src_ip"].nunique()
        )

        print("Ataques:")
        for attack, count in attack_counts.items():
            if count:
                print(f"  {attack}: {count}")

    with open(
        OUTPUT_DIR / "preparation_report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("\nPreparação V2-B concluída.")
    print("Resultados:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
