
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "temporal_scanning" / "minute_features.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "processed"
    / "temporal_scanning" / "model_data"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 5
STRIDE = 1

CUTOFF = int(
    pd.Timestamp("2019-04-23T14:00:00Z").timestamp()
)

FEATURES = [
    "connections",
    "src_bytes_sum",
    "dst_bytes_sum",
    "src_pkts_sum",
    "dst_pkts_sum",
    "duration_mean",
]


def create_windows(df):
    X, y, metadata = [], [], []

    for ip, group in df.groupby("src_ip"):
        group = group.sort_values("minute").copy()

        # Uma quebra temporal inicia uma nova sequência.
        run_id = group["minute"].diff().ne(60).cumsum()

        for _, run in group.groupby(run_id):
            if len(run) < WINDOW_SIZE:
                continue

            values = run[FEATURES].to_numpy(
                dtype=np.float32
            )
            labels = run["label"].to_numpy(
                dtype=np.int64
            )
            minutes = run["minute"].to_numpy(
                dtype=np.int64
            )

            for start in range(
                0, len(run) - WINDOW_SIZE + 1, STRIDE
            ):
                end = start + WINDOW_SIZE

                X.append(values[start:end])
                y.append(labels[end - 1])

                metadata.append({
                    "src_ip": ip,
                    "start_minute": int(minutes[start]),
                    "end_minute": int(minutes[end - 1])
                })

    if not X:
        raise ValueError(
            "Não foram encontradas sequências válidas."
        )

    return (
        np.stack(X),
        np.asarray(y, dtype=np.int64),
        pd.DataFrame(metadata)
    )


def main():
    df = pd.read_csv(INPUT_FILE)

    required = {"src_ip", "minute", "label", *FEATURES}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Colunas em falta: {missing}")

    df["minute"] = pd.to_numeric(
        df["minute"], errors="raise"
    ).astype("int64")

    df["label"] = pd.to_numeric(
        df["label"], errors="raise"
    ).astype("int64")

    if not df["label"].isin([0, 1]).all():
        raise ValueError("Labels inválidas.")

    if df[["src_ip", "minute"]].duplicated().any():
        raise ValueError(
            "Existem minutos duplicados para o mesmo IP."
        )

    for col in FEATURES:
        df[col] = pd.to_numeric(
            df[col], errors="raise"
        )

    values = df[FEATURES].to_numpy()

    if not np.isfinite(values).all():
        raise ValueError("Valores NaN ou infinitos.")

    if (values < 0).any():
        raise ValueError(
            "Valores negativos nas características."
        )

    dates = pd.to_datetime(
        df["minute"], unit="s", utc=True
    ).dt.strftime("%Y-%m-%d")

    train_df = df[
        (dates == "2019-04-23")
        & (df["minute"] < CUTOFF)
    ].copy()

    val_df = df[
        (dates == "2019-04-23")
        & (df["minute"] >= CUTOFF)
    ].copy()

    test_df = df[
        dates == "2019-04-24"
    ].copy()

    splits = {
        "train": train_df,
        "validation": val_df,
        "test": test_df
    }

    # Reduzir assimetria das características positivas.
    # Ajustar o StandardScaler apenas com o treino.
    scaler = StandardScaler()

    train_log = np.log1p(
        train_df[FEATURES].to_numpy(
            dtype=np.float64
        )
    )

    scaler.fit(train_log)

    report = {
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "features": FEATURES,
        "cutoff_utc": "2019-04-23T14:00:00Z",
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "splits": {}
    }

    for name, split_df in splits.items():
        split_df = split_df.copy()

        transformed = scaler.transform(
            np.log1p(
                split_df[FEATURES].to_numpy(
                    dtype=np.float64
                )
            )
        )

        split_df[FEATURES] = transformed.astype(
            np.float32
        )

        X, y, metadata = create_windows(split_df)

        if len(np.unique(y)) != 2:
            raise ValueError(
                f"{name}: não contém ambas as classes."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                f"{name}: valores inválidos nas sequências."
            )

        np.savez_compressed(
            OUTPUT_DIR / f"{name}.npz",
            X=X,
            y=y
        )

        metadata.to_csv(
            OUTPUT_DIR / f"{name}_metadata.csv",
            index=False
        )

        counts = np.bincount(y, minlength=2)

        report["splits"][name] = {
            "windows": int(len(y)),
            "normal": int(counts[0]),
            "scanning": int(counts[1]),
            "shape": list(X.shape)
        }

        print(f"\n{name.upper()}")
        print(f"Sequências: {len(y):,}")
        print(f"Normal: {counts[0]}")
        print(f"Scanning: {counts[1]}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

    with open(
        OUTPUT_DIR / "model_data_report.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print("\nPreparação concluída!")
    print(f"Resultados em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
