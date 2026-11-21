
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
    / "ton_iot_v2" / "ip_generalization"
)

WINDOW_SIZE = 5

FEATURES = [
    "connections",
    "src_bytes_sum",
    "dst_bytes_sum",
    "src_pkts_sum",
    "dst_pkts_sum",
    "duration_mean",
]

ATTACK_TYPES = [
    "scanning",
    "dos",
    "ddos",
    "injection",
    "password",
    "xss",
    "backdoor",
    "ransomware",
    "mitm",
]

EVALUATION_IPS = {
    "192.168.1.30",
    "192.168.1.193",
    "192.168.1.195",
}

TRAIN_END_DATE = "2019-04-26"


def create_windows(df, scaler):
    X = []
    y = []
    metadata = []

    for (ip, date), group in df.groupby(
        ["src_ip", "date"], sort=False
    ):
        group = group.sort_values("minute")

        # Identificar sequências de minutos consecutivos.
        run_ids = (
            group["minute"].diff().ne(60)
        ).cumsum()

        for _, run in group.groupby(run_ids):
            if len(run) < WINDOW_SIZE:
                continue

            features = run[FEATURES].to_numpy(
                dtype=np.float64
            )

            transformed = scaler.transform(
                np.log1p(features)
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
                    "label": int(last["label"]),
                }

                for attack in ATTACK_TYPES:
                    row[attack] = int(
                        last[f"type_{attack}"] > 0
                    )

                metadata.append(row)

    if not X:
        raise ValueError(
            "Não foram criadas janelas."
        )

    return (
        np.stack(X),
        np.asarray(y, dtype=np.int64),
        pd.DataFrame(metadata),
    )


def main():
    df = pd.read_csv(
        INPUT_FILE,
        dtype={"src_ip": "string"}
    )

    required = (
        ["src_ip", "date", "minute", "label"]
        + FEATURES
        + [f"type_{attack}" for attack in ATTACK_TYPES]
    )

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Colunas em falta: {sorted(missing)}"
        )

    # Usar exclusivamente o período original de treino.
    df = df[
        df["date"] <= TRAIN_END_DATE
    ].copy()

    if df.empty:
        raise ValueError(
            "Sem dados no período de treino."
        )

    if df[FEATURES].isna().any().any():
        raise ValueError(
            "Existem características em falta."
        )

    if (df[FEATURES] < 0).any().any():
        raise ValueError(
            "Existem características negativas."
        )

    available_ips = set(df["src_ip"].dropna())

    if not EVALUATION_IPS.issubset(available_ips):
        raise ValueError(
            "Existem IPs de avaliação ausentes."
        )

    # Separar os IPs ANTES de ajustar o scaler.
    is_evaluation = df["src_ip"].isin(EVALUATION_IPS)

    train_df = df.loc[~is_evaluation].copy()
    evaluation_df = df.loc[is_evaluation].copy()

    train_ips = set(train_df["src_ip"].dropna())
    evaluation_ips = set(
        evaluation_df["src_ip"].dropna()
    )

    if train_ips & evaluation_ips:
        raise ValueError(
            "Existem IPs partilhados."
        )

    # Normalização exclusivamente com dados de treino.
    scaler = StandardScaler()

    scaler.fit(
        np.log1p(
            train_df[FEATURES].to_numpy(
                dtype=np.float64
            )
        )
    )

    train_X, train_y, train_metadata = (
        create_windows(train_df, scaler)
    )

    evaluation_X, evaluation_y, evaluation_metadata = (
        create_windows(evaluation_df, scaler)
    )

    # Confirmar o isolamento também nas janelas.
    train_window_ips = set(
        train_metadata["src_ip"]
    )
    evaluation_window_ips = set(
        evaluation_metadata["src_ip"]
    )

    if train_window_ips & evaluation_window_ips:
        raise ValueError(
            "IPs partilhados nas janelas."
        )

    for name, y in [
        ("train", train_y),
        ("evaluation", evaluation_y),
    ]:
        if set(np.unique(y)) != {0, 1}:
            raise ValueError(
                f"{name} não contém ambas as classes."
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_DIR / "train.npz",
        X=train_X,
        y=train_y,
    )

    np.savez_compressed(
        OUTPUT_DIR / "evaluation.npz",
        X=evaluation_X,
        y=evaluation_y,
    )

    train_metadata.to_csv(
        OUTPUT_DIR / "train_metadata.csv",
        index=False,
    )

    evaluation_metadata.to_csv(
        OUTPUT_DIR / "evaluation_metadata.csv",
        index=False,
    )

    scaler_info = {
        "features": FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "transform": "log1p_then_standard_scaler",
        "fitted_on": "train_ips_only",
        "window_size": WINDOW_SIZE,
    }

    with open(
        OUTPUT_DIR / "scaler.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            scaler_info,
            file,
            indent=2,
        )

    report = {
        "experiment": "V2-C_ip_generalization",
        "source": str(INPUT_FILE),
        "period_end": TRAIN_END_DATE,
        "window_size": WINDOW_SIZE,
        "features": FEATURES,
        "evaluation_ips": sorted(EVALUATION_IPS),
        "split_strategy": "directed_ip_holdout",
        "scaler_fitted_on": "train_ips_only",
        "splits": {},
    }

    for name, X, y, metadata in [
        ("train", train_X, train_y, train_metadata),
        (
            "evaluation",
            evaluation_X,
            evaluation_y,
            evaluation_metadata,
        ),
    ]:
        report["splits"][name] = {
            "windows": int(len(y)),
            "normal": int((y == 0).sum()),
            "attack": int((y == 1).sum()),
            "unique_ips": int(
                metadata["src_ip"].nunique()
            ),
            "attack_types": {
                attack: int(metadata[attack].sum())
                for attack in ATTACK_TYPES
            },
        }

        print(f"\n=== {name.upper()} ===")
        print("X:", X.shape)
        print("Normais:", int((y == 0).sum()))
        print("Ataques:", int((y == 1).sum()))
        print(
            "IPs:",
            metadata["src_ip"].nunique()
        )

    with open(
        OUTPUT_DIR / "preparation_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== V2-C CONCLUÍDO ===")
    print(
        "IPs partilhados:",
        len(train_window_ips & evaluation_window_ips)
    )
    print("Resultados:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
