
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.preprocessing.prepare_v2_ip_generalization import (
    INPUT_FILE,
    FEATURES,
    EVALUATION_IPS,
    TRAIN_END_DATE,
    WINDOW_SIZE,
    ATTACK_TYPES,
    create_windows,
)


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    ROOT / "data" / "processed"
    / "ton_iot_v2" / "ip_generalization"
    / "internal_validation"
)

INTERNAL_VALIDATION_IPS = {
    "192.168.1.34",
    "192.168.1.39",
    "192.168.1.32",
    "192.168.1.36",
    "192.168.1.79",
    "192.168.1.180",
    "fe80::a691:b1ff:fe1e:5790",
}


def describe_split(name, X, y, metadata):
    attack_counts = {
        attack: int(metadata[attack].sum())
        for attack in ATTACK_TYPES
    }

    report = {
        "windows": int(len(y)),
        "normal": int((y == 0).sum()),
        "attack": int((y == 1).sum()),
        "unique_ips": int(metadata["src_ip"].nunique()),
        "attack_types": attack_counts,
    }

    print(f"\n=== {name.upper()} ===")
    print("X:", X.shape)
    print("Normal:", report["normal"])
    print("Ataques:", report["attack"])
    print("IPs:", report["unique_ips"])

    print("Tipos de ataque:")
    for attack, count in attack_counts.items():
        if count:
            print(f"  {attack}: {count}")

    return report


def main():
    print("=== PREPARACAO V2-C INTERNA ===")

    if INTERNAL_VALIDATION_IPS & EVALUATION_IPS:
        raise ValueError(
            "IPs de validacao interna sobrepoem-se "
            "aos IPs reservados."
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"src_ip": "string"},
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

    if df["src_ip"].isna().any():
        raise ValueError("Existem IPs em falta.")

    # Apenas o periodo de treino original V2-B.
    df = df.loc[
        df["date"] <= TRAIN_END_DATE
    ].copy()

    if df.empty:
        raise ValueError("Periodo de treino vazio.")

    if df.duplicated(["src_ip", "minute"]).any():
        raise ValueError(
            "Existem minutos duplicados por IP."
        )

    if df[FEATURES].isna().any().any():
        raise ValueError(
            "Existem caracteristicas em falta."
        )

    if not np.isfinite(
        df[FEATURES].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "Existem valores nao finitos."
        )

    if (df[FEATURES] < 0).any().any():
        raise ValueError(
            "Existem caracteristicas negativas."
        )

    # Os tres IPs reservados nunca entram
    # no desenvolvimento V2-C.
    reserved = df["src_ip"].isin(EVALUATION_IPS)

    development_df = df.loc[~reserved].copy()

    # Separar a validacao interna por IP.
    is_validation = development_df["src_ip"].isin(
        INTERNAL_VALIDATION_IPS
    )

    train_df = development_df.loc[
        ~is_validation
    ].copy()

    validation_df = development_df.loc[
        is_validation
    ].copy()

    if train_df.empty or validation_df.empty:
        raise ValueError(
            "Treino ou validacao interna vazios."
        )

    train_ips = set(train_df["src_ip"])
    val_ips = set(validation_df["src_ip"])

    if val_ips != INTERNAL_VALIDATION_IPS:
        raise ValueError(
            "Faltam IPs de validacao interna."
        )

    if train_ips & val_ips:
        raise ValueError(
            "IPs partilhados entre treino e validacao."
        )

    if (train_ips | val_ips) & EVALUATION_IPS:
        raise ValueError(
            "IPs reservados presentes no desenvolvimento."
        )

    # Scaler ajustado APENAS com treino interno.
    scaler = StandardScaler()

    scaler.fit(
        np.log1p(
            train_df[FEATURES].to_numpy(
                dtype=np.float64
            )
        )
    )

    train_X, train_y, train_metadata = create_windows(
        train_df, scaler
    )

    val_X, val_y, val_metadata = create_windows(
        validation_df, scaler
    )

    # Verificacoes das janelas efetivas.
    train_window_ips = set(train_metadata["src_ip"])
    val_window_ips = set(val_metadata["src_ip"])

    if train_window_ips & val_window_ips:
        raise ValueError(
            "IPs partilhados nas janelas."
        )

    if val_window_ips != INTERNAL_VALIDATION_IPS:
        raise ValueError(
            "Faltam IPs com janelas na validacao."
        )

    if (train_window_ips | val_window_ips) & EVALUATION_IPS:
        raise ValueError(
            "IPs reservados encontrados nas janelas."
        )

    for name, X, y, metadata in [
        ("internal_train", train_X, train_y, train_metadata),
        ("internal_validation", val_X, val_y, val_metadata),
    ]:
        if X.shape[1:] != (WINDOW_SIZE, len(FEATURES)):
            raise ValueError(
                f"Dimensoes incorretas em {name}."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                f"Valores nao finitos em {name}."
            )

        if set(np.unique(y)) != {0, 1}:
            raise ValueError(
                f"{name} nao contem ambas as classes."
            )

        if not np.array_equal(
            y, metadata["label"].to_numpy()
        ):
            raise ValueError(
                f"Rotulos inconsistentes em {name}."
            )

    train_report = describe_split(
        "internal_train",
        train_X,
        train_y,
        train_metadata,
    )

    validation_report = describe_split(
        "internal_validation",
        val_X,
        val_y,
        val_metadata,
    )

    report = {
        "experiment": "V2-C_internal_validation",
        "source": str(INPUT_FILE),
        "period_end": TRAIN_END_DATE,
        "window_size": WINDOW_SIZE,
        "features": FEATURES,
        "split_strategy": "directed_ip_holdout",
        "internal_validation_ips": sorted(
            INTERNAL_VALIDATION_IPS
        ),
        "reserved_evaluation_ips": sorted(
            EVALUATION_IPS
        ),
        "scaler_fitted_on": "internal_train_ips_only",
        "internal_train": train_report,
        "internal_validation": validation_report,
        "integrity": {
            "shared_ips": 0,
            "reserved_ips_used": 0,
        },
        "limitations": [
            "Directed exploratory split, not random.",
            "Validation attack distribution is IP-dependent.",
            "Overlapping windows are not independent.",
        ],
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_DIR / "internal_train.npz",
        X=train_X,
        y=train_y,
    )

    np.savez_compressed(
        OUTPUT_DIR / "internal_validation.npz",
        X=val_X,
        y=val_y,
    )

    train_metadata.to_csv(
        OUTPUT_DIR / "internal_train_metadata.csv",
        index=False,
    )

    val_metadata.to_csv(
        OUTPUT_DIR / "internal_validation_metadata.csv",
        index=False,
    )

    scaler_info = {
        "features": FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "window_size": WINDOW_SIZE,
        "transform": "log1p_then_standard_scaler",
        "fitted_on": "internal_train_ips_only",
        "fit_rows": int(len(train_df)),
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

    print("\n=== VERIFICACOES ===")
    print(
        "IPs partilhados:",
        len(train_window_ips & val_window_ips)
    )
    print("IPs reservados utilizados: 0")
    print("Scaler: apenas treino interno")
    print("\nPreparacao V2-C interna concluida.")
    print("Resultados:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
