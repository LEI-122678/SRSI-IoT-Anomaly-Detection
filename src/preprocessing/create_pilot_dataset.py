
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "ton_iot"
OUT_DIR = ROOT / "data" / "processed" / "pilot_scanning"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SAMPLE_PER_CLASS_DAY = 12000
CHUNK_SIZE = 100000

rng = np.random.default_rng(SEED)

DAYS = ["2019-04-23", "2019-04-24"]
TYPES = {"normal": 0, "scanning": 1}


def main():
    files = sorted(
        DATA_DIR.glob("Network_dataset_*.csv"),
        key=lambda p: int(p.stem.split("_")[-1])
    )

    if len(files) != 23:
        raise RuntimeError(
            f"Esperados 23 ficheiros, encontrados {len(files)}"
        )

    # Amostras mantidas separadamente por dia/classe
    samples = {
        (day, label): pd.DataFrame()
        for day in DAYS
        for label in (0, 1)
    }

    for i, file in enumerate(files, start=1):
        print(f"[{i}/23] {file.name}")

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):
            # Validar classe e tipo
            kind = chunk["type"].astype(str).str.lower().str.strip()
            label = pd.to_numeric(
                chunk["label"], errors="coerce"
            )

            valid = (
                ((kind == "normal") & (label == 0))
                | ((kind == "scanning") & (label == 1))
            )

            filtered = chunk.loc[valid].copy()

            if filtered.empty:
                continue

            filtered["ts"] = pd.to_numeric(
                filtered["ts"], errors="coerce"
            )

            filtered = filtered.dropna(subset=["ts"])

            filtered["_day"] = pd.to_datetime(
                filtered["ts"], unit="s", utc=True
            ).dt.strftime("%Y-%m-%d")

            filtered = filtered[
                filtered["_day"].isin(DAYS)
            ].copy()

            if filtered.empty:
                continue

            # Prioridade aleatória independente por registo
            filtered["_priority"] = rng.random(len(filtered))

            for day in DAYS:
                for class_label in (0, 1):
                    part = filtered[
                        (filtered["_day"] == day)
                        & (filtered["label"] == class_label)
                    ]

                    if part.empty:
                        continue

                    key = (day, class_label)

                    combined = pd.concat(
                        [samples[key], part],
                        ignore_index=True
                    )

                    # Manter os registos com menor prioridade
                    samples[key] = combined.nsmallest(
                        SAMPLE_PER_CLASS_DAY,
                        "_priority"
                    ).copy()

    # Dia 23: desenvolvimento
    development = pd.concat(
        [samples[("2019-04-23", 0)],
         samples[("2019-04-23", 1)]],
        ignore_index=True
    )

    # Dia 24: teste temporal
    test = pd.concat(
        [samples[("2019-04-24", 0)],
         samples[("2019-04-24", 1)]],
        ignore_index=True
    )

    if development.empty or test.empty:
        raise RuntimeError("Dados insuficientes.")

    development = development.sort_values(
        "ts", kind="stable"
    ).reset_index(drop=True)

    test = test.sort_values(
        "ts", kind="stable"
    ).reset_index(drop=True)

    # Divisão cronológica do dia 23
    cutoff = int(len(development) * 0.80)

    train = development.iloc[:cutoff].copy()
    val = development.iloc[cutoff:].copy()

    datasets = {
        "train": train,
        "validation": val,
        "test": test
    }

    report = {
        "seed": SEED,
        "max_samples_per_class_day": SAMPLE_PER_CLASS_DAY,
        "splits": {}
    }

    for name, df in datasets.items():
        if df["label"].nunique() != 2:
            raise RuntimeError(
                f"{name} não contém ambas as classes."
            )

        df = df.drop(columns=["_day", "_priority"])

        output = OUT_DIR / f"{name}.csv"
        df.to_csv(output, index=False)

        report["splits"][name] = {
            "rows": int(len(df)),
            "class_counts": {
                str(k): int(v)
                for k, v in df["label"]
                .value_counts().items()
            },
            "min_ts": float(df["ts"].min()),
            "max_ts": float(df["ts"].max())
        }

        print(f"\n{name.upper()}")
        print(f"Registos: {len(df):,}")
        print(df["label"].value_counts())
        print(f"Guardado em: {output}")

    with open(
        OUT_DIR / "split_report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("\nDataset piloto criado com sucesso!")


if __name__ == "__main__":
    main()
