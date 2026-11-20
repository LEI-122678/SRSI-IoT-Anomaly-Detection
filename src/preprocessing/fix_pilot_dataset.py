
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "pilot_scanning"
OUT_DIR = ROOT / "data" / "processed" / "pilot_scanning"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    train = pd.read_csv(DATA_DIR / "train.csv", low_memory=False)
    val = pd.read_csv(DATA_DIR / "validation.csv", low_memory=False)
    test = pd.read_csv(DATA_DIR / "test.csv", low_memory=False)

    datasets = {
        "train": train,
        "validation": val,
        "test": test
    }

    # 1. Verificar diferenças de colunas
    reference = set(train.columns)

    for name, df in datasets.items():
        extra = sorted(set(df.columns) - reference)
        missing = sorted(reference - set(df.columns))

        print(f"\n{name.upper()}")
        print("Colunas extra:", extra)
        print("Colunas em falta:", missing)

    # Remover o identificador de ligação do conjunto piloto.
    # O campo uid não existe em todas as partições.
    for name, df in datasets.items():
        if "uid" in df.columns:
            datasets[name] = df.drop(columns=["uid"])
            print(f"{name}: coluna uid removida")

    # Não remover colunas sem primeiro as analisar
    schemas = [set(df.columns) for df in datasets.values()]

    if len({frozenset(s) for s in schemas}) != 1:
        raise RuntimeError(
            "Os esquemas são diferentes. "
            "Identifica primeiro a coluna extra."
        )

    # 2. Remover duplicados exatos
    for name in datasets:
        before = len(datasets[name])

        datasets[name] = (
            datasets[name]
            .drop_duplicates()
            .sort_values("ts", kind="stable")
            .reset_index(drop=True)
        )

        removed = before - len(datasets[name])
        print(f"{name}: {removed} duplicados removidos")

    # 3. Corrigir a fronteira temporal
    train = datasets["train"]
    val = datasets["validation"]
    test = datasets["test"]

    boundary = train["ts"].max()

    # Manter na validação apenas timestamps
    # posteriores ao último timestamp do treino
    val = val[val["ts"] > boundary].copy()

    datasets["validation"] = val

    # 4. Validar a ordem temporal
    if not train["ts"].max() < val["ts"].min():
        raise RuntimeError("Sobreposição treino/validação")

    if not val["ts"].max() < test["ts"].min():
        raise RuntimeError("Sobreposição validação/teste")

    # 5. Verificar classes e guardar
    report = {}

    for name, df in datasets.items():
        if df.empty:
            raise RuntimeError(f"{name} ficou vazio")

        if df["label"].nunique() != 2:
            raise RuntimeError(
                f"{name} não contém ambas as classes"
            )

        output = OUT_DIR / f"{name}.csv"
        df.to_csv(output, index=False)

        report[name] = {
            "rows": len(df),
            "labels": {
                str(k): int(v)
                for k, v in df["label"]
                .value_counts().items()
            },
            "min_ts": float(df["ts"].min()),
            "max_ts": float(df["ts"].max())
        }

        print(f"\n{name.upper()}: {len(df):,} registos")
        print(df["label"].value_counts())

    with open(
        OUT_DIR / "validation_report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("\nDataset v2 criado com sucesso!")


if __name__ == "__main__":
    main()
