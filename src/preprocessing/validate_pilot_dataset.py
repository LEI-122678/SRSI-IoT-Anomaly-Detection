
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "processed" / "pilot_scanning"
)


def main():
    datasets = {}

    for name in ["train", "validation", "test"]:
        path = DATA_DIR / f"{name}.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(
            path,
            low_memory=False,
            na_values=["-", ""]
        )

        datasets[name] = df

        print("\n" + "=" * 60)
        print(name.upper())
        print("=" * 60)

        print(f"Registos: {len(df):,}")
        print(f"Colunas: {len(df.columns)}")

        print("\nClasses:")
        print(df["label"].value_counts(dropna=False))

        print("\nTipos:")
        print(df["type"].value_counts(dropna=False))

        print("\nIntervalo temporal:")
        print(pd.to_datetime(
            df["ts"].min(), unit="s", utc=True
        ))
        print(pd.to_datetime(
            df["ts"].max(), unit="s", utc=True
        ))

        print("\nDuplicados internos exatos:")
        print(df.duplicated().sum())

        print("\nTimestamps duplicados:")
        print(df["ts"].duplicated().sum())

        print("\nOrdem temporal crescente:")
        print(df["ts"].is_monotonic_increasing)

        # Verificar correspondência entre label e type
        valid_labels = (
            ((df["label"] == 0) &
             (df["type"] == "normal"))
            |
            ((df["label"] == 1) &
             (df["type"] == "scanning"))
        )

        print("\nRegistos com label/type incoerentes:")
        print((~valid_labels).sum())

    train = datasets["train"]
    val = datasets["validation"]
    test = datasets["test"]

    print("\n" + "=" * 60)
    print("VERIFICAÇÃO ENTRE PARTIÇÕES")
    print("=" * 60)

    for name_a, name_b in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test")
    ]:
        a = datasets[name_a]
        b = datasets[name_b]

        # Comparar registos completos
        common_columns = list(a.columns)

        a_unique = a[common_columns].drop_duplicates()
        b_unique = b[common_columns].drop_duplicates()

        overlap = a_unique.merge(
            b_unique,
            how="inner",
            on=common_columns
        )

        print(
            f"\n{name_a} vs {name_b}:"
        )

        print(
            f"Registos exatos em comum: "
            f"{len(overlap):,}"
        )

        common_ts = (
            set(a["ts"]) & set(b["ts"])
        )

        print(
            f"Timestamps em comum: "
            f"{len(common_ts):,}"
        )

    print("\nSEPARAÇÃO TEMPORAL:")

    print(
        "Train < Validation:",
        train["ts"].max() < val["ts"].min()
    )

    print(
        "Validation < Test:",
        val["ts"].max() < test["ts"].min()
    )


if __name__ == "__main__":
    main()