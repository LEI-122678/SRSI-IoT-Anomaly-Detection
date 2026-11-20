
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "ton_iot"
)

CHUNK_SIZE = 100_000


def main():
    files = sorted(DATA_DIR.rglob("*.csv"))

    if not files:
        print("Nenhum CSV encontrado!")
        return

    print(f"Ficheiros encontrados: {len(files)}")

    for file in files:
        print("\n" + "=" * 65)
        print(f"FICHEIRO: {file.name}")
        print("=" * 65)

        total_rows = 0
        labels = Counter()
        attack_types = Counter()

        min_ts = None
        max_ts = None

        missing_counts = Counter()

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            na_values=["-", ""],
            keep_default_na=True,
        ):
            total_rows += len(chunk)

            if "label" in chunk.columns:
                labels.update(
                    chunk["label"].value_counts(
                        dropna=False
                    ).to_dict()
                )

            if "type" in chunk.columns:
                attack_types.update(
                    chunk["type"].value_counts(
                        dropna=False
                    ).to_dict()
                )

            if "ts" in chunk.columns:
                ts = pd.to_numeric(
                    chunk["ts"], errors="coerce"
                ).dropna()

                if not ts.empty:
                    chunk_min = ts.min()
                    chunk_max = ts.max()

                    min_ts = (
                        chunk_min if min_ts is None
                        else min(min_ts, chunk_min)
                    )

                    max_ts = (
                        chunk_max if max_ts is None
                        else max(max_ts, chunk_max)
                    )

            missing_counts.update(
                chunk.isna().sum().to_dict()
            )

        print(f"\nTOTAL DE REGISTOS: {total_rows:,}")

        print("\nDISTRIBUIÇÃO DAS LABELS:")
        for label, count in labels.items():
            percentage = 100 * count / total_rows
            print(
                f"  {label}: {count:,} "
                f"({percentage:.2f}%)"
            )

        print("\nTIPOS DE ATAQUE:")
        for attack, count in attack_types.most_common():
            percentage = 100 * count / total_rows
            print(
                f"  {attack}: {count:,} "
                f"({percentage:.2f}%)"
            )

        print("\nINTERVALO TEMPORAL:")
        if min_ts is not None:
            start = datetime.fromtimestamp(
                min_ts, timezone.utc
            )
            end = datetime.fromtimestamp(
                max_ts, timezone.utc
            )

            print(f"  Início: {start}")
            print(f"  Fim:    {end}")
            print(f"  Duração: {end - start}")

        print("\nCOLUNAS COM VALORES EM FALTA:")
        for col, count in missing_counts.most_common():
            if count > 0:
                percentage = 100 * count / total_rows
                print(
                    f"  {col}: {count:,} "
                    f"({percentage:.2f}%)"
                )


if __name__ == "__main__":
    main()
