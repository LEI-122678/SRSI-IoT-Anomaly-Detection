
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "ton_iot"

PROCESSED_FILE = (
    ROOT / "data" / "processed"
    / "temporal_scanning" / "minute_features.csv"
)

TARGET_IP = "192.168.1.152"

TARGET_MINUTE = int(
    pd.Timestamp(
        "2019-04-24T18:37:00Z"
    ).timestamp()
)

CHUNK_SIZE = 100_000

NEEDED_COLUMNS = [
    "ts",
    "src_ip",
    "label",
    "type",
    "src_pkts",
    "src_bytes",
]


def main():
    total_connections = 0
    total_packets = 0.0
    total_bytes = 0.0
    matching_files = {}
    matching_types = {}
    samples = []

    files = sorted(
        RAW_DIR.glob("Network_dataset_*.csv"),
        key=lambda p: int(p.stem.split("_")[-1])
    )

    for file in files:
        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            usecols=lambda col: col in NEEDED_COLUMNS,
            low_memory=False
        ):
            chunk["ts"] = pd.to_numeric(
                chunk["ts"], errors="coerce"
            )

            minutes = (
                chunk["ts"].floordiv(60).mul(60)
            )

            mask = (
                (chunk["src_ip"] == TARGET_IP)
                & (minutes == TARGET_MINUTE)
            )

            matches = chunk.loc[mask].copy()

            if matches.empty:
                continue

            # Reproduzir os filtros do piloto
            kinds = (
                matches["type"].astype(str)
                .str.strip().str.lower()
            )

            labels = pd.to_numeric(
                matches["label"], errors="coerce"
            )

            valid = (
                ((kinds == "normal") & (labels == 0))
                | ((kinds == "scanning") & (labels == 1))
            )

            matches = matches.loc[valid].copy()

            if matches.empty:
                continue

            matches["src_pkts"] = pd.to_numeric(
                matches["src_pkts"], errors="coerce"
            ).fillna(0)

            matches["src_bytes"] = pd.to_numeric(
                matches["src_bytes"], errors="coerce"
            ).fillna(0)

            total_connections += len(matches)
            total_packets += matches["src_pkts"].sum()
            total_bytes += matches["src_bytes"].sum()

            matching_files[file.name] = (
                matching_files.get(file.name, 0)
                + len(matches)
            )

            for attack_type, count in (
                matches["type"].value_counts().items()
            ):
                matching_types[str(attack_type)] = (
                    matching_types.get(str(attack_type), 0)
                    + int(count)
                )

            if len(samples) < 10:
                samples.extend(
                    matches[
                        [
                            "ts", "src_ip", "label",
                            "type", "src_pkts", "src_bytes"
                        ]
                    ].head(
                        10 - len(samples)
                    ).to_dict("records")
                )

        print(f"Verificado: {file.name}")

    print("\n=== RESULTADOS NOS CSV ORIGINAIS ===")

    print("IP:", TARGET_IP)
    print("Minuto:", TARGET_MINUTE)
    print("Ligações:", total_connections)
    print("Soma src_pkts:", total_packets)
    print("Soma src_bytes:", total_bytes)

    print("\nTipos encontrados:")
    print(matching_types)

    print("\nFicheiros com correspondências:")
    print(matching_files)

    print("\nExemplos de registos:")
    print(pd.DataFrame(samples).to_string(index=False))

    processed = pd.read_csv(PROCESSED_FILE)

    row = processed.loc[
        (processed["src_ip"] == TARGET_IP)
        & (processed["minute"] == TARGET_MINUTE)
    ]

    print("\n=== RESULTADO PROCESSADO ===")

    if row.empty:
        print("Intervalo não encontrado.")
        return

    print(
        row[
            [
                "connections",
                "src_pkts_sum",
                "src_bytes_sum",
                "label"
            ]
        ].to_string(index=False)
    )

    p = row.iloc[0]

    consistent = (
        int(p["connections"]) == total_connections
        and abs(
            float(p["src_pkts_sum"]) - total_packets
        ) < 0.01
        and abs(
            float(p["src_bytes_sum"]) - total_bytes
        ) < 0.01
    )

    print("\nAgregação consistente:", consistent)


if __name__ == "__main__":
    main()
