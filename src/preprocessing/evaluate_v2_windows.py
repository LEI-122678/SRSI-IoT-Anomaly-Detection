
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "ton_iot_v2" / "minute_features.csv"
)

OUTPUT_DIR = ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZES = [5, 10, 20]

ATTACK_TYPES = [
    "scanning", "dos", "ddos", "injection",
    "password", "xss", "backdoor",
    "ransomware", "mitm"
]


def main():
    df = pd.read_csv(INPUT_FILE)

    df = df.sort_values(
        ["src_ip", "date", "minute"]
    ).reset_index(drop=True)

    # Uma sequência não pode atravessar um IP,
    # um dia ou uma lacuna superior a 1 minuto.
    groups = df.groupby(
        ["src_ip", "date"], sort=False
    )

    new_run = groups["minute"].diff().ne(60)

    df["run_id"] = new_run.cumsum()
    df["position"] = (
        df.groupby("run_id").cumcount()
    )

    print("=== AUDITORIA DAS JANELAS V2 ===")

    print("\nTotal de IPs:", df["src_ip"].nunique())

    ip_counts = df.groupby("src_ip").size()

    print(
        "IPs com apenas um registo:",
        int((ip_counts == 1).sum())
    )

    print(
        "IPs com pelo menos 5 registos:",
        int((ip_counts >= 5).sum())
    )

    records = []
    summary_rows = []

    for size in WINDOW_SIZES:
        # Cada linha elegível representa uma janela
        # que termina nesse minuto.
        windows = df[
            df["position"] >= size - 1
        ].copy()

        summary_rows.append({
            "window_size": size,
            "windows": len(windows),
            "normal": int(
                (windows["label"] == 0).sum()
            ),
            "attack": int(
                (windows["label"] == 1).sum()
            ),
            "unique_ips": windows[
                "src_ip"
            ].nunique(),
            "independent_runs": windows[
                "run_id"
            ].nunique()
        })

        for date, group in windows.groupby("date"):
            row = {
                "window_size": size,
                "date": date,
                "windows": len(group),
                "normal": int(
                    (group["label"] == 0).sum()
                ),
                "attack": int(
                    (group["label"] == 1).sum()
                ),
                "unique_ips": group[
                    "src_ip"
                ].nunique()
            }

            for kind in ATTACK_TYPES:
                row[kind] = int(
                    (group["type_" + kind] > 0).sum()
                )

            records.append(row)

    summary = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(records)

    summary.to_csv(
        OUTPUT_DIR / "v2_window_summary.csv",
        index=False
    )

    detail.to_csv(
        OUTPUT_DIR / "v2_windows_by_day.csv",
        index=False
    )

    print("\n1. RESUMO POR TAMANHO")
    print(summary.to_string(index=False))

    print("\n2. JANELAS DE 5 MINUTOS POR DIA")

    print(
        detail.loc[
            detail["window_size"] == 5,
            [
                "date", "windows",
                "normal", "attack",
                "unique_ips"
            ]
        ].to_string(index=False)
    )

    print("\n3. TIPOS DE ATAQUE NAS JANELAS DE 5 MIN")

    five = detail[
        detail["window_size"] == 5
    ]

    for kind in ATTACK_TYPES:
        print(
            f"{kind}: {int(five[kind].sum())}"
        )

    print("\nResultados guardados em:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
