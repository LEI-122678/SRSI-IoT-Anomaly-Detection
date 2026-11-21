
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "ton_iot_v2" / "minute_features.csv"
)

OUTPUT_DIR = ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 5

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

    groups = df.groupby(
        ["src_ip", "date"], sort=False
    )

    # Um novo segmento começa quando:
    # - muda o IP;
    # - muda o dia;
    # - existe uma lacuna temporal.
    new_run = groups["minute"].diff().ne(60)

    df["run_id"] = new_run.cumsum()
    df["position"] = df.groupby(
        "run_id"
    ).cumcount()

    # Cada linha elegível representa uma janela
    # terminada nesse minuto.
    windows = df.loc[
        df["position"] >= WINDOW_SIZE - 1
    ].copy()

    print("=== SEGMENTOS INDEPENDENTES POR ATAQUE ===")

    rows = []

    for attack in ATTACK_TYPES:
        selected = windows.loc[
            windows[f"type_{attack}"] > 0
        ]

        run_sizes = (
            selected.groupby(
                ["run_id", "src_ip", "date"]
            ).size().sort_values(ascending=False)
        )

        rows.append({
            "attack": attack,
            "windows": len(selected),
            "independent_runs": selected[
                "run_id"
            ].nunique(),
            "unique_ips": selected[
                "src_ip"
            ].nunique(),
            "days": selected["date"].nunique(),
            "largest_run_windows": (
                int(run_sizes.iloc[0])
                if len(run_sizes) else 0
            )
        })

        print(f"\n{attack.upper()}")
        print("Janelas:", len(selected))
        print(
            "Segmentos:",
            selected["run_id"].nunique()
        )
        print(
            "IPs:",
            selected["src_ip"].nunique()
        )
        print("Dias:", selected["date"].nunique())

        if not run_sizes.empty:
            print("Maiores segmentos:")
            print(run_sizes.head(5).to_string())

    summary = pd.DataFrame(rows)

    summary.to_csv(
        OUTPUT_DIR / "v2_attack_run_summary.csv",
        index=False
    )

    # Resumo por segmento, útil para
    # planear os conjuntos sem sobreposição.
    run_summary = (
        windows.groupby(
            ["run_id", "src_ip", "date"],
            as_index=False
        )
        .agg(
            windows=("label", "size"),
            normal_windows=(
                "label",
                lambda x: int((x == 0).sum())
            ),
            attack_windows=(
                "label",
                lambda x: int((x == 1).sum())
            ),
            **{
                attack: (
                    f"type_{attack}",
                    lambda x: int((x > 0).sum())
                )
                for attack in ATTACK_TYPES
            }
        )
    )

    run_summary.to_csv(
        OUTPUT_DIR / "v2_run_details.csv",
        index=False
    )

    print("\n=== RESUMO FINAL ===")
    print(summary.to_string(index=False))

    print("\nFicheiros guardados em:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
