
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "ton_iot_v2" / "minute_features.csv"
)

OUTPUT_DIR = ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACK_TYPES = [
    "scanning", "dos", "ddos", "injection",
    "password", "xss", "backdoor",
    "ransomware", "mitm"
]

WINDOW_SIZE = 5


def main():
    df = pd.read_csv(INPUT_FILE)

    df = df.sort_values(
        ["src_ip", "date", "minute"]
    ).reset_index(drop=True)

    groups = df.groupby(["src_ip", "date"], sort=False)

    new_run = groups["minute"].diff().ne(60)
    df["run_id"] = new_run.cumsum()

    df["position"] = (
        df.groupby("run_id").cumcount()
    )

    # Apenas minutos que podem terminar
    # uma janela consecutiva de 5 minutos
    windows = df.loc[
        df["position"] >= WINDOW_SIZE - 1
    ].copy()

    windows["hour"] = pd.to_datetime(
        windows["minute"], unit="s", utc=True
    ).dt.floor("h")

    attack_columns = [
        f"type_{attack}" for attack in ATTACK_TYPES
    ]

    for column in attack_columns:
        windows[column] = (
            windows[column] > 0
        ).astype(int)

    hourly = windows.groupby(
        "hour", as_index=False
    ).agg({
        "label": ["size", "sum"],
        **{
            col: "sum"
            for col in attack_columns
        }
    })

    hourly.columns = [
        "hour", "windows", "attack_windows",
        *ATTACK_TYPES
    ]

    hourly["normal_windows"] = (
        hourly["windows"] - hourly["attack_windows"]
    )

    hourly.to_csv(
        OUTPUT_DIR / "v2_attack_hours.csv",
        index=False
    )

    print("=== DISTRIBUIÇÃO HORÁRIA DOS ATAQUES ===")

    rows = []

    for attack in ATTACK_TYPES:
        active = hourly.loc[
            hourly[attack] > 0,
            ["hour", attack]
        ].copy()

        total = int(active[attack].sum())

        if active.empty:
            continue

        peak = active.loc[
            active[attack].idxmax()
        ]

        rows.append({
            "attack": attack,
            "active_hours": len(active),
            "total_windows": total,
            "peak_hour": str(peak["hour"]),
            "peak_windows": int(peak[attack]),
            "peak_share_pct": round(
                100 * peak[attack] / total, 2
            )
        })

    summary = pd.DataFrame(rows)

    print(summary.to_string(index=False))

    summary.to_csv(
        OUTPUT_DIR / "v2_attack_hour_summary.csv",
        index=False
    )

    print("\nResultados guardados em:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
