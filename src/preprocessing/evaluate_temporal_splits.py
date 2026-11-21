
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data" / "processed"
    / "temporal_scanning" / "minute_features.csv"
)

OUTPUT = (
    ROOT / "results" / "tables"
    / "temporal_split_analysis.csv"
)

WINDOW_SIZE = 5
STRIDE = 1

CUTOFFS = [
    "2019-04-23 13:00",
    "2019-04-23 13:15",
    "2019-04-23 13:30",
    "2019-04-23 13:45",
    "2019-04-23 14:00",
    "2019-04-23 14:10",
    "2019-04-23 14:15",
]


def count_windows(data):
    normal = 0
    scanning = 0
    eligible_runs = 0

    for _, group in data.groupby("src_ip"):
        group = group.sort_values("minute").copy()

        run_id = (
            group["minute"].diff().ne(60)
        ).cumsum()

        for _, run in group.groupby(run_id):
            if len(run) < WINDOW_SIZE:
                continue

            eligible_runs += 1
            labels = run["label"].to_numpy()

            for start in range(
                0,
                len(run) - WINDOW_SIZE + 1,
                STRIDE
            ):
                last_label = labels[
                    start + WINDOW_SIZE - 1
                ]

                if last_label == 1:
                    scanning += 1
                else:
                    normal += 1

    return normal, scanning, eligible_runs


def main():
    df = pd.read_csv(INPUT)

    df["minute"] = pd.to_numeric(
        df["minute"], errors="raise"
    )

    df["label"] = pd.to_numeric(
        df["label"], errors="raise"
    )

    df["date"] = pd.to_datetime(
        df["minute"], unit="s", utc=True
    ).dt.strftime("%Y-%m-%d")

    day23 = df[
        df["date"] == "2019-04-23"
    ].copy()

    day24 = df[
        df["date"] == "2019-04-24"
    ].copy()

    test_normal, test_scanning, _ = (
        count_windows(day24)
    )

    print("\nTESTE — 24 DE ABRIL")
    print("Normal:", test_normal)
    print("Scanning:", test_scanning)

    results = []

    for cutoff_text in CUTOFFS:
        cutoff = int(
            pd.Timestamp(
                cutoff_text, tz="UTC"
            ).timestamp()
        )

        # Separar primeiro, criar janelas depois
        train = day23[
            day23["minute"] < cutoff
        ].copy()

        val = day23[
            day23["minute"] >= cutoff
        ].copy()

        tr_n, tr_a, tr_runs = count_windows(train)
        va_n, va_a, va_runs = count_windows(val)

        results.append({
            "cutoff_utc": cutoff_text,
            "train_normal": tr_n,
            "train_scanning": tr_a,
            "train_total": tr_n + tr_a,
            "val_normal": va_n,
            "val_scanning": va_a,
            "val_total": va_n + va_a,
            "train_runs": tr_runs,
            "val_runs": va_runs,
        })

    results_df = pd.DataFrame(results)

    print("\nCOMPARAÇÃO DE LIMITES TEMPORAIS")
    print(results_df.to_string(index=False))

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    results_df.to_csv(
        OUTPUT, index=False
    )

    print(f"\nResultados guardados em: {OUTPUT}")


if __name__ == "__main__":
    main()
