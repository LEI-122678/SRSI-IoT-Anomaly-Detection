
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data" / "processed"
    / "temporal_scanning" / "minute_features.csv"
)

WINDOW_SIZES = [5, 10, 20, 30]
STRIDE = 1


def main():
    df = pd.read_csv(INPUT)

    df["minute"] = pd.to_numeric(df["minute"])
    df["label"] = pd.to_numeric(df["label"])

    df["date"] = pd.to_datetime(
        df["minute"], unit="s", utc=True
    ).dt.strftime("%Y-%m-%d")

    # Separar dias e IPs para não atravessar capturas
    results = []

    for window_size in WINDOW_SIZES:
        for date, day_df in df.groupby("date"):
            normal = 0
            attacks = 0
            total_runs = 0
            eligible_runs = 0

            for _, group in day_df.groupby("src_ip"):
                group = group.sort_values("minute").copy()

                # Quebra sempre que não há
                # continuidade de 60 segundos
                run_id = (
                    group["minute"].diff().ne(60)
                ).cumsum()

                for _, run in group.groupby(run_id):
                    total_runs += 1

                    if len(run) < window_size:
                        continue

                    eligible_runs += 1
                    labels = run["label"].to_numpy()

                    for start in range(
                        0,
                        len(run) - window_size + 1,
                        STRIDE
                    ):
                        end = start + window_size

                        # Label do último minuto
                        if labels[end - 1] == 1:
                            attacks += 1
                        else:
                            normal += 1

            results.append({
                "window_size": window_size,
                "date": date,
                "normal": normal,
                "scanning": attacks,
                "total": normal + attacks,
                "eligible_runs": eligible_runs,
                "all_runs": total_runs
            })

    results_df = pd.DataFrame(results)

    print("\nCOMPARAÇÃO DOS TAMANHOS DE JANELA")
    print(results_df.to_string(index=False))

    output_dir = ROOT / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = output_dir / "window_size_analysis.csv"

    results_df.to_csv(output, index=False)

    print(f"\nResultados guardados em: {output}")


if __name__ == "__main__":
    main()
