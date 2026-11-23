
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RESULTS = ROOT / "results" / "demonstration"

BENIGN = (
    RESULTS / "flower_20rounds_benign"
    / "validation_history.csv"
)

MALICIOUS = (
    RESULTS / "flower_20rounds_malicious_neg10_verified"
    / "validation_history.csv"
)

OUTPUT = RESULTS / "comparison_20rounds"

METRICS = {
    "f1": ("F1-score", "f1_comparison.png"),
    "recall": ("Recall", "recall_comparison.png"),
    "fpr": ("False Positive Rate", "fpr_comparison.png"),
    "loss": ("Validation Loss", "loss_comparison.png"),
}


def main():
    benign = pd.read_csv(BENIGN)
    malicious = pd.read_csv(MALICIOUS)

    required = {"round", *METRICS.keys()}

    for name, df in [
        ("Benigno", benign),
        ("Malicioso", malicious),
    ]:
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name}: colunas em falta: {missing}"
            )

        if df["round"].tolist() != list(range(21)):
            raise ValueError(
                f"{name}: esperadas rondas 0 a 20."
            )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    for metric, (title, filename) in METRICS.items():
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(
            benign["round"],
            benign[metric],
            label="Benigno",
            linewidth=2,
            marker="o",
            markersize=3,
        )

        ax.plot(
            malicious["round"],
            malicious[metric],
            label="Malicioso (-10)",
            linewidth=2,
            marker="s",
            markersize=3,
        )

        ax.set_title(
            f"{title}: benigno vs. cliente malicioso"
        )
        ax.set_xlabel("Ronda FedAvg")
        ax.set_ylabel(title)
        ax.set_xticks(range(0, 21, 2))
        ax.grid(alpha=0.3)
        ax.legend()

        if metric in {"f1", "recall", "fpr"}:
            ax.set_ylim(0, 1)

        fig.tight_layout()
        fig.savefig(
            OUTPUT / filename,
            dpi=200,
            bbox_inches="tight"
        )
        plt.close(fig)

        print(f"[OK] {filename}")

    print(f"\nGraficos guardados em: {OUTPUT}")


if __name__ == "__main__":
    main()
