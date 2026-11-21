
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from src.models.lstm import LSTMClassifier
from src.models.gru import GRUClassifier
from src.models.transformer import TransformerClassifier


ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT / "data" / "processed" / "ton_iot_v2"
    / "temporal_generalization" / "test.npz"
)

MODEL_DIR = ROOT / "results" / "models" / "v2b"

OUTPUT_FILE = (
    ROOT / "results" / "tables"
    / "v2b_efficiency_comparison.csv"
)

MODELS = {
    "LSTM": LSTMClassifier,
    "GRU": GRUClassifier,
    "Transformer": TransformerClassifier
}

SEED = 42
BATCH_SIZE = 32
WARMUP = 5
REPEATS = 15


def main():
    torch.set_num_threads(1)

    with np.load(DATA_FILE) as data:
        X = torch.from_numpy(
            data["X"].astype(np.float32)
        )

    batches = list(
        torch.split(X, BATCH_SIZE)
    )

    results = []

    print("=== BENCHMARK V2-B — CPU ===")
    print("Janelas:", len(X))
    print("Batch size:", BATCH_SIZE)
    print("Repetições:", REPEATS)

    for name, model_class in MODELS.items():
        path = MODEL_DIR / (
            f"{name.lower()}_seed{SEED}_best.pt"
        )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )

        model = model_class(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"]
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.eval()

        params = sum(
            p.numel() for p in model.parameters()
        )

        checkpoint_kb = path.stat().st_size / 1024

        # Aquecimento para evitar incluir custos
        # de inicialização nas medições.
        with torch.inference_mode():
            for _ in range(WARMUP):
                for batch in batches:
                    _ = model(batch)

        durations = []

        with torch.inference_mode():
            for _ in range(REPEATS):
                start = time.perf_counter()

                for batch in batches:
                    _ = model(batch)

                elapsed = time.perf_counter() - start
                durations.append(elapsed)

        durations = np.asarray(durations)

        mean_ms_per_window = (
            durations.mean() * 1000 / len(X)
        )

        throughput = len(X) / durations.mean()

        row = {
            "model": name,
            "seed": SEED,
            "parameters": params,
            "checkpoint_kb": round(
                checkpoint_kb, 2
            ),
            "latency_ms_per_window": round(
                mean_ms_per_window, 6
            ),
            "throughput_windows_per_second": round(
                throughput, 2
            ),
            "total_inference_seconds_mean": round(
                float(durations.mean()), 4
            ),
            "total_inference_seconds_std": round(
                float(durations.std(ddof=1)), 4
            )
        }

        results.append(row)

        print(f"\n{name}")
        for key, value in row.items():
            print(f"{key}: {value}")

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    pd.DataFrame(results).to_csv(
        OUTPUT_FILE, index=False
    )

    print("\nResultados guardados em:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
