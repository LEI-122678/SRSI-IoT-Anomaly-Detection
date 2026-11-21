
import os

# Evitar excesso de threads em cada processo Ray.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from flwr.simulation import run_simulation

from src.federated.flower_client import (
    app as client_app,
)
from src.federated.flower_server import (
    app as server_app,
    MODEL_NAME,
    NUM_ROUNDS,
    NUM_CLIENTS,
)


def main():
    print("=== SIMULAÇÃO FLOWER V2-B ===")
    print("Arquitetura:", MODEL_NAME.upper())
    print("Clientes:", NUM_CLIENTS)
    print("Algoritmo: FedAvg")
    print("Máximo de rondas:", NUM_ROUNDS)
    print("Backend: Ray")
    print("Dispositivo: CPU")
    print("============================")

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
        backend_name="ray",
        backend_config={
            "init_args": {
                "num_cpus": 2,
                "include_dashboard": False,
            },
            "client_resources": {
                "num_cpus": 1,
                "num_gpus": 0,
            },
        },
        verbose_logging=False,
    )

    print("\nSIMULAÇÃO FLOWER CONCLUÍDA")


if __name__ == "__main__":
    main()
