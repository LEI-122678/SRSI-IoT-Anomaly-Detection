
import torch
from flwr.app import ArrayRecord

from src.federated.train_fedavg_gru import (
    build_model,
    SEED,
)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    model = build_model()

    original = model.state_dict()

    # Simular o envio dos pesos pelo Flower.
    record = ArrayRecord(original)

    # Simular a receção dos pesos pelo cliente.
    recovered = record.to_torch_state_dict()

    if original.keys() != recovered.keys():
        raise ValueError(
            "As chaves dos parâmetros são diferentes."
        )

    differences = {}

    for key in original:
        a = original[key]
        b = recovered[key]

        if a.shape != b.shape:
            raise ValueError(
                f"Shape diferente em {key}"
            )

        if a.dtype != b.dtype:
            raise ValueError(
                f"Tipo diferente em {key}"
            )

        differences[key] = (
            a - b
        ).abs().max().item()

    max_difference = max(differences.values())

    print("=== SERIALIZAÇÃO FLOWER ===")
    print("Parâmetros:", len(original))
    print(
        "Diferença máxima:",
        max_difference
    )

    if all(
        torch.equal(original[key], recovered[key])
        for key in original
    ):
        print("RESULTADO: PESOS IDÊNTICOS")
    else:
        print("RESULTADO: DIFERENÇAS DETETADAS")
        print(differences)


if __name__ == "__main__":
    main()
