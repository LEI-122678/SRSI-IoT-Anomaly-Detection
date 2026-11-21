import torch

from flwr.app import ArrayRecord, MetricRecord, RecordDict
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords


def test_flower_fedavg_weighted_aggregation():
    states = [
        {"weight": torch.tensor([1.0, 3.0], dtype=torch.float32)},
        {"weight": torch.tensor([5.0, 7.0], dtype=torch.float32)},
        {"weight": torch.tensor([9.0, 11.0], dtype=torch.float32)},
    ]

    counts = [10, 30, 60]

    records = [
        RecordDict({
            "arrays": ArrayRecord(state),
            "metrics": MetricRecord({"num-examples": count}),
        })
        for state, count in zip(states, counts)
    ]

    result = aggregate_arrayrecords(
        records,
        "num-examples",
    ).to_torch_state_dict()

    total = sum(counts)

    expected = sum(
        state["weight"].to(torch.float64) * count
        for state, count in zip(states, counts)
    ) / total

    assert set(result) == {"weight"}
    assert torch.allclose(
        result["weight"].to(torch.float64),
        expected,
        rtol=0,
        atol=1e-6,
    )
