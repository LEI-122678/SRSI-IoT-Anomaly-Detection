import pytest
import torch

from src.models.gru import GRUClassifier
from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerClassifier


@pytest.mark.parametrize(
    "model_class,extra_args",
    [
        (GRUClassifier, {}),
        (LSTMClassifier, {}),
        (TransformerClassifier, {"nhead": 4, "max_seq_len": 5}),
    ],
)
def test_model_forward_and_backward(model_class, extra_args):
    torch.manual_seed(42)

    model = model_class(
        input_size=6,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        **extra_args,
    )

    x = torch.randn(4, 5, 6)
    output = model(x)

    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 4
    assert torch.isfinite(output).all()

    loss = output.sum()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert gradients
    assert all(grad is not None for grad in gradients)
    assert all(torch.isfinite(grad).all() for grad in gradients)
