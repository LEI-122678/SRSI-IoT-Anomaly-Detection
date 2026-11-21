
from torch import nn


class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_size=6,
        hidden_size=32,
        num_layers=1,
        dropout=0.2
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        output, _ = self.gru(x)

        last_output = output[:, -1, :]

        return self.classifier(
            last_output
        ).squeeze(-1)
