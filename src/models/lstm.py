
import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size=6,
        hidden_size=32,
        num_layers=1,
        dropout=0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
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
        # x: (batch, minutos, características)
        output, _ = self.lstm(x)

        # Representação do último minuto
        last_output = output[:, -1, :]

        # Logit para classificação binária
        return self.classifier(
            last_output
        ).squeeze(-1)
