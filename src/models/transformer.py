
import torch
from torch import nn


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size=6,
        hidden_size=32,
        num_layers=1,
        dropout=0.2,
        nhead=4,
        max_seq_len=5
    ):
        super().__init__()

        # Projetar as 6 características para 32 dimensões
        self.input_projection = nn.Linear(
            input_size, hidden_size
        )

        # Codificação posicional aprendida
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_seq_len, hidden_size)
        )

        nn.init.normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=64,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x: (batch, 5, 6)

        x = self.input_projection(x)

        seq_len = x.size(1)

        if seq_len > self.position_embedding.size(1):
            raise ValueError(
                "Sequência maior do que max_seq_len"
            )

        # Preservar informação sobre a posição temporal
        x = x + self.position_embedding[:, :seq_len, :]

        # Mecanismo de atenção
        x = self.transformer(x)

        # Classificar usando a representação
        # contextualizada do último minuto
        last_output = x[:, -1, :]

        return self.classifier(
            last_output
        ).squeeze(-1)
