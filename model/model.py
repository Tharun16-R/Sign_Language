import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        # take last timestep
        last = out[:, -1, :]
        logits = self.head(last)
        return logits
