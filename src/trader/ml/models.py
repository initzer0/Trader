from __future__ import annotations

import torch
from torch import nn


class LSTMReturnRegressor(nn.Module):
    def __init__(
        self,
        *,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # (B, L, H)
        last = out[:, -1, :]  # (B, H)
        return self.head(last)  # (B, 1)
