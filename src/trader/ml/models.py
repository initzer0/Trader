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


class GRUReturnRegressor(nn.Module):
    def __init__(
        self,
        *,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
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
        out, _ = self.gru(x)  # (B, L, H)
        last = out[:, -1, :]  # (B, H)
        return self.head(last)  # (B, 1)


class MLPReturnRegressor(nn.Module):
    def __init__(
        self,
        *,
        lookback: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if lookback <= 0:
            raise ValueError("lookback must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")

        layers = []
        in_dim = lookback
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_size

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 1)
        x = x.squeeze(-1)
        out = self.backbone(x)
        return self.head(out)


class CNNReturnRegressor(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int = 1,
        base_channels: int = 16,
        num_layers: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        layers = []
        in_ch = input_channels
        for i in range(num_layers):
            out_ch = base_channels * (2**i)
            layers.append(
                nn.Conv1d(
                    in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2
                )
            )
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.LayerNorm(in_ch),
            nn.Linear(in_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 1) -> (B, 1, L)
        x = x.transpose(1, 2)
        out = self.conv(x)
        pooled = out.mean(dim=-1)  # global average pool over time
        return self.head(pooled)
