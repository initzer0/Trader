from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class TorchBatch:
    x: "object"  # torch.Tensor
    y: "object"  # torch.Tensor


class ReturnSequenceTorchDataset:
    """Thin wrapper so the core sampling code stays numpy/pandas-only."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        import torch

        self.x = torch.from_numpy(x).float().unsqueeze(-1)  # (N, L, 1)
        self.y = torch.from_numpy(y).float().unsqueeze(-1)  # (N, 1)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple["object", "object"]:
        return self.x[idx], self.y[idx]
