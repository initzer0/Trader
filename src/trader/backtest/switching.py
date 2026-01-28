from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Position = Literal["USDT", "BTC"]


@dataclass(frozen=True)
class BacktestResult:
    equity_curve_gross: np.ndarray
    equity_curve_net: np.ndarray
    positions: np.ndarray
    n_trades: int

    @property
    def final_value_gross(self) -> float:
        """Final portfolio value when starting from 1.0 (e.g. €1)."""

        return float(self.equity_curve_gross[-1])

    @property
    def final_value_net(self) -> float:
        """Final portfolio value when starting from 1.0 (e.g. €1), net of fees."""

        return float(self.equity_curve_net[-1])

    @property
    def total_return_gross(self) -> float:
        return float(self.equity_curve_gross[-1] / self.equity_curve_gross[0] - 1.0)

    @property
    def total_return_net(self) -> float:
        return float(self.equity_curve_net[-1] / self.equity_curve_net[0] - 1.0)


def backtest_long_or_cash(
    *,
    close: np.ndarray,
    base_index: np.ndarray,
    pred_log_return: np.ndarray,
    trade_horizon: int = 1,
    fee_rate: float = 0.001,
    start_in: Position = "USDT",
) -> BacktestResult:
    """Simple BTC/USDT switching strategy.

    At each decision time i (given by base_index[k]):
      - if pred_log_return[k] > 0 -> hold BTC
      - else -> hold USDT

    The position is held for `trade_horizon` bars, then a new decision is made.

    Fee model:
      - Apply `fee_rate` once whenever we switch assets.
      - Results reported both gross (no fees) and net (with fees).
    """

    if close.ndim != 1:
        raise ValueError("close must be 1D")
    if base_index.ndim != 1 or pred_log_return.ndim != 1:
        raise ValueError("base_index and pred_log_return must be 1D")
    if base_index.size != pred_log_return.size:
        raise ValueError("base_index and pred_log_return must have same length")
    if trade_horizon <= 0:
        raise ValueError("trade_horizon must be > 0")
    if not (0.0 <= fee_rate < 0.1):
        raise ValueError("fee_rate should be a fraction, e.g. 0.001 for 0.1%")

    # Ensure decisions are in chronological order.
    order = np.argsort(base_index)
    base_index = base_index[order]
    pred_log_return = pred_log_return[order]

    n = base_index.size
    equity_g = np.ones(n + 1, dtype=np.float64)
    equity_n = np.ones(n + 1, dtype=np.float64)

    positions = np.empty(n, dtype=object)
    prev_pos: Position = start_in
    trades = 0

    for k in range(n):
        i = int(base_index[k])
        j = i + trade_horizon
        if j >= close.size:
            # No future price to realize; stop.
            equity_g = equity_g[: k + 1]
            equity_n = equity_n[: k + 1]
            positions = positions[:k]
            break

        pos: Position = "BTC" if pred_log_return[k] > 0.0 else "USDT"
        positions[k] = pos

        # Gross update
        if pos == "BTC":
            equity_g[k + 1] = equity_g[k] * (close[j] / close[i])
        else:
            equity_g[k + 1] = equity_g[k]

        # Net update (fee on switch)
        switch_fee = 0.0
        if pos != prev_pos:
            switch_fee = fee_rate
            trades += 1
            prev_pos = pos

        if pos == "BTC":
            equity_n[k + 1] = equity_n[k] * (1.0 - switch_fee) * (close[j] / close[i])
        else:
            equity_n[k + 1] = equity_n[k] * (1.0 - switch_fee)

    return BacktestResult(
        equity_curve_gross=equity_g,
        equity_curve_net=equity_n,
        positions=positions,
        n_trades=trades,
    )
