from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from trader.dataset.dataset import Dataset


@dataclass(frozen=True)
class DataConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1m"

    downsample_every: int = 1
    lookback: int = 30
    prediction_horizon: int = 1
    sample_every: int = 1

    split_dt: datetime = datetime(2025, 1, 1, tzinfo=timezone.utc)
    val_fraction: float = 0.1


@dataclass(frozen=True)
class PreparedSeries:
    times: pd.DatetimeIndex
    close: np.ndarray


@dataclass(frozen=True)
class Samples:
    x: np.ndarray
    y: np.ndarray
    base_index: np.ndarray


@dataclass(frozen=True)
class InferenceSamples:
    x: np.ndarray
    base_index: np.ndarray


def default_btc_1m_csv(project_root: Path) -> Path:
    return project_root / "data" / "crypto" / "btc" / "BTCUSDT_1m.csv"


def load_ohlc_close_series(
    csv_path: Path,
    *,
    downsample_every: int = 1,
) -> PreparedSeries:
    """Load only the columns needed for close-price based modeling.

    Uses Dataset to parse open_time and optionally downsample with OHLC-aware rules.
    """

    ds = Dataset.from_csv(
        csv_path,
        usecols=["open_time", "close"],
        parse_open_time=True,
        coerce_numeric=True,
    )

    if downsample_every > 1:
        ds = ds.splice(downsample_every)

    df = ds.df.dropna(subset=["open_time", "close"]).sort_values("open_time")

    times = pd.DatetimeIndex(df["open_time"], name="open_time")
    close = df["close"].astype(float).to_numpy()

    return PreparedSeries(times=times, close=close)


def make_supervised_samples(
    series: PreparedSeries,
    *,
    lookback: int,
    prediction_horizon: int,
    sample_every: int = 1,
) -> Samples:
    """Create (X, y) for return prediction.

    Features: last `lookback` realized 1-step log returns up to time t:
        r_t = log(close_t / close_{t-1})

    Target: forward log return over `prediction_horizon` steps from time t:
        y_t = log(close_{t+h} / close_t)

    Samples are generated at base indices t spaced by `sample_every` bars.
    """

    if lookback <= 0:
        raise ValueError("lookback must be > 0")
    if prediction_horizon <= 0:
        raise ValueError("prediction_horizon must be > 0")
    if sample_every <= 0:
        raise ValueError("sample_every must be > 0")

    close = series.close
    if close.size < lookback + prediction_horizon + 2:
        raise ValueError("Not enough data for requested lookback/horizon")

    log_close = np.log(close)
    realized = np.diff(
        log_close, prepend=np.nan
    )  # realized[t] = log(close_t/close_{t-1})

    start = max(lookback, 1)
    end = close.size - prediction_horizon - 1

    base_indices = np.arange(start, end + 1, sample_every, dtype=int)

    x = np.empty((base_indices.size, lookback), dtype=np.float32)
    y = np.empty((base_indices.size,), dtype=np.float32)

    for j, i in enumerate(base_indices):
        window = realized[i - lookback + 1 : i + 1]
        if np.isnan(window).any():
            # This only happens near the beginning; skip by forcing start>=lookback.
            raise RuntimeError("Unexpected NaN in realized return window")
        x[j, :] = window.astype(np.float32)
        y[j] = float(log_close[i + prediction_horizon] - log_close[i])

    return Samples(x=x, y=y, base_index=base_indices)


def make_inference_samples(
    series: PreparedSeries,
    *,
    lookback: int,
    sample_every: int = 1,
    start_index: int = 0,
    end_index: Optional[int] = None,
) -> InferenceSamples:
    """Create input windows (X) for inference/trading without requiring targets.

    This is crucial for evaluating a strategy over a fixed calendar interval,
    because supervised targets stop at `close.size - prediction_horizon - 1`,
    but in live trading you still produce predictions near the end.
    """

    if lookback <= 0:
        raise ValueError("lookback must be > 0")
    if sample_every <= 0:
        raise ValueError("sample_every must be > 0")

    close = series.close
    if end_index is None:
        end_index = int(close.size - 1)
    else:
        end_index = int(end_index)

    start_index = int(start_index)
    if end_index <= 0 or end_index >= close.size:
        raise ValueError("end_index out of range")

    start = max(lookback, 1, start_index)
    end = end_index
    if end < start:
        raise ValueError("Not enough data for requested lookback")

    log_close = np.log(close)
    realized = np.diff(log_close, prepend=np.nan)

    base_indices = np.arange(start, end + 1, sample_every, dtype=int)
    x = np.empty((base_indices.size, lookback), dtype=np.float32)

    for j, i in enumerate(base_indices):
        window = realized[i - lookback + 1 : i + 1]
        if np.isnan(window).any():
            raise RuntimeError("Unexpected NaN in realized return window")
        x[j, :] = window.astype(np.float32)

    return InferenceSamples(x=x, base_index=base_indices)


@dataclass(frozen=True)
class SplitSamples:
    train: Samples
    val: Samples
    test: Samples


def time_split_samples(
    series: PreparedSeries,
    samples: Samples,
    *,
    split_dt: datetime,
    val_fraction: float,
    prediction_horizon: int,
) -> SplitSamples:
    """Split into train/val/test by time.

    - Train/val: base time < split_dt AND target time < split_dt
    - Test: base time >= split_dt

    Validation is the last `val_fraction` portion of the pre-split samples (time-ordered).
    """

    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be in (0, 0.5)")

    times = series.times
    base_t = times[samples.base_index]
    target_t = times[samples.base_index + prediction_horizon]

    pre_mask = (base_t < split_dt) & (target_t < split_dt)
    post_mask = base_t >= split_dt

    pre_idx = np.nonzero(np.asarray(pre_mask))[0]
    post_idx = np.nonzero(np.asarray(post_mask))[0]

    if pre_idx.size < 100:
        raise ValueError("Not enough pre-2025 samples for training")
    if post_idx.size < 10:
        raise ValueError("Not enough 2025+ samples for testing")

    # Keep time order.
    split_point = int(np.floor((1.0 - val_fraction) * pre_idx.size))
    split_point = max(1, min(pre_idx.size - 1, split_point))

    train_rows = pre_idx[:split_point]
    val_rows = pre_idx[split_point:]
    test_rows = post_idx

    def take(rows: np.ndarray) -> Samples:
        return Samples(
            x=samples.x[rows],
            y=samples.y[rows],
            base_index=samples.base_index[rows],
        )

    return SplitSamples(
        train=take(train_rows), val=take(val_rows), test=take(test_rows)
    )
