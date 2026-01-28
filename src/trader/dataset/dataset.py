from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import pandas as pd


AggFunc = Union[str, Callable[[pd.Series], Any]]
AggMap = Mapping[str, AggFunc]


@dataclass(frozen=True)
class ColumnRoles:
    """Declarative way to describe how columns should be aggregated.

    Anything not mentioned falls back to Dataset's default policy.
    """

    first: Sequence[str] = ()
    last: Sequence[str] = ()
    min: Sequence[str] = ()
    max: Sequence[str] = ()
    sum: Sequence[str] = ()
    mean: Sequence[str] = ()

    def to_agg_map(self) -> Dict[str, str]:
        agg: Dict[str, str] = {}
        for col in self.first:
            agg[col] = "first"
        for col in self.last:
            agg[col] = "last"
        for col in self.min:
            agg[col] = "min"
        for col in self.max:
            agg[col] = "max"
        for col in self.sum:
            agg[col] = "sum"
        for col in self.mean:
            agg[col] = "mean"
        return agg


class Dataset:
    """Small helper around a pandas DataFrame with OHLCV-friendly aggregation."""

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df if df is not None else pd.DataFrame()

    def __getitem__(self, key: Any) -> Any:
        """Enable Python slicing / selection on the underlying DataFrame.

        Notes
        - `ds[::60]` here means "take every 60th row" (like `.iloc[::60]`).
        - Aggregating 60x 1m -> 1x 1h is a different operation; use `ds.splice(60)`.
        """

        if isinstance(key, slice):
            return Dataset(self.df.iloc[key])

        result = self.df.__getitem__(key)
        if isinstance(result, pd.DataFrame):
            return Dataset(result)
        return result

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        nrows: Optional[int] = None,
        parse_open_time: bool = True,
        coerce_numeric: bool = True,
        **read_csv_kwargs: Any,
    ) -> "Dataset":
        ds = cls()
        ds.load_csv(
            path,
            nrows=nrows,
            parse_open_time=parse_open_time,
            coerce_numeric=coerce_numeric,
            **read_csv_kwargs,
        )
        return ds

    def load_csv(
        self,
        path: str | Path,
        *,
        nrows: Optional[int] = None,
        parse_open_time: bool = True,
        coerce_numeric: bool = True,
        **read_csv_kwargs: Any,
    ) -> "Dataset":
        """Load a CSV into `self.df`.

        - If `open_time` exists and `parse_open_time=True`, it is parsed to UTC datetime.
        - If `coerce_numeric=True`, object columns that look numeric are converted.
        """

        df = pd.read_csv(Path(path), nrows=nrows, **read_csv_kwargs)

        if parse_open_time and "open_time" in df.columns:
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")

        if coerce_numeric:
            df = self._coerce_numeric_like(df)

        self.df = df
        return self

    def splice(
        self,
        every: int,
        *,
        agg: Optional[AggMap] = None,
        roles: Optional[ColumnRoles] = None,
        default: AggFunc = "last",
        keep_remainder: bool = False,
    ) -> "Dataset":
        """Aggregate every N rows into a single row (OHLCV-style chunking).

        This is the semantic equivalent of taking 60x 1m candles and producing one 1h
        candle, but with correct per-column aggregation.

        Parameters
        - every: chunk size (e.g. 60 for 1m -> 1h)
        - agg: explicit column->aggregation mapping (pandas agg strings/callables)
        - roles: declarative helper for building the agg map
        - default: aggregation for columns not mentioned in agg/roles (default: last)
        - keep_remainder: include the final partial chunk (default: False)
        """

        if every <= 0:
            raise ValueError("every must be a positive integer")
        if self.df is None or self.df.empty:
            return Dataset(
                pd.DataFrame(columns=self.df.columns if self.df is not None else None)
            )

        df = self.df.copy()
        if not keep_remainder:
            usable = len(df) - (len(df) % every)
            df = df.iloc[:usable]

        if df.empty:
            return Dataset(df)

        agg_map = self._build_agg_map(df, agg=agg, roles=roles, default=default)

        group_ids = pd.Series(range(len(df)), index=df.index) // every
        out = df.groupby(group_ids, sort=False).agg(agg_map).reset_index(drop=True)
        return Dataset(out)

    def resample_time(
        self,
        rule: str,
        *,
        time_col: str = "open_time",
        agg: Optional[AggMap] = None,
        roles: Optional[ColumnRoles] = None,
        default: AggFunc = "last",
        label: str = "left",
        closed: str = "left",
    ) -> "Dataset":
        """Time-based resampling using a datetime column (preferred when available)."""

        if self.df is None or self.df.empty:
            return Dataset(
                pd.DataFrame(columns=self.df.columns if self.df is not None else None)
            )
        if time_col not in self.df.columns:
            raise KeyError(f"Missing time column: {time_col}")

        df = self.df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

        df = df.dropna(subset=[time_col]).sort_values(time_col)
        agg_map = self._build_agg_map(df, agg=agg, roles=roles, default=default)

        out = (
            df.set_index(time_col)
            .resample(rule, label=label, closed=closed)
            .agg(agg_map)
            .dropna(how="all")
            .reset_index()
        )
        return Dataset(out)

    @staticmethod
    def default_ohlcv_roles() -> ColumnRoles:
        """Defaults tailored to Binance OHLC CSVs in this repo."""

        return ColumnRoles(
            first=("open",),
            last=("close",),
            max=("high", "close_time_ms"),
            min=("low", "open_time", "open_time_ms"),
            sum=(
                "volume",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
            ),
        )

    @staticmethod
    def _build_agg_map(
        df: pd.DataFrame,
        *,
        agg: Optional[AggMap],
        roles: Optional[ColumnRoles],
        default: AggFunc,
    ) -> Dict[str, AggFunc]:
        overrides: Dict[str, AggFunc] = {}

        # Start with strong defaults for the known OHLCV schema.
        overrides.update(Dataset.default_ohlcv_roles().to_agg_map())

        if roles is not None:
            overrides.update(roles.to_agg_map())
        if agg is not None:
            overrides.update(dict(agg))

        # Ensure every column has an aggregation.
        full: Dict[str, AggFunc] = {c: overrides.get(c, default) for c in df.columns}
        return full

    @staticmethod
    def _coerce_numeric_like(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:
            if pd.api.types.is_object_dtype(out[col]):
                out[col] = pd.to_numeric(out[col], errors="ignore")
        return out
