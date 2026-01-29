from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import time
from typing import Optional, Sequence

import pandas as pd


class GoogleTrendsError(RuntimeError):
    pass


def _open_time_ms_from_idx(idx: pd.DatetimeIndex) -> list[int]:
    """Convert a UTC DatetimeIndex to epoch milliseconds.

    pytrends sometimes returns datetime64[s] (seconds resolution) instead of
    the more typical datetime64[ns]. We must respect the underlying unit.
    """

    # pandas uses a DatetimeTZDtype for tz-aware indexes
    unit = getattr(getattr(idx, "dtype", None), "unit", "ns")
    ints = idx.view("int64")

    if unit == "s":
        ms = ints * 1000
    elif unit == "ms":
        ms = ints
    elif unit == "us":
        ms = ints // 1000
    elif unit == "ns":
        ms = ints // 1_000_000
    else:  # pragma: no cover
        raise GoogleTrendsError(f"Unsupported datetime unit from pytrends: {unit}")

    # Return as a plain sequence to avoid index-alignment issues on DataFrame.insert.
    return ms.astype("int64").tolist()


@dataclass(frozen=True)
class GoogleTrendsQuery:
    keywords: Sequence[str]
    timeframe: str
    geo: str = ""
    gprop: str = ""
    hl: str = "en-US"
    tz: int = 0  # 0 == UTC in pytrends


def _require_pytrends():
    try:
        from pytrends.request import TrendReq  # type: ignore

        return TrendReq
    except Exception as exc:  # pragma: no cover
        raise GoogleTrendsError(
            "pytrends is required. Install it with: pip install pytrends"
        ) from exc


def fetch_interest_over_time(query: GoogleTrendsQuery) -> pd.DataFrame:
    """Fetch Google Trends interest-over-time for one or more keywords.

    Notes
    - Google Trends does not provide a single universal 'best resolution' for all ranges.
      The returned frequency depends on the timeframe:
        - `now 7-d` often yields hourly data
        - longer periods typically yield daily/weekly data
    - Returned values are 0-100 scaled *within the requested timeframe*.

    Returns a DataFrame with columns:
      - open_time (UTC ISO string)
      - open_time_ms
      - one column per keyword
      - is_partial (if present from Google)
    """

    if not query.keywords:
        raise ValueError("keywords must be non-empty")

    TrendReq = _require_pytrends()
    pytrends = TrendReq(hl=query.hl, tz=query.tz)

    pytrends.build_payload(
        list(query.keywords),
        timeframe=query.timeframe,
        geo=query.geo,
        gprop=query.gprop,
    )

    df = pytrends.interest_over_time()
    if df is None or df.empty:
        raise GoogleTrendsError(
            f"No Google Trends data returned for {query.keywords} ({query.timeframe})."
        )

    # pytrends returns a DateTimeIndex; treat as UTC when tz=0.
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    out = df.copy()
    out.insert(0, "open_time", idx)
    out["open_time"] = out["open_time"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    out.insert(1, "open_time_ms", _open_time_ms_from_idx(idx))

    # Ensure deterministic column order.
    cols = ["open_time", "open_time_ms", *list(query.keywords)]
    if "isPartial" in out.columns:
        out = out.rename(columns={"isPartial": "is_partial"})
        cols.append("is_partial")

    keep = [c for c in cols if c in out.columns]
    out = out[keep].reset_index(drop=True)
    return out


def timeframe_from_start_end(
    start: datetime,
    end: datetime,
) -> str:
    """Build a pytrends-compatible timeframe string from datetimes."""

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    if end_utc <= start_utc:
        raise ValueError("end must be after start")

    # pytrends accepts 'YYYY-MM-DD YYYY-MM-DD'
    return f"{start_utc:%Y-%m-%d} {end_utc:%Y-%m-%d}"


def default_keywords() -> list[str]:
    return ["cryptocurrency"]


def fetch_hourly_interest_stitched(
    *,
    keywords: Sequence[str],
    start: datetime,
    end: datetime,
    geo: str = "",
    gprop: str = "",
    hl: str = "en-US",
    tz: int = 0,
    window_days: int = 7,
    overlap_hours: int = 24,
    sleep_s: float = 1.0,
    max_retries: int = 5,
    renormalize_0_100: bool = False,
) -> pd.DataFrame:
    """Backfill an *hourly* Google Trends series by stitching short windows.

    Why this exists
    - Google Trends only returns hourly resolution for short timeframes.
      For multi-year ranges it returns daily/weekly.
    - To approximate hourly history (e.g. since 2016), we query many short
      windows (default 7 days) and stitch them.

    Stitching / scaling
    - Google scales values 0-100 *within each request*.
    - We use an overlap region and compute a robust multiplicative factor to
      align each new window to the already-stitched series.
    - Resulting values may exceed 100 (unless renormalize_0_100=True).

        Practical notes
        - This can take a long time and may hit rate limits; use sleep_s.
        - The series is still an approximation because Trends data is sampled.
        - IMPORTANT: Google often only provides hourly resolution for recent ranges.
            If the returned data for a window is not hourly, this function raises.
    """

    if not keywords:
        raise ValueError("keywords must be non-empty")
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    if end <= start:
        raise ValueError("end must be after start")
    if window_days <= 0:
        raise ValueError("window_days must be > 0")
    if overlap_hours < 0:
        raise ValueError("overlap_hours must be >= 0")

    TrendReq = _require_pytrends()
    pytrends = TrendReq(hl=hl, tz=tz)

    window = timedelta(days=window_days)
    step = window - timedelta(hours=overlap_hours)
    if step.total_seconds() <= 0:
        raise ValueError("overlap_hours too large for window_days")

    def fetch_window(w_start: datetime, w_end: datetime) -> pd.DataFrame:
        timeframe = timeframe_from_start_end(w_start, w_end)
        query = GoogleTrendsQuery(
            keywords=list(keywords),
            timeframe=timeframe,
            geo=geo,
            gprop=gprop,
            hl=hl,
            tz=tz,
        )

        last_exc: Optional[BaseException] = None
        for attempt in range(max_retries + 1):
            try:
                pytrends.build_payload(
                    list(query.keywords),
                    timeframe=query.timeframe,
                    geo=query.geo,
                    gprop=query.gprop,
                )
                df = pytrends.interest_over_time()
                if df is None or df.empty:
                    raise GoogleTrendsError(
                        f"Empty Trends response for {query.keywords} ({query.timeframe})."
                    )

                idx = pd.to_datetime(df.index, utc=True, errors="coerce")
                out = df.copy()
                out.insert(0, "open_time", idx)
                out["open_time"] = out["open_time"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
                out.insert(1, "open_time_ms", _open_time_ms_from_idx(idx))

                cols = ["open_time", "open_time_ms", *list(query.keywords)]
                if "isPartial" in out.columns:
                    out = out.rename(columns={"isPartial": "is_partial"})
                    cols.append("is_partial")

                keep = [c for c in cols if c in out.columns]
                return out[keep].reset_index(drop=True)
            except Exception as exc:  # pragma: no cover
                last_exc = exc
                backoff = min(60.0, (2.0**attempt) * 1.0)
                time.sleep(backoff)
        raise GoogleTrendsError(f"Failed to fetch window {timeframe}: {last_exc}")

    stitched: Optional[pd.DataFrame] = None
    w_start = start
    while w_start < end:
        w_end = min(end, w_start + window)

        df = fetch_window(w_start, w_end)

        # Enforce hourly resolution; otherwise this isn't the data the caller asked for.
        if df.shape[0] >= 3:
            diffs = df["open_time_ms"].astype("int64").diff().dropna()
            if not diffs.empty:
                # Median step should be 1 hour (3600000 ms)
                if float(diffs.median()) > 3_600_000:
                    raise GoogleTrendsError(
                        "Google Trends did not return hourly data for this timeframe. "
                        "Hourly resolution is typically only available for recent ranges; "
                        "use fetch_interest_over_time for long-history daily/weekly data."
                    )

        if sleep_s > 0:
            time.sleep(sleep_s)

        if stitched is None:
            stitched = df
        else:
            anchor = keywords[0]
            if (
                anchor in stitched.columns
                and anchor in df.columns
                and overlap_hours > 0
            ):
                # Overlap region: last overlap_hours of stitched vs first overlap_hours of new window
                overlap_start_ms = int(
                    stitched["open_time_ms"].max() - overlap_hours * 3600 * 1000
                )
                prev_ov = stitched.loc[
                    stitched["open_time_ms"] >= overlap_start_ms,
                    ["open_time_ms", anchor],
                ]
                cur_ov = df.loc[
                    df["open_time_ms"] >= overlap_start_ms, ["open_time_ms", anchor]
                ]

                merged = prev_ov.merge(
                    cur_ov, on="open_time_ms", suffixes=("_prev", "_cur")
                )
                merged = merged.dropna()
                merged = merged[
                    (merged[f"{anchor}_prev"] > 0) & (merged[f"{anchor}_cur"] > 0)
                ]

                if not merged.empty:
                    ratios = merged[f"{anchor}_prev"] / merged[f"{anchor}_cur"]
                    factor = float(ratios.median())
                    if factor > 0 and factor != 1.0:
                        for k in keywords:
                            if k in df.columns:
                                df[k] = df[k].astype(float) * factor

            stitched = pd.concat([stitched, df], ignore_index=True)

        w_start = w_start + step

    if stitched is None or stitched.empty:
        raise GoogleTrendsError("No data stitched")

    stitched = stitched.drop_duplicates(subset=["open_time_ms"], keep="last")
    stitched = stitched.sort_values("open_time_ms").reset_index(drop=True)

    if renormalize_0_100:
        for k in keywords:
            if k in stitched.columns:
                mx = float(pd.to_numeric(stitched[k], errors="coerce").max())
                if mx > 0:
                    stitched[k] = stitched[k].astype(float) * (100.0 / mx)

    return stitched
