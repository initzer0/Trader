from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from dotenv import load_dotenv
from tqdm import tqdm


class BinanceOHLCError(RuntimeError):
    pass


_INTERVAL_TO_MS: Dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
    "1M": 30 * 24 * 60 * 60_000,  # approximate; Binance month boundary isn't fixed
}


_CSV_FIELDS: Sequence[str] = (
    "open_time",
    "open_time_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
)


def _utc_iso_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _ms_from_datetime(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def _interval_ms(interval: str) -> int:
    try:
        return _INTERVAL_TO_MS[interval]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported interval '{interval}'. Supported: {sorted(_INTERVAL_TO_MS.keys())}"
        ) from exc


@dataclass(frozen=True)
class CandleRow:
    open_time_ms: int
    open: str
    high: str
    low: str
    close: str
    volume: str
    close_time_ms: int
    quote_asset_volume: str
    number_of_trades: int
    taker_buy_base_asset_volume: str
    taker_buy_quote_asset_volume: str

    def to_csv_row(self) -> Dict[str, str]:
        return {
            "open_time": _utc_iso_from_ms(self.open_time_ms),
            "open_time_ms": str(self.open_time_ms),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "close_time_ms": str(self.close_time_ms),
            "quote_asset_volume": self.quote_asset_volume,
            "number_of_trades": str(self.number_of_trades),
            "taker_buy_base_asset_volume": self.taker_buy_base_asset_volume,
            "taker_buy_quote_asset_volume": self.taker_buy_quote_asset_volume,
        }


def create_binance_client(env_path: Optional[os.PathLike[str] | str] = None) -> Client:
    """Create a Binance client from .env variables.

    Expected vars:
      - BINANCE_API_KEY
      - BINANCE_SECRET_KEY

    Notes:
      - Public OHLC data does not require keys, but providing them is fine.
    """

    if env_path is None:
        load_dotenv()
    else:
        load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")

    if not api_key or not api_secret:
        raise BinanceOHLCError(
            "Missing BINANCE_API_KEY/BINANCE_SECRET_KEY. Put them in a .env file or pass env_path."
        )

    return Client(api_key, api_secret)


def _parse_klines_to_rows(klines: Sequence[Sequence]) -> List[CandleRow]:
    rows: List[CandleRow] = []
    for k in klines:
        # Binance kline array:
        # [0 open_time, 1 open, 2 high, 3 low, 4 close, 5 volume,
        #  6 close_time, 7 quote_asset_volume, 8 number_of_trades,
        #  9 taker_buy_base_asset_volume, 10 taker_buy_quote_asset_volume, 11 ignore]
        rows.append(
            CandleRow(
                open_time_ms=int(k[0]),
                open=str(k[1]),
                high=str(k[2]),
                low=str(k[3]),
                close=str(k[4]),
                volume=str(k[5]),
                close_time_ms=int(k[6]),
                quote_asset_volume=str(k[7]),
                number_of_trades=int(k[8]),
                taker_buy_base_asset_volume=str(k[9]),
                taker_buy_quote_asset_volume=str(k[10]),
            )
        )
    return rows


def _get_earliest_timestamp_ms(client: Client, symbol: str, interval: str) -> int:
    # python-binance provides this helper in newer versions. Some older versions
    # don't have it, so fall back to a single kline query.
    method = getattr(client, "get_earliest_valid_timestamp", None)
    if callable(method):
        return int(method(symbol, interval))

    klines = client.get_klines(symbol=symbol, interval=interval, startTime=0, limit=1)
    if not klines:
        raise BinanceOHLCError(
            f"Could not determine earliest timestamp for {symbol} {interval} (no klines returned)."
        )
    return int(klines[0][0])


def fetch_klines(
    client: Client,
    *,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    max_retries: int = 6,
) -> Iterator[CandleRow]:
    """Fetch klines from Binance, yielding CandleRow.

    start_ms/end_ms are timestamps in milliseconds.
    end_ms is treated as inclusive for open_time.
    """

    step_ms = _interval_ms(interval)
    cursor = start_ms

    pbar_total = max(0, ((end_ms - start_ms) // step_ms) + 1)
    with tqdm(total=pbar_total, unit="candles", desc=f"{symbol} {interval}") as pbar:
        while cursor <= end_ms:
            retries = 0
            while True:
                try:
                    klines = client.get_klines(
                        symbol=symbol,
                        interval=interval,
                        startTime=int(cursor),
                        endTime=int(end_ms),
                        limit=limit,
                    )
                    break
                except (BinanceRequestException, BinanceAPIException) as exc:
                    retries += 1
                    if retries > max_retries:
                        raise BinanceOHLCError(
                            f"Binance request failed after {max_retries} retries: {exc}"
                        ) from exc
                    sleep_s = min(30.0, 2.0**retries)
                    time.sleep(sleep_s)

            if not klines:
                return

            rows = _parse_klines_to_rows(klines)
            for r in rows:
                yield r

            pbar.update(len(rows))

            last_open = rows[-1].open_time_ms
            next_cursor = last_open + step_ms
            if next_cursor <= cursor:
                # Safety guard: avoid infinite loops.
                next_cursor = cursor + step_ms
            cursor = next_cursor

            # Gentle pacing to reduce the chance of triggering rate limits.
            time.sleep(0.05)


def _read_existing_csv(csv_path: Path) -> Dict[int, Dict[str, str]]:
    if not csv_path.exists():
        return {}

    rows: Dict[int, Dict[str, str]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}

        # accept either open_time_ms or open_time in older files
        for row in reader:
            ms_str = row.get("open_time_ms")
            if ms_str is None or ms_str == "":
                # Try parsing ISO timestamp if needed
                iso = row.get("open_time")
                if not iso:
                    continue
                dt = datetime.fromisoformat(iso)
                open_ms = _ms_from_datetime(dt)
            else:
                open_ms = int(ms_str)

            rows[open_ms] = row
    return rows


def _write_csv(csv_path: Path, rows_by_open_ms: Dict[int, Dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    keys = sorted(rows_by_open_ms.keys())

    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(_CSV_FIELDS))
        writer.writeheader()
        for k in keys:
            row = rows_by_open_ms[k]
            # normalize row fields
            normalized = {field: row.get(field, "") for field in _CSV_FIELDS}
            if not normalized.get("open_time"):
                normalized["open_time"] = _utc_iso_from_ms(k)
            if not normalized.get("open_time_ms"):
                normalized["open_time_ms"] = str(k)
            writer.writerow(normalized)

    tmp_path.replace(csv_path)


def _find_gaps(sorted_open_times: Sequence[int], step_ms: int) -> List[Tuple[int, int]]:
    """Return missing segments as [(gap_start_open_ms, gap_end_open_ms), ...]."""
    gaps: List[Tuple[int, int]] = []
    if len(sorted_open_times) < 2:
        return gaps

    prev = sorted_open_times[0]
    for current in sorted_open_times[1:]:
        expected = prev + step_ms
        if current > expected:
            gaps.append((expected, current - step_ms))
        prev = current

    return gaps


def update_ohlc_csv(
    *,
    client: Client,
    symbol: str,
    interval: str,
    csv_path: os.PathLike[str] | str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Path:
    """Ensure a CSV contains all candles in [start, end].

    - If the file does not exist, downloads the full range.
    - If the file exists, detects gaps and downloads missing segments.

    If start is None, uses Binance's earliest valid timestamp for the symbol+interval.
    If end is None, uses current UTC time.
    """

    csv_path = Path(csv_path)
    step_ms = _interval_ms(interval)

    end_ms = _ms_from_datetime(end or datetime.now(timezone.utc))

    if start is None:
        start_ms = _get_earliest_timestamp_ms(client, symbol, interval)
    else:
        start_ms = _ms_from_datetime(start)

    # Build in-memory index
    existing_by_open = _read_existing_csv(csv_path)
    existing_open_times = sorted(existing_by_open.keys())

    segments: List[Tuple[int, int]] = []

    if not existing_open_times:
        segments.append((start_ms, end_ms))
    else:
        first = existing_open_times[0]
        last = existing_open_times[-1]

        # Extend earlier/later than existing coverage
        if start_ms < first:
            segments.append((start_ms, first - step_ms))
        if end_ms > last:
            segments.append((last + step_ms, end_ms))

        # Fill internal gaps within existing file
        gaps = _find_gaps(existing_open_times, step_ms)
        segments.extend(gaps)

        # Also fill gaps that fall inside requested [start, end] but outside file range
        # (handled by the earlier/later segments above)

    # Clip segments to requested range and remove invalid ones
    clipped: List[Tuple[int, int]] = []
    for seg_start, seg_end in segments:
        seg_start = max(seg_start, start_ms)
        seg_end = min(seg_end, end_ms)
        if seg_start <= seg_end:
            clipped.append((seg_start, seg_end))

    # Deduplicate/merge overlapping segments
    clipped.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in clipped:
        if not merged:
            merged.append((s, e))
            continue
        ps, pe = merged[-1]
        if s <= pe + step_ms:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    if not merged:
        # Nothing to do
        return csv_path

    # Download and merge
    for seg_start, seg_end in merged:
        for candle in fetch_klines(
            client,
            symbol=symbol,
            interval=interval,
            start_ms=seg_start,
            end_ms=seg_end,
        ):
            existing_by_open[candle.open_time_ms] = candle.to_csv_row()

    _write_csv(csv_path, existing_by_open)
    return csv_path
