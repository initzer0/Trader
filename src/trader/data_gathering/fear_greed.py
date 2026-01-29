from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests


class FearGreedError(RuntimeError):
    pass


_DEFAULT_URL = "https://api.alternative.me/fng/"


@dataclass(frozen=True)
class FearGreedQuery:
    limit: int = 0  # 0 means 'all' per API
    url: str = _DEFAULT_URL


def fetch_fear_greed_index(query: FearGreedQuery = FearGreedQuery()) -> pd.DataFrame:
    """Fetch the Alternative.me Crypto Fear & Greed Index.

    Returns a DataFrame with columns:
      - open_time (UTC ISO)
      - open_time_ms
      - value (float)
      - classification (string)

    Notes
    - Frequency is typically daily.
    - This is a free, unauthenticated endpoint but can rate-limit.
    """

    params = {
        "limit": str(query.limit),
        "format": "json",
    }

    try:
        resp = requests.get(query.url, params=params, timeout=30)
    except requests.RequestException as exc:
        raise FearGreedError(f"Request failed: {exc}") from exc

    if resp.status_code != 200:
        raise FearGreedError(f"HTTP {resp.status_code}: {resp.text[:300]}")

    payload = resp.json()
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        raise FearGreedError("Unexpected response payload (missing data array)")

    out_rows: list[dict] = []
    for r in rows:
        try:
            ts = int(r["timestamp"])
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            value = float(r["value"])
            cls = str(r.get("value_classification", ""))
        except Exception:
            continue

        out_rows.append(
            {
                "open_time": dt.isoformat(),
                "open_time_ms": int(ts * 1000),
                "value": value,
                "classification": cls,
            }
        )

    if not out_rows:
        raise FearGreedError("No usable rows parsed from response")

    df = pd.DataFrame(out_rows).sort_values("open_time_ms").reset_index(drop=True)
    return df
