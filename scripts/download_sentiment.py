#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

# Allow running without installing the package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trader.data_gathering.fear_greed import FearGreedQuery, fetch_fear_greed_index
from trader.data_gathering.google_trends import (
    GoogleTrendsQuery,
    default_keywords,
    fetch_interest_over_time,
    fetch_hourly_interest_stitched,
    timeframe_from_start_end,
)


def _parse_dt(s: str) -> datetime:
    # Accept YYYY-MM-DD or full ISO; assume UTC if no tz.
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.strptime(s, "%Y-%m-%d")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def cmd_google_trends(args: argparse.Namespace) -> None:
    keywords: Sequence[str] = args.keywords or default_keywords()

    if args.hourly:
        start = (
            _parse_dt(args.start)
            if args.start
            else datetime(2016, 1, 1, tzinfo=timezone.utc)
        )
        end = _parse_dt(args.end) if args.end else datetime.now(tz=timezone.utc)

        out_path = Path(args.out)
        existing: pd.DataFrame | None = None
        if args.resume and out_path.exists():
            existing = pd.read_csv(out_path)
            if "open_time_ms" in existing.columns and not existing.empty:
                last_ms = int(
                    pd.to_numeric(existing["open_time_ms"], errors="coerce").max()
                )
                # Back up by overlap to preserve normalization continuity.
                start = datetime.fromtimestamp(
                    max(0, (last_ms // 1000) - args.overlap_hours * 3600),
                    tz=timezone.utc,
                )

        df = fetch_hourly_interest_stitched(
            keywords=list(keywords),
            start=start,
            end=end,
            geo=args.geo,
            gprop=args.gprop,
            hl=args.hl,
            tz=args.tz,
            window_days=args.window_days,
            overlap_hours=args.overlap_hours,
            sleep_s=args.sleep_s,
            max_retries=args.max_retries,
            renormalize_0_100=args.renormalize_0_100,
        )

        if existing is not None and not existing.empty:
            df = pd.concat([existing, df], ignore_index=True)
            if "open_time_ms" in df.columns:
                df = df.drop_duplicates(subset=["open_time_ms"], keep="last")
                df = df.sort_values("open_time_ms").reset_index(drop=True)
    else:
        if args.timeframe:
            timeframe = args.timeframe
        else:
            if not args.start or not args.end:
                raise SystemExit("Provide --timeframe or both --start and --end")
            timeframe = timeframe_from_start_end(
                _parse_dt(args.start), _parse_dt(args.end)
            )

        query = GoogleTrendsQuery(
            keywords=list(keywords),
            timeframe=timeframe,
            geo=args.geo,
            gprop=args.gprop,
            hl=args.hl,
            tz=args.tz,
        )

        df = fetch_interest_over_time(query)

    out_path = Path(args.out)
    _write_csv(df, out_path)
    print(f"Wrote {len(df)} rows -> {out_path}")


def cmd_fear_greed(args: argparse.Namespace) -> None:
    df = fetch_fear_greed_index(FearGreedQuery(limit=args.limit))
    out_path = Path(args.out)
    _write_csv(df, out_path)
    print(f"Wrote {len(df)} rows -> {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download sentiment/signal data (Google Trends, Fear & Greed)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_gt = sub.add_parser(
        "google-trends", help="Download Google Trends interest-over-time"
    )
    p_gt.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help='Keywords, e.g. --keywords "cryptocurrency" "bitcoin"',
    )
    p_gt.add_argument(
        "--timeframe",
        default=None,
        help='pytrends timeframe, e.g. "now 7-d" (often hourly) or "today 12-m"',
    )
    p_gt.add_argument(
        "--start", default=None, help="Start date/time (YYYY-MM-DD or ISO)"
    )
    p_gt.add_argument("--end", default=None, help="End date/time (YYYY-MM-DD or ISO)")
    p_gt.add_argument(
        "--hourly",
        action="store_true",
        help=(
            "Attempt hourly backfill by stitching many short windows (slow; may rate-limit). "
            "Note: Google often only provides hourly resolution for recent ranges; "
            "historical ranges may fail."
        ),
    )
    p_gt.add_argument(
        "--resume",
        action="store_true",
        help="In --hourly mode, resume from existing --out file if present.",
    )
    p_gt.add_argument(
        "--window-days",
        type=int,
        default=7,
        help="Window size used for --hourly (default: 7).",
    )
    p_gt.add_argument(
        "--overlap-hours",
        type=int,
        default=24,
        help="Overlap used for --hourly normalization (default: 24).",
    )
    p_gt.add_argument(
        "--sleep-s",
        type=float,
        default=1.0,
        help="Sleep between requests in --hourly mode (default: 1.0).",
    )
    p_gt.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries per window in --hourly mode (default: 5).",
    )
    p_gt.add_argument(
        "--renormalize-0-100",
        action="store_true",
        help="After stitching, rescale each keyword to max=100 over the full period.",
    )
    p_gt.add_argument(
        "--geo", default="", help="Geo code, e.g. US, GB, or empty for world"
    )
    p_gt.add_argument(
        "--gprop",
        default="",
        help="Google property: images, news, youtube, or empty for web",
    )
    p_gt.add_argument("--hl", default="en-US", help="Host language")
    p_gt.add_argument(
        "--tz", type=int, default=0, help="Timezone offset in minutes (0=UTC)"
    )
    p_gt.add_argument(
        "--out",
        default="data/sentiment/google_trends/cryptocurrency.csv",
        help="Output CSV path",
    )
    p_gt.set_defaults(func=cmd_google_trends)

    p_fg = sub.add_parser(
        "fear-greed", help="Download Alternative.me Fear & Greed index"
    )
    p_fg.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of rows (0 = all available)",
    )
    p_fg.add_argument(
        "--out",
        default="data/sentiment/fear_greed.csv",
        help="Output CSV path",
    )
    p_fg.set_defaults(func=cmd_fear_greed)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
