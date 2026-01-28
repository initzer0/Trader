from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

# Allow running this script directly without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trader.data_gathering.binance_ohlc import (
    create_binance_client,
    update_ohlc_csv,
)  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Binance OHLC history to CSV."
    )
    parser.add_argument(
        "--interval",
        default="1h",
        help="Binance kline interval (e.g. 1m, 5m, 15m, 1h, 4h, 1d). Default: 1h",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start datetime ISO (UTC). Example: 2017-01-01T00:00:00+00:00. Default: earliest on Binance.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End datetime ISO (UTC). Default: now.",
    )
    parser.add_argument(
        "--env",
        default=str(PROJECT_ROOT / ".env"),
        help="Path to .env file. Default: ./.env",
    )
    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start) if args.start else None
    end_dt = datetime.fromisoformat(args.end) if args.end else None
    if start_dt and start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt and end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    client = create_binance_client(env_path=args.env)

    coins = {
        "btc": "BTCUSDT",
        "eth": "ETHUSDT",
    }

    base_out = PROJECT_ROOT / "data" / "crypto"

    for coin, symbol in coins.items():
        out_dir = base_out / coin
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{symbol}_{args.interval}.csv"
        print(f"Downloading {symbol} {args.interval} -> {out_csv}")
        update_ohlc_csv(
            client=client,
            symbol=symbol,
            interval=args.interval,
            csv_path=out_csv,
            start=start_dt,
            end=end_dt,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
