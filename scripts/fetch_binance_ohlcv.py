"""Fetch daily Binance BTC/USDT OHLCV data."""

from __future__ import annotations

import argparse
from pathlib import Path

import ccxt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--since", default="2017-08-17T00:00:00Z")
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=Path("data/binance_btc_usdt.csv"))
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    exchange = ccxt.binance()
    since_ms = exchange.parse8601(args.since)
    rows = exchange.fetch_ohlcv(args.symbol, args.timeframe, since=since_ms, limit=args.limit)

    frame = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
    frame["datetime"] = pd.to_datetime(frame["datetime"], unit="ms")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame):,} rows to {args.output}")


if __name__ == "__main__":
    main(parse_args())
