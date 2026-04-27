"""Fetch Upbit KRW-BTC OHLCV data with pyupbit."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyupbit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="KRW-BTC")
    parser.add_argument("--interval", default="day")
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=Path("data/upbit_krw_btc.csv"))
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    frame = pyupbit.get_ohlcv(args.ticker, interval=args.interval, count=args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output)
    print(f"Wrote {len(frame):,} rows to {args.output}")


if __name__ == "__main__":
    main(parse_args())
