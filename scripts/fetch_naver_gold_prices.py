"""Fetch international gold price pages from Naver Finance."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=120)
    parser.add_argument("--output", type=Path, default=Path("data/naver_gold_prices.csv"))
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    frames = []
    for page in range(1, args.pages + 1):
        url = (
            "https://finance.naver.com/marketindex/worldDailyQuote.nhn"
            f"?marketindexCd=CMDT_GC&fdtc=2&page={page}"
        )
        frames.append(pd.read_html(url)[0])

    frame = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame):,} rows to {args.output}")


if __name__ == "__main__":
    main(parse_args())
