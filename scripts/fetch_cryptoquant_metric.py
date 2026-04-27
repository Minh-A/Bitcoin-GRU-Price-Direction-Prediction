"""Fetch a CryptoQuant metric with an API token from the environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=(
            "https://api.cryptoquant.com/v1/btc/exchange-flows/reserve"
            "?exchange=all_exchange&window=day&from=20170101&to=20210522&limit=100000"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("data/cryptoquant_metric.csv"))
    parser.add_argument("--token-env", default="CRYPTOQUANT_API_TOKEN")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    token = os.getenv(args.token_env)
    if not token:
        raise RuntimeError(f"Set {args.token_env} before calling the CryptoQuant API.")

    response = requests.get(
        args.url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    frame = pd.DataFrame(payload["result"]["data"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame):,} rows to {args.output}")


if __name__ == "__main__":
    main(parse_args())
