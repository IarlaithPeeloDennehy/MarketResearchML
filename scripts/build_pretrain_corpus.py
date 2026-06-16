#!/usr/bin/env python
"""
build_pretrain_corpus.py — OFFLINE corpus builder for encoder pre-training.

Reuses the PRODUCTION fetch path (ml.data_fetcher → Finnhub) so the corpus is
byte-structurally identical to live data: price parquets with Close+Volume, info
JSON with sector (finnhubIndustry), and series JSON with full earnings/analyst
history. This is what guarantees train/inference channel parity — the offline
trainer and production read the SAME cache layout and build channels through the
SAME channels.py.

Per the design: the corpus is restricted to **Finnhub US tickers** (UK/IE .L/.IR
tickers lack the Finnhub series, so they're skipped). Reference ETFs (SPY + sector
SPDRs) are always included so the market/sector channels have their references.

Run on the training box with FINNHUB_API_KEY set:
    python scripts/build_pretrain_corpus.py \
        --out-dir data/corpus_cache --years 18 --tickers-file sp1500.txt
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("corpus")

_DEFAULT = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AMD", "INTC", "CRM", "ORCL",
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "BMY", "AMGN", "GILD",
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "V", "MA",
    "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "WMT", "PG", "KO", "PEP",
    "CAT", "HON", "UPS", "GE", "BA", "LMT", "XOM", "CVX", "SLB",
]


def load_tickers(path: str | None) -> list[str]:
    if not path:
        return _DEFAULT
    lines = Path(path).read_text().splitlines()
    return [ln.strip().upper() for ln in lines if ln.strip() and not ln.startswith("#")]


async def _run(tickers: list[str], years: int, batch: int):
    # Imported AFTER CACHE_DIR/FINNHUB_API_KEY are set in os.environ.
    from ml.data_fetcher import fetch_multiple_stocks
    ok = 0
    for i in range(0, len(tickers), batch):
        chunk = tickers[i: i + batch]
        out, _ = await fetch_multiple_stocks(chunk, lookback_years=years)
        ok += len(out)
        logger.info(f"[{i + len(chunk)}/{len(tickers)}] cumulative loaded: {ok}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Build a Finnhub-US corpus for pre-training")
    ap.add_argument("--out-dir", default="data/corpus_cache",
                    help="becomes CACHE_DIR: prices/, info/, series/ are written here")
    ap.add_argument("--tickers-file", default=None, help="one US symbol per line")
    ap.add_argument("--years", type=int, default=18)
    ap.add_argument("--finnhub-key", default=None, help="overrides FINNHUB_API_KEY env")
    ap.add_argument("--batch", type=int, default=20)
    args = ap.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # data_fetcher reads CACHE_DIR at import time → set BEFORE importing it.
    os.environ["CACHE_DIR"] = str(out)
    if args.finnhub_key:
        os.environ["FINNHUB_API_KEY"] = args.finnhub_key
    if not os.environ.get("FINNHUB_API_KEY"):
        logger.warning("FINNHUB_API_KEY not set — US tickers will fall back to Yahoo and "
                       "Group B (earnings/analyst) series will be MISSING. Set the key.")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from ml.data_fetcher import _is_us_ticker
    from ml.channels import REFERENCE_TICKERS

    raw = load_tickers(args.tickers_file)
    us = [t for t in raw if _is_us_ticker(t)]
    skipped = [t for t in raw if not _is_us_ticker(t)]
    if skipped:
        logger.info(f"Skipping {len(skipped)} non-Finnhub (UK/IE) tickers: {skipped[:10]}…")
    # Reference ETFs must be present for market/sector channels.
    tickers = sorted(set(us) | set(REFERENCE_TICKERS))
    logger.info(f"Fetching {len(tickers)} tickers (incl {len(REFERENCE_TICKERS)} reference ETFs) "
                f"into {out}")

    ok = asyncio.run(_run(tickers, args.years, args.batch))
    logger.info(f"Corpus build complete: {ok}/{len(tickers)} cached under {out}. "
                f"Now run: python scripts/pretrain_encoder.py --corpus-dir {out}")


if __name__ == "__main__":
    main()
