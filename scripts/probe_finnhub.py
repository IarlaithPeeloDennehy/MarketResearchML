#!/usr/bin/env python
"""
probe_finnhub.py — confirm the Finnhub key returns the history Group B needs.

Group B channels (earn_surprise_pulse, analyst_net_buy_ts) depend on
`company_earnings` and `recommendation_trends` returning multi-period HISTORY,
not just the latest value. Free-tier coverage varies, so probe before building a
corpus around them.

    FINNHUB_API_KEY=... python scripts/probe_finnhub.py AAPL MSFT JPM
"""
from __future__ import annotations

import os
import sys


def main(tickers: list[str]):
    key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if not key:
        sys.exit("FINNHUB_API_KEY is not set.")
    try:
        import finnhub
    except ImportError:
        sys.exit("pip install finnhub-python")

    client = finnhub.Client(api_key=key)
    print(f"{'ticker':8} {'earnings_hist':>14} {'analyst_months':>15} {'basic_series?':>14}")
    for t in tickers:
        try:
            earn = client.company_earnings(t, limit=24) or []
        except Exception as e:
            earn = []
            print(f"{t:8} earnings error: {e}")
        try:
            rec = client.recommendation_trends(t) or []
        except Exception as e:
            rec = []
            print(f"{t:8} recommendation error: {e}")
        try:
            bf = client.company_basic_financials(t, "all") or {}
            has_series = bool(bf.get("series"))   # the discarded fundamentals TS
        except Exception:
            has_series = False
        n_earn = sum(1 for e in earn if e.get("surprisePercent") is not None)
        n_rec = sum(1 for r in rec if r.get("period"))
        print(f"{t:8} {n_earn:>14} {n_rec:>15} {str(has_series):>14}")
    print("\nGroup B is viable if earnings_hist and analyst_months are both > ~4.")


if __name__ == "__main__":
    main(sys.argv[1:] or ["AAPL", "MSFT", "JPM"])
