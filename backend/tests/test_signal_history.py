"""
signal_history pure-function tests (no DB).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_history import build_signal_payload, extract_prior_signals


class TestBuildPayload:
    def test_builds_ticker_signal_map(self):
        results = [
            {"ticker": "AAPL", "signal": "BUY",  "composite_score": 71.0},
            {"ticker": "KO",   "signal": "SELL", "composite_score": 32.0},
            {"ticker": "NOSIG"},                       # no signal -> skipped
        ]
        p = build_signal_payload(results)
        assert p == {"signals": {
            "AAPL": {"signal": "BUY",  "score": 71.0},
            "KO":   {"signal": "SELL", "score": 32.0},
        }}

    def test_empty(self):
        assert build_signal_payload([]) == {"signals": {}}


class TestExtractPrior:
    def test_picks_most_recent_per_ticker(self):
        # events MUST be most-recent-first
        events = [
            ({"signals": {"AAPL": {"signal": "HOLD", "score": 50}}}, "2026-06-15T10:00:00"),
            ({"signals": {"AAPL": {"signal": "BUY",  "score": 70},
                          "KO":   {"signal": "SELL", "score": 30}}}, "2026-06-01T10:00:00"),
        ]
        out = extract_prior_signals(events, ["AAPL", "KO"])
        assert out["AAPL"]["signal"] == "HOLD"          # newer event wins
        assert out["KO"]["signal"] == "SELL"            # only in older event
        assert out["AAPL"]["date"] == "15 Jun 2026"

    def test_ignores_unrequested_tickers(self):
        events = [({"signals": {"MSFT": {"signal": "BUY", "score": 80}}}, "2026-06-10T00:00:00")]
        assert extract_prior_signals(events, ["AAPL"]) == {}

    def test_missing_ticker_absent(self):
        events = [({"signals": {}}, "2026-06-10T00:00:00")]
        assert extract_prior_signals(events, ["AAPL"]) == {}
