"""
Diversification engine tests (pure functions — no DB/network).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.diversification import (
    normalize_sector,
    portfolio_composition,
    concentration_flags,
    recommend_additions,
    ANCHOR_SECTORS,
    CANONICAL_SECTORS,
)


class TestNormalizeSector:
    def test_anchor_ticker_uses_static_map(self):
        assert normalize_sector("whatever", "AAPL") == "Technology"
        assert normalize_sector(None, "JNJ") == "Healthcare"
        assert normalize_sector("garbage", "GOOGL") == "Communication Services"

    def test_yfinance_exact_sectors(self):
        assert normalize_sector("Financial Services") == "Financials"
        assert normalize_sector("Consumer Cyclical") == "Consumer Discretionary"
        assert normalize_sector("Basic Materials") == "Materials"

    def test_finnhub_keyword_fallback(self):
        assert normalize_sector("Pharmaceuticals") == "Healthcare"
        assert normalize_sector("Semiconductors") == "Technology"
        assert normalize_sector("Oil & Gas Refining") == "Energy"
        assert normalize_sector("Banking") == "Financials"

    def test_unmappable_is_unknown(self):
        assert normalize_sector("") == "Unknown"
        assert normalize_sector(None) == "Unknown"
        assert normalize_sector("Quantum Flux Widgets") == "Unknown"

    def test_every_anchor_sector_is_canonical(self):
        assert set(ANCHOR_SECTORS.values()).issubset(set(CANONICAL_SECTORS))


class TestComposition:
    def test_equal_weight_when_no_weights(self):
        h = [{"sector": "Technology"}, {"sector": "Technology"}, {"sector": "Healthcare"}]
        comp = portfolio_composition(h)
        assert round(comp["Technology"]) == 67
        assert round(comp["Healthcare"]) == 33

    def test_explicit_weights_normalised_to_100(self):
        h = [{"sector": "Technology", "weight": 30},
             {"sector": "Energy", "weight": 10}]
        comp = portfolio_composition(h)
        assert round(comp["Technology"]) == 75
        assert round(comp["Energy"]) == 25
        assert round(sum(comp.values())) == 100

    def test_partial_weights_fall_back_to_equal(self):
        h = [{"sector": "Technology", "weight": 90}, {"sector": "Energy"}]  # one missing
        comp = portfolio_composition(h)
        assert comp["Technology"] == 50.0 and comp["Energy"] == 50.0

    def test_empty(self):
        assert portfolio_composition([]) == {}


class TestFlags:
    def test_over_concentration_flagged(self):
        flags = concentration_flags({"Technology": 80.0, "Healthcare": 20.0})
        over = [f for f in flags if f["type"] == "over"]
        assert any(f["sector"] == "Technology" for f in over)

    def test_missing_core_sector_gap(self):
        # all in Tech -> Healthcare/Financials/etc are 0% gaps
        flags = concentration_flags({"Technology": 100.0})
        gaps = {f["sector"] for f in flags if f["type"] == "gap"}
        assert "Healthcare" in gaps and "Financials" in gaps

    def test_balanced_has_no_over_flag(self):
        comp = {s: 100.0 / len(CANONICAL_SECTORS) for s in CANONICAL_SECTORS}
        assert not [f for f in concentration_flags(comp) if f["type"] == "over"]


class TestRecommend:
    CANDS = [
        {"ticker": "JNJ", "sector": "Healthcare", "composite_score": 72, "signal": "BUY"},
        {"ticker": "PFE", "sector": "Healthcare", "composite_score": 60, "signal": "HOLD"},
        {"ticker": "XOM", "sector": "Energy",     "composite_score": 68, "signal": "BUY"},
        {"ticker": "MSFT","sector": "Technology", "composite_score": 90, "signal": "STRONG BUY"},
    ]

    def test_excludes_held(self):
        comp = {"Technology": 100.0}
        recs = recommend_additions(comp, self.CANDS, held=["JNJ"], n=5)
        assert all(r["ticker"] != "JNJ" for r in recs)

    def test_prefers_underallocated_sectors(self):
        # Tech is saturated; recommendations should be non-Tech gap fillers.
        comp = {"Technology": 100.0}
        recs = recommend_additions(comp, self.CANDS, held=[], n=5)
        assert recs and all(r["sector"] != "Technology" for r in recs)
        assert all("Adds" in r["rationale"] for r in recs)

    def test_higher_score_wins_within_sector(self):
        comp = {"Technology": 100.0}
        recs = recommend_additions(comp, self.CANDS, held=[], n=5, per_sector=1)
        hc = [r for r in recs if r["sector"] == "Healthcare"]
        assert hc and hc[0]["ticker"] == "JNJ"   # 72 beats PFE 60

    def test_per_sector_cap(self):
        comp = {"Technology": 100.0}
        recs = recommend_additions(comp, self.CANDS, held=[], n=5, per_sector=1)
        sectors = [r["sector"] for r in recs]
        assert len(sectors) == len(set(sectors))   # no sector repeats
