"""
Behavioural overlay tests: relative sizing tiers + signal-stability / hold
discipline. These are guidance helpers, not financial advice.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.scoring import (
    _size_tier,
    _signal_stability,
    _action_note,
    _apply_hysteresis,
    _TIER_ORDER,
)


class TestSizeTier:
    def test_conviction_sets_base_tier(self):
        assert _size_tier("STRONG BUY", 0.80, 0.20, 1.0)[0] == "Large"
        assert _size_tier("BUY",        0.60, 0.20, 1.0)[0] == "Medium"

    def test_non_buys(self):
        assert _size_tier("SELL", 0.10, 0.20, 1.0)[0] == "Avoid"
        assert _size_tier("HOLD", 0.45, 0.20, 1.0)[0] == "—"

    def test_high_volatility_downsizes(self):
        # Same conviction, higher vol -> smaller-or-equal tier (monotonic).
        low  = _size_tier("STRONG BUY", 0.80, 0.20, 1.0)[0]
        high = _size_tier("STRONG BUY", 0.80, 0.60, 1.0)[0]
        assert _TIER_ORDER.index(high) <= _TIER_ORDER.index(low)
        assert high == "Medium"  # Large downsized one notch

    def test_high_beta_downsizes(self):
        base = _size_tier("BUY", 0.60, 0.20, 1.0)[0]
        hi   = _size_tier("BUY", 0.60, 0.20, 2.0)[0]
        assert _TIER_ORDER.index(hi) <= _TIER_ORDER.index(base)

    def test_high_vol_has_explanatory_note(self):
        _, note = _size_tier("STRONG BUY", 0.80, 0.60, 1.0)
        assert note and "volatility" in note.lower()


class TestSignalStability:
    def test_borderline_near_threshold(self):
        # default BUY edge = 0.54
        label, note = _signal_stability(0.545, None)
        assert label == "Borderline"
        assert note is not None

    def test_firm_away_from_threshold(self):
        label, note = _signal_stability(0.80, None)
        assert label == "Firm"
        assert note is None


class TestActionNote:
    def test_hold_is_reframed_as_no_action(self):
        note = _action_note("HOLD")
        assert "no action" in note.lower()
        assert "sell" in note.lower()  # explicitly discourages selling on this alone


class TestHysteresis:
    """Stateful hold discipline. Defaults: hold=0.38, grace floor=0.33."""
    BUY = {"signal": "BUY", "date": "12 Jun 2026"}

    def test_no_prior_is_passthrough(self):
        assert _apply_hysteresis("SELL", 0.36, None, None) == ("SELL", None, False)

    def test_fresh_sell_on_recent_buy_softens_to_hold(self):
        sig, note, applied = _apply_hysteresis("SELL", 0.36, self.BUY, None)
        assert sig == "HOLD" and applied is True
        assert note and "holding" in note.lower()
        assert "12 Jun 2026" in note

    def test_deep_sell_on_recent_buy_stays_sell(self):
        sig, note, applied = _apply_hysteresis("SELL", 0.20, self.BUY, None)
        assert sig == "SELL" and applied is False
        assert note and "weakened" in note.lower()

    def test_hold_on_recent_buy_is_stay_invested(self):
        sig, note, applied = _apply_hysteresis("HOLD", 0.45, self.BUY, None)
        assert sig == "HOLD" and applied is False
        assert note and "intact" in note.lower()

    def test_prior_non_buy_never_grants_grace(self):
        assert _apply_hysteresis("SELL", 0.36, {"signal": "HOLD"}, None) == ("SELL", None, False)

    def test_never_hardens_or_upgrades(self):
        # A still-strong signal is left untouched even with a prior buy.
        assert _apply_hysteresis("STRONG BUY", 0.80, self.BUY, None) == ("STRONG BUY", None, False)
