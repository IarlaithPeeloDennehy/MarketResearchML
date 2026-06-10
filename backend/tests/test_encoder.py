"""
Tests for the offline-pretrained price encoder integration.

Two layers:
  1. Torch-dependent tests (skipped when torch is absent) — exercise the real
     TS2Vec encoder: save/load round-trip, embedding determinism, causality.
  2. Stub-encoder tests (no torch needed) — exercise the *wiring* in
     model.py / feature_engineering.py via a lightweight deterministic stub
     installed with set_encoder(). These run everywhere, including CI without
     torch, and guard the look-ahead / fallback invariants.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml import embedding_features as ef
from ml import channels as ch
from ml.embedding_features import (
    set_encoder,
    encoder_enabled,
    embedding_columns,
    point_in_time_embedding,
    standardize_emb_columns,
)
from ml.feature_engineering import PRICE_FEATURE_COLS, FEATURE_COLS
from ml.model import NUMKTEnsemble

FUNDAMENTAL_COLS = [c for c in FEATURE_COLS if c not in PRICE_FEATURE_COLS]
FUNDAMENTAL_RANK_COLS = [f"{c}_rank" for c in FUNDAMENTAL_COLS]


# ── Lightweight deterministic stub (no torch) ───────────────────────────────
class _StubEncoder:
    """Mimics LoadedEncoder's surface: .embed_dim, .version, .channels,
    .embed_channels(raw_mat). Uses base channels only so tests don't depend on
    whatever ETF/volume data happens to be in the dev cache."""
    embed_dim = 4
    window = 252
    version = "stub-v1"
    channels = ["logret", "vol"]

    def embed_channels(self, mat) -> np.ndarray:
        mat = np.asarray(mat, dtype=float)
        if mat.size == 0:
            return np.zeros(self.embed_dim, dtype=np.float32)
        lr = mat[:, 0]
        feats = [float(lr.mean()), float(lr.std()),
                 float(lr[-1]), float(mat.shape[0])]
        return np.asarray(feats[: self.embed_dim], dtype=np.float32)


@pytest.fixture
def stub_encoder():
    """Install the stub for the duration of a test, then clear it."""
    set_encoder(_StubEncoder())
    try:
        yield
    finally:
        set_encoder(None)


@pytest.fixture(autouse=True)
def _clear_encoder():
    """Make sure no test leaks a global encoder into the next one."""
    set_encoder(None)
    ef._MEM_CACHE.clear()
    yield
    set_encoder(None)
    ef._MEM_CACHE.clear()


def _write_price_parquet(tmp_path: Path, ticker: str, n: int = 400, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    idx = pd.bdate_range(end="2024-12-31", periods=n)
    df = pd.DataFrame({"Close": prices, "Open": prices, "High": prices, "Low": prices}, index=idx)
    safe = ticker.replace(".", "_").replace("-", "_")
    df.to_parquet(tmp_path / f"{safe}.parquet")


# ════════════════════════════════════════════════════════════════════════════
# Fallback (no encoder) — must behave exactly as before pre-training existed
# ════════════════════════════════════════════════════════════════════════════
def test_disabled_is_noop():
    assert encoder_enabled() is False
    assert embedding_columns() == []
    s = pd.Series(100 * np.cumprod(1 + np.random.normal(0, 0.01, 300)),
                  index=pd.bdate_range(end="2024-12-31", periods=300))
    assert point_in_time_embedding(s, end_idx=200, ticker="AAPL") == {}


def test_labels_for_horizon_no_emb_when_disabled(tmp_path, monkeypatch):
    import ml.model as m
    for i, t in enumerate(["AAPL", "MSFT", "GOOG", "AMZN"]):
        _write_price_parquet(tmp_path, t, n=400, seed=i)
    monkeypatch.setattr(m, "PRICE_DIR", tmp_path)
    model = NUMKTEnsemble()
    rows, labels, fwd = model._labels_for_horizon(["AAPL", "MSFT", "GOOG", "AMZN"], horizon=63)
    assert len(rows) > 0
    assert not any(k.startswith("emb_") for r in rows for k in r), \
        "No emb_* columns should appear when the encoder is disabled"


# ════════════════════════════════════════════════════════════════════════════
# Stub-encoder wiring (no torch needed)
# ════════════════════════════════════════════════════════════════════════════
def test_embedding_columns_reflect_dim(stub_encoder):
    assert encoder_enabled() is True
    assert embedding_columns() == ["emb_0", "emb_1", "emb_2", "emb_3"]


def test_point_in_time_is_causal(stub_encoder):
    """Appending FUTURE bars must not change the embedding at a past end_idx."""
    base = pd.Series(
        100 * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, 150)),
        index=pd.bdate_range(end="2024-06-30", periods=150),
    )
    e1 = point_in_time_embedding(base, end_idx=99, ticker="X")
    # extend with 50 more future bars (same first 150 dates/values preserved)
    future = pd.Series(
        np.r_[base.values, 100 * np.cumprod(1 + np.random.default_rng(2).normal(0, 0.01, 50))],
        index=pd.bdate_range(end="2024-09-09", periods=200),
    )
    e2 = point_in_time_embedding(future, end_idx=99, ticker="Y")  # diff ticker → no cache hit
    assert e1 and e2
    assert e1 == e2, "Embedding at end_idx=99 changed when future bars were appended"


def test_mem_cache_hit(stub_encoder):
    s = pd.Series(100 * np.cumprod(1 + np.random.default_rng(3).normal(0, 0.01, 120)),
                  index=pd.bdate_range(end="2024-06-30", periods=120))
    a = point_in_time_embedding(s, end_idx=80, ticker="AAPL")
    b = point_in_time_embedding(s, end_idx=80, ticker="AAPL")
    assert a == b
    assert (("AAPL", str(s.index[80]), "stub-v1") in ef._MEM_CACHE)


def test_standardize_emb_columns():
    df = pd.DataFrame({"emb_0": [1.0, 2.0, 3.0], "emb_1": [5.0, 5.0, 5.0], "other": [1, 2, 3]})
    out = standardize_emb_columns(df.copy())
    assert abs(out["emb_0"].mean()) < 1e-9
    assert abs(out["emb_0"].std(ddof=0) - 1.0) < 1e-6 or abs(out["emb_0"].std() - 1.0) < 1e-6
    assert (out["emb_1"] == 0.0).all()          # constant column → 0
    assert list(out["other"]) == [1, 2, 3]      # untouched


def test_labels_for_horizon_has_emb_and_no_fundamentals(tmp_path, monkeypatch, stub_encoder):
    """Integrity: historical rows carry price ranks + emb_*, never fundamentals."""
    import ml.model as m
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
    for i, t in enumerate(tickers):
        _write_price_parquet(tmp_path, t, n=400, seed=i)
    monkeypatch.setattr(m, "PRICE_DIR", tmp_path)
    model = NUMKTEnsemble()
    rows, labels, fwd = model._labels_for_horizon(tickers, horizon=63)
    assert len(rows) > 0
    keys = set().union(*[set(r) for r in rows])
    # embeddings present
    assert {"emb_0", "emb_1", "emb_2", "emb_3"}.issubset(keys)
    # price ranks present
    assert any(k.endswith("_rank") for k in keys)
    # fundamentals absent (look-ahead invariant preserved)
    assert not (keys & set(FUNDAMENTAL_RANK_COLS)), \
        f"Fundamental rank cols leaked into historical rows: {keys & set(FUNDAMENTAL_RANK_COLS)}"


def test_select_rank_cols_includes_emb(stub_encoder):
    model = NUMKTEnsemble()
    df = pd.DataFrame({
        "mom_12m_rank": [0.1, 0.9],
        "rsi_14_rank": [0.5, 0.5],
        "emb_0": [0.1, -0.1],
        "emb_1": [0.2, -0.2],
    })
    cols = model._select_rank_cols(df)
    assert "emb_0" in cols and "emb_1" in cols
    assert "mom_12m_rank" in cols
    # emb cols ordered numerically and after price ranks
    assert cols.index("emb_0") < cols.index("emb_1")


def test_orchestration_loads_inputs_and_activates_channels(tmp_path, monkeypatch):
    """The by-ticker loaders pull volume/sector/reference/series from the cache,
    and those inputs make the Group A/B channels non-neutral."""
    import json
    from ml import series_cache

    pdir, idir, sdir = tmp_path / "prices", tmp_path / "info", tmp_path / "series"
    for d in (pdir, idir, sdir):
        d.mkdir()
    monkeypatch.setattr(ef, "_PRICE_DIR", pdir)
    monkeypatch.setattr(ef, "_INFO_DIR", idir)
    monkeypatch.setattr(series_cache, "_SERIES_DIR", sdir)

    idx = pd.bdate_range(end="2024-12-31", periods=300)

    def _write(tk, seed, with_vol=False):
        px = 100 * np.cumprod(1 + np.random.default_rng(seed).normal(0, 0.01, 300))
        data = {"Close": px}
        if with_vol:
            data["Volume"] = np.random.default_rng(seed + 1).integers(1_000_000, 5_000_000, 300).astype(float)
        pd.DataFrame(data, index=idx).to_parquet(pdir / f"{tk}.parquet")

    _write("AAPL", 1, with_vol=True)
    _write("SPY", 2)
    _write("XLK", 3)
    (idir / "AAPL.json").write_text(json.dumps({"sector": "Technology"}))
    series_cache.save(
        "AAPL",
        [{"period": "2024-06-30", "surprisePercent": 40.0}],
        [{"period": "2024-06-01", "strongBuy": 3, "buy": 5, "hold": 1, "sell": 0, "strongSell": 0}],
    )
    ef._reset_input_caches()

    # loaders find the files
    assert ef._load_volume("AAPL") is not None
    assert ef._load_reference("SPY") is not None
    assert ef._load_sector("AAPL") == "Technology"
    earn, ana = ef._load_history("AAPL")
    assert earn and ana

    # channels built from loaded inputs are non-neutral where data exists
    close = pd.read_parquet(pdir / "AAPL.parquet")["Close"]
    spec = ch.ALL_CHANNELS
    mat = ch.build_window_channels(
        close, spec, end_idx=len(close) - 1, window=120,
        volume=ef._load_volume("AAPL"),
        market_close=ef._load_reference("SPY"),
        sector_close=ef._load_reference("XLK"),
        earnings_hist=earn, analyst_hist=ana,
    )
    col = {name: i for i, name in enumerate(spec)}
    assert np.any(mat[:, col["rel_volume"]] != 0), "volume channel should be active"
    assert np.any(mat[:, col["excess_ret_vs_market"]] != 0), "market channel should be active"
    assert np.any(mat[:, col["earn_surprise_pulse"]] != 0), "earnings pulse should fire"


def test_input_cache_invalidates_on_mtime(tmp_path, monkeypatch):
    """Rewriting a ticker's parquet (cache refresh) must invalidate the loader."""
    pdir = tmp_path / "prices"
    pdir.mkdir()
    monkeypatch.setattr(ef, "_PRICE_DIR", pdir)
    ef._reset_input_caches()
    idx = pd.bdate_range(end="2024-12-31", periods=100)
    pd.DataFrame({"Close": np.ones(100), "Volume": np.ones(100)}, index=idx).to_parquet(pdir / "AAPL.parquet")
    first = ef._load_volume("AAPL")
    assert len(first) == 100
    # rewrite with more bars and a newer mtime
    idx2 = pd.bdate_range(end="2024-12-31", periods=150)
    import os, time
    pd.DataFrame({"Close": np.ones(150), "Volume": np.ones(150)}, index=idx2).to_parquet(pdir / "AAPL.parquet")
    os.utime(pdir / "AAPL.parquet", (time.time() + 10, time.time() + 10))
    assert len(ef._load_volume("AAPL")) == 150, "loader must pick up the refreshed file"


# ════════════════════════════════════════════════════════════════════════════
# Real torch encoder (skipped when torch unavailable)
# ════════════════════════════════════════════════════════════════════════════
def test_real_encoder_roundtrip_and_determinism(tmp_path):
    torch = pytest.importorskip("torch")
    from ml.encoder import EncoderConfig, TS2VecEncoder, save_checkpoint, load_encoder

    torch.manual_seed(0)
    cfg = EncoderConfig(n_channels=2, window=64, hidden_dim=8, depth=2, embed_dim=4)
    model = TS2VecEncoder(cfg)
    ckpt = tmp_path / "encoder.pt"
    save_checkpoint(model, cfg, {"version": "test-v1", "pretrain_cutoff_date": "2020-01-01",
                                 "channels": ["logret", "vol"], "norm": "per_window"}, ckpt)

    enc = load_encoder(ckpt)
    assert enc is not None
    assert enc.embed_dim == 4 and enc.version == "test-v1"

    close = pd.Series(100 * np.cumprod(1 + np.random.default_rng(7).normal(0, 0.01, 200)),
                      index=pd.bdate_range(end="2024-12-31", periods=200))
    mat = ch.build_window_channels(close, enc.channels, end_idx=len(close) - 1, window=enc.window)
    e1 = enc.embed_channels(mat)
    e2 = enc.embed_channels(mat)
    assert e1.shape == (4,)
    np.testing.assert_array_equal(e1, e2)   # deterministic (eval mode, no dropout)


def test_real_encoder_window_causality(tmp_path):
    torch = pytest.importorskip("torch")
    from ml.encoder import EncoderConfig, TS2VecEncoder, save_checkpoint, load_encoder

    torch.manual_seed(0)
    cfg = EncoderConfig(n_channels=2, window=64, hidden_dim=8, depth=2, embed_dim=4)
    save_checkpoint(TS2VecEncoder(cfg), cfg, {"version": "test-v2", "channels": ["logret", "vol"]},
                    tmp_path / "e.pt")
    enc = load_encoder(tmp_path / "e.pt")

    full = pd.Series(100 * np.cumprod(1 + np.random.default_rng(9).normal(0, 0.01, 300)),
                     index=pd.bdate_range(end="2024-12-31", periods=300))
    # embedding the window ending at idx 119 must ignore bars after it
    mat_a = ch.build_window_channels(full, enc.channels, end_idx=119, window=enc.window)
    mat_b = ch.build_window_channels(full.iloc[:160], enc.channels, end_idx=119, window=enc.window)
    np.testing.assert_array_equal(enc.embed_channels(mat_a), enc.embed_channels(mat_b))


def test_load_encoder_missing_returns_none():
    from ml.encoder import load_encoder
    assert load_encoder("definitely_not_here.pt") is None
