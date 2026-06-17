"""
diversification.py
──────────────────
Sector-based portfolio diversification: compose a user's holdings by sector,
flag over-concentration and missing core sectors, and recommend BUY-rated
additions that fill the gaps.

Design notes
- ONE canonical sector taxonomy (GICS-ish, 11 buckets) is used for BOTH the
  user's held stocks (sectors come live from the fetcher, normalised here) and
  the candidate universe (the curated anchors, tagged statically below). Using
  the same taxonomy on both sides is what makes "you hold 0% Healthcare" honest.
- The candidate universe is the ~165 cached US anchors (geography is out of
  scope by product decision). Candidates are scored with the SAME model/pipeline
  as /analyse — no new modelling.
- The pure functions (normalize_sector, portfolio_composition,
  concentration_flags, recommend_additions) take plain dicts and are unit-tested
  without any DB/network. score_candidates is the only data-touching helper.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

CANONICAL_SECTORS = [
    "Technology", "Communication Services", "Healthcare", "Financials",
    "Consumer Discretionary", "Consumer Staples", "Industrials", "Energy",
    "Utilities", "Materials", "Real Estate",
]

# Core sectors a reasonably balanced equity portfolio is expected to touch.
# A 0% allocation to any of these is surfaced as a gap.
_CORE_SECTORS = [
    "Technology", "Healthcare", "Financials", "Consumer Staples", "Industrials",
]

_OVER_CONCENTRATION_PCT = 40.0   # single-sector weight above this is flagged

# Canonical sector per anchor ticker (transcribed from model._ANCHOR_TICKERS,
# corrected to GICS where the grouping there was a loose convenience label —
# e.g. GOOGL/META/NFLX → Communication Services, AMZN → Consumer Discretionary).
# ETFs (XLK/XLF/XLV/XLE/XLI/IWM) are intentionally absent: never candidates.
_SECTOR_GROUPS: dict[str, list[str]] = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AMD", "INTC", "CRM", "ORCL", "CSCO", "IBM",
        "TXN", "QCOM", "NOW", "INTU", "AMAT", "MU", "ADI", "LRCX", "KLAC",
        "SNPS", "CDNS", "PANW", "AVGO", "ADBE", "UBER", "SMCI",
    ],
    "Communication Services": [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T", "CHTR",
    ],
    "Healthcare": [
        "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "BMY", "AMGN", "GILD",
        "LLY", "DHR", "ISRG", "VRTX", "REGN", "ZTS", "CI", "CVS", "HUM", "CNC",
        "MDT", "SYK", "BSX",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "V", "MA", "SCHW",
        "USB", "PNC", "TFC", "COF", "SPGI", "CME", "ICE", "AON", "MMC", "PGR",
        "TRV", "CB", "AIG", "MET", "PRU", "PYPL", "BRK-B",
    ],
    "Consumer Discretionary": [
        "AMZN", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX", "BKNG", "MAR",
        "GM", "F",
    ],
    "Consumer Staples": [
        "COST", "WMT", "PG", "KO", "PEP", "MDLZ", "CL", "KMB", "GIS", "KHC",
        "STZ", "MO", "PM", "MNST", "KDP", "CELH",
    ],
    "Industrials": [
        "CAT", "HON", "UPS", "GE", "BA", "LMT", "DE", "MMM", "EMR", "ETN",
        "ITW", "NSC", "CSX", "FDX", "GD", "NOC", "RTX",
    ],
    "Energy": [
        "XOM", "CVX", "SLB", "COP", "EOG", "MPC", "PSX", "VLO", "OXY", "KMI",
        "WMB", "HAL",
    ],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "PEG", "ED"],
    "Materials": ["LIN", "APD", "SHW", "ECL", "NEM", "FCX", "DOW", "NUE"],
    "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "SPG", "O", "PSA", "WELL", "DLR"],
}

ANCHOR_SECTORS: dict[str, str] = {
    t: sector for sector, tickers in _SECTOR_GROUPS.items() for t in tickers
}

# Map a fetched sector/industry string → canonical bucket. Exact yfinance
# sectors first, then keyword fallback for Finnhub's granular `finnhubIndustry`.
_YF_SECTOR_MAP = {
    "technology":             "Technology",
    "communication services": "Communication Services",
    "healthcare":             "Healthcare",
    "financial services":     "Financials",
    "consumer cyclical":      "Consumer Discretionary",
    "consumer defensive":     "Consumer Staples",
    "industrials":            "Industrials",
    "energy":                 "Energy",
    "utilities":              "Utilities",
    "basic materials":        "Materials",
    "real estate":            "Real Estate",
}

# Ordered keyword → bucket (first match wins; order matters for ambiguous words).
_KEYWORD_MAP = [
    ("real estate", "Real Estate"), ("reit", "Real Estate"),
    ("semiconduct", "Technology"), ("software", "Technology"),
    ("technology", "Technology"), ("hardware", "Technology"), ("electronic", "Technology"),
    ("telecom", "Communication Services"), ("media", "Communication Services"),
    ("communication", "Communication Services"), ("entertain", "Communication Services"),
    ("interactive", "Communication Services"),
    ("pharma", "Healthcare"), ("biotech", "Healthcare"), ("health", "Healthcare"),
    ("medical", "Healthcare"), ("drug", "Healthcare"), ("life science", "Healthcare"),
    ("bank", "Financials"), ("insur", "Financials"), ("financ", "Financials"),
    ("capital market", "Financials"), ("asset manage", "Financials"), ("payment", "Financials"),
    ("oil", "Energy"), ("gas", "Energy"), ("petroleum", "Energy"), ("energy", "Energy"),
    ("utilit", "Utilities"), ("electric util", "Utilities"), ("water util", "Utilities"),
    ("chemical", "Materials"), ("metal", "Materials"), ("mining", "Materials"),
    ("steel", "Materials"), ("material", "Materials"), ("paper", "Materials"),
    ("aerospace", "Industrials"), ("defense", "Industrials"), ("machinery", "Industrials"),
    ("airline", "Industrials"), ("transport", "Industrials"), ("logistic", "Industrials"),
    ("construction", "Industrials"), ("industrial", "Industrials"),
    ("food", "Consumer Staples"), ("beverage", "Consumer Staples"),
    ("tobacco", "Consumer Staples"), ("household", "Consumer Staples"),
    ("grocery", "Consumer Staples"), ("staple", "Consumer Staples"),
    ("retail", "Consumer Discretionary"), ("auto", "Consumer Discretionary"),
    ("apparel", "Consumer Discretionary"), ("leisure", "Consumer Discretionary"),
    ("hotel", "Consumer Discretionary"), ("restaurant", "Consumer Discretionary"),
    ("travel", "Consumer Discretionary"), ("discretionary", "Consumer Discretionary"),
    ("cyclical", "Consumer Discretionary"),
]


def normalize_sector(raw: str | None, ticker: str | None = None) -> str:
    """Canonical sector for a stock. Anchors use the static map (keeps held
    stocks and candidates on one taxonomy); otherwise normalise the fetched
    sector/industry string. Unmappable → 'Unknown' (never raises)."""
    if ticker and ticker.upper() in ANCHOR_SECTORS:
        return ANCHOR_SECTORS[ticker.upper()]
    if not raw:
        return "Unknown"
    s = str(raw).strip().lower()
    if s in _YF_SECTOR_MAP:
        return _YF_SECTOR_MAP[s]
    for kw, bucket in _KEYWORD_MAP:
        if kw in s:
            return bucket
    return "Unknown"


def portfolio_composition(holdings: list[dict]) -> dict[str, float]:
    """{sector: weight_pct} for the holdings. Uses provided weights only when
    EVERY holding has a positive weight; otherwise falls back to equal-weight
    (so a half-filled weight set can't silently mislead)."""
    n = len(holdings or [])
    if n == 0:
        return {}
    weights = [h.get("weight") for h in holdings]
    if all(isinstance(w, (int, float)) and w and w > 0 for w in weights):
        total = float(sum(weights))
        norm = [float(w) / total * 100.0 for w in weights]
    else:
        norm = [100.0 / n] * n

    comp: dict[str, float] = {}
    for h, w in zip(holdings, norm):
        sec = h.get("sector") or "Unknown"
        comp[sec] = comp.get(sec, 0.0) + w
    return {s: round(v, 1) for s, v in comp.items()}


def concentration_flags(composition: dict[str, float]) -> list[dict]:
    """Over-concentration (single sector > 40%) and missing-core-sector gaps."""
    flags: list[dict] = []
    known = {s: v for s, v in composition.items() if s != "Unknown"}

    for sec, pct in sorted(known.items(), key=lambda x: -x[1]):
        if pct > _OVER_CONCENTRATION_PCT:
            flags.append({
                "type": "over", "sector": sec, "pct": round(pct, 1),
                "message": f"{pct:.0f}% of your portfolio is in {sec} — heavy "
                           f"single-sector concentration raises your risk.",
            })

    for sec in _CORE_SECTORS:
        if composition.get(sec, 0.0) <= 0.0:
            flags.append({
                "type": "gap", "sector": sec, "pct": 0.0,
                "message": f"No {sec} exposure — a core sector most balanced "
                           f"portfolios hold.",
            })
    return flags


def recommend_additions(
    composition: dict[str, float],
    candidates: list[dict],
    held: list[str],
    n: int = 5,
    per_sector: int = 2,
) -> list[dict]:
    """Rank scored candidates that would improve diversification.

    `candidates` are scored anchor dicts (ticker, sector, composite_score,
    signal, ...). Only names in UNDER-allocated sectors qualify; within those,
    the biggest sector gap wins first, then the highest model score. Held names
    are excluded and suggestions are spread (≤per_sector per sector)."""
    held_set = {str(t).upper() for t in (held or [])}
    target   = 100.0 / len(CANONICAL_SECTORS)   # even-allocation reference (~9%)
    gap = {s: max(0.0, target - composition.get(s, 0.0)) for s in CANONICAL_SECTORS}

    pool = [
        c for c in candidates
        if str(c.get("ticker", "")).upper() not in held_set
        and c.get("sector") in CANONICAL_SECTORS
        and gap.get(c.get("sector"), 0.0) > 0.0
    ]
    pool.sort(
        key=lambda c: (gap.get(c.get("sector"), 0.0), c.get("composite_score", 0.0)),
        reverse=True,
    )

    out: list[dict] = []
    per: dict[str, int] = {}
    for c in pool:
        sec = c["sector"]
        if per.get(sec, 0) >= per_sector:
            continue
        cur = composition.get(sec, 0.0)
        sig = c.get("signal")
        rationale = f"Adds {sec} exposure (you hold {cur:.0f}%)"
        rationale += f"; the model rates it {sig}." if sig else "."
        out.append({**c, "rationale": rationale, "diversifies": sec})
        per[sec] = per.get(sec, 0) + 1
        if len(out) >= n:
            break
    return out


# ── candidate scoring (the only data-touching helper) ────────────────────────

_CANDIDATE_CACHE: dict = {"key": None, "ts": 0.0, "rows": []}
_CANDIDATE_TTL = 1800.0   # seconds


def score_candidates(model=None) -> list[dict]:
    """Score the cached anchor universe with the trained model, tagged with
    canonical sector. Cached briefly (keyed by model.trained_at) to avoid
    recompute per request. When no fitted model is supplied, trains one from the
    anchor price cache. Returns [] gracefully on any failure."""
    try:
        import pandas as pd
        from .model import PRICE_DIR
        from .feature_engineering import build_features, apply_reference_ranks, safe_ticker_filename
        from .reference_panel import load_reference_panel
        from .scoring import score_universe

        key = getattr(model, "trained_at", None)
        now = time.time()
        if (_CANDIDATE_CACHE["key"] == key
                and now - _CANDIDATE_CACHE["ts"] < _CANDIDATE_TTL
                and _CANDIDATE_CACHE["rows"]):
            return _CANDIDATE_CACHE["rows"]

        raw_data = {}
        for ticker, sector in ANCHOR_SECTORS.items():
            path = PRICE_DIR / f"{safe_ticker_filename(ticker)}.parquet"
            if not path.exists():
                continue
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            if "Close" not in df.columns or len(df) < 60:
                continue
            raw_data[ticker] = {
                "ticker": ticker, "name": ticker, "sector": sector,
                "market": "US", "prices": df, "_raw": {"prices": df}, "info": {},
                "last_price": float(df["Close"].iloc[-1]),
            }

        if len(raw_data) < 10:
            logger.info(f"score_candidates: only {len(raw_data)} cached anchors — skipping")
            return []

        features_df = build_features(raw_data)
        features_df = apply_reference_ranks(features_df, load_reference_panel())

        # No production model handy → train one on the anchor cache so the
        # builder still works (model.fit pulls in the cached anchors itself).
        if model is None or not getattr(model, "is_fitted", False):
            from .model import NUMKTEnsemble
            model = NUMKTEnsemble()
            model.fit(features_df)

        rows = score_universe(model, features_df, raw_data, "quality", "medium")
        # Authoritative sector tag (don't trust whatever build_features carried).
        for r in rows:
            r["sector"] = ANCHOR_SECTORS.get(r.get("ticker"), r.get("sector", "Unknown"))

        _CANDIDATE_CACHE.update({"key": key, "ts": now, "rows": rows})
        return rows
    except Exception as exc:
        logger.warning(f"score_candidates failed (non-fatal): {exc}")
        return []


def build_portfolio(model=None, size: int = 15, max_per_sector: int = 3) -> dict:
    """Construct a diversified portfolio of the model's strongest current picks.

    Scores the whole anchor universe, then greedily takes the highest-conviction
    names subject to a per-sector cap so the result spans multiple sectors. If
    the cap leaves too few names (small/imbalanced universe) it relaxes to fill
    up to `size`. Returns equal-weighted picks + sector composition.
    """
    cands = score_candidates(model)
    if not cands:
        return {"portfolio": [], "composition": {}, "size": 0, "equal_weight_pct": 0}

    ranked = sorted(cands, key=lambda c: c.get("composite_score", 0.0), reverse=True)

    picks, per = [], {}
    for c in ranked:
        sec = c.get("sector", "Unknown")
        if per.get(sec, 0) >= max_per_sector:
            continue
        picks.append(c)
        per[sec] = per.get(sec, 0) + 1
        if len(picks) >= size:
            break
    if len(picks) < size:   # relax the cap to reach the target count
        have = {c.get("ticker") for c in picks}
        for c in ranked:
            if c.get("ticker") in have:
                continue
            picks.append(c)
            if len(picks) >= size:
                break

    w = round(100.0 / len(picks), 1)
    comp: dict[str, float] = {}
    out: list[dict] = []
    for c in picks:
        sec = c.get("sector", "Unknown")
        comp[sec] = round(comp.get(sec, 0.0) + 100.0 / len(picks), 1)
        sig = c.get("signal")
        out.append({
            **c,
            "weight": w,
            "rationale": (f"{sig} · {sec} — among the model's strongest current picks"
                          if sig else f"{sec} — model pick"),
        })
    return {"portfolio": out, "composition": comp, "size": len(out), "equal_weight_pct": w}
