"""
psi.py — pure statistics for drift monitoring. No model/DB/app dependencies, so
it's trivially unit-testable.

Population Stability Index (PSI) measures how much a live distribution has
shifted from a training baseline:

    PSI = Σ (actual_pct - expected_pct) * ln(actual_pct / expected_pct)

over a fixed set of bins (frozen from the training baseline). Convention:
    PSI < 0.10            → stable
    0.10 ≤ PSI < 0.25     → warning
    PSI ≥ 0.25            → critical
"""
from __future__ import annotations

import numpy as np

from . import config

_EPS = 1e-6
_QUANTILES = [5, 25, 50, 75, 95]


def _clean(values) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def distribution_stats(values) -> dict:
    """mean/std/missing_rate + p5..p95. Robust to all-NaN / empty input."""
    a = np.asarray(values, dtype=np.float64).ravel()
    n = a.size
    finite = a[np.isfinite(a)]
    missing_rate = float(1.0 - finite.size / n) if n else 1.0
    if finite.size == 0:
        out = {"mean": 0.0, "std": 0.0, "missing_rate": missing_rate, "n": int(n)}
        out.update({f"p{q}": 0.0 for q in _QUANTILES})
        return out
    out = {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "missing_rate": missing_rate,
        "n": int(n),
    }
    qs = np.percentile(finite, _QUANTILES)
    out.update({f"p{q}": float(v) for q, v in zip(_QUANTILES, qs)})
    return out


def histogram_edges(values, n_bins: int | None = None) -> list[float]:
    """Quantile-based bin edges from the BASELINE values (frozen for reuse).
    Falls back to a single degenerate bin for constant/empty input."""
    n_bins = n_bins or config.PSI_BINS
    finite = _clean(values)
    if finite.size < 2:
        return []
    qs = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(finite, qs))
    if edges.size < 2:
        return []
    # widen the outer edges so future extreme values still land in a bin
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges.tolist()


def proportions(values, edges: list[float]) -> list[float]:
    """Fraction of `values` falling in each bin defined by `edges`."""
    finite = _clean(values)
    if not edges or len(edges) < 2 or finite.size == 0:
        return []
    counts, _ = np.histogram(finite, bins=np.asarray(edges, dtype=float))
    total = counts.sum()
    if total == 0:
        return [0.0] * len(counts)
    return (counts / total).tolist()


def psi_value(expected_props: list[float], actual_props: list[float]) -> float:
    """PSI between two proportion vectors over identical bins."""
    e = np.asarray(expected_props, dtype=np.float64)
    a = np.asarray(actual_props, dtype=np.float64)
    if e.size == 0 or e.size != a.size:
        return 0.0
    e = np.clip(e, _EPS, None)
    a = np.clip(a, _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value: float) -> str:
    if value >= config.PSI_CRITICAL:
        return "critical"
    if value >= config.PSI_WARNING:
        return "warning"
    return "stable"


def histogram(values, n_bins: int = 20) -> dict:
    """Plain histogram for dashboard rendering (edges + counts)."""
    finite = _clean(values)
    if finite.size == 0:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(finite, bins=n_bins)
    return {"edges": edges.tolist(), "counts": counts.tolist()}
