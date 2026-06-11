#!/usr/bin/env python
"""
pretrain_encoder.py — OFFLINE self-supervised pre-training (GPU box).

Scenario-1 training half. Reads the corpus cache produced by
build_pretrain_corpus.py (prices/ + info/ + series/), builds multi-channel
windows through the SHARED ml.channels.build_window_channels (identical to
production inference — this is what guarantees train/inference parity), trains a
TS2Vec-style contrastive encoder, and writes a frozen encoder.pt whose meta
records the exact channel spec.

Channels are driven by --channels (default = base + Group A). The same spec is
stored in the checkpoint so production rebuilds the same inputs.

Example:
    python scripts/pretrain_encoder.py \
        --corpus-dir data/corpus_cache \
        --out artifacts/encoder.pt \
        --cutoff-date 2020-01-01 --embed-dim 32 --epochs 40 --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ml.encoder import EncoderConfig, TS2VecEncoder, save_checkpoint  # noqa: E402
from ml import channels as ch  # noqa: E402
from ml.feature_engineering import safe_ticker_filename  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pretrain")


# ── corpus loading (mirrors production channel inputs) ──────────────────────
def _load_close_vol(price_dir: Path, ticker: str):
    p = price_dir / f"{safe_ticker_filename(ticker)}.parquet"
    if not p.exists():
        return None, None
    try:
        df = pd.read_parquet(p)
        close = df["Close"].sort_index() if "Close" in df.columns else None
        vol = df["Volume"].sort_index() if "Volume" in df.columns else None
        return close, vol
    except Exception:
        return None, None


def _load_sector(info_dir: Path, ticker: str):
    p = info_dir / f"{safe_ticker_filename(ticker)}.json"
    if not p.exists():
        return None
    try:
        return (json.load(open(p)) or {}).get("sector")
    except Exception:
        return None


def _load_series(series_dir: Path, ticker: str):
    p = series_dir / f"{safe_ticker_filename(ticker)}.json"
    if not p.exists():
        return None, None
    try:
        d = json.load(open(p)) or {}
        return d.get("earnings"), d.get("analyst")
    except Exception:
        return None, None


def build_windows(corpus_dir: Path, spec: list[str], window: int, stride: int,
                  cutoff_date: str | None, max_tickers: int | None,
                  max_windows: int | None) -> np.ndarray:
    price_dir, info_dir, series_dir = corpus_dir / "prices", corpus_dir / "info", corpus_dir / "series"
    cutoff = pd.Timestamp(cutoff_date) if cutoff_date else None

    # reference ETF Close series (market + sectors), truncated to cutoff
    refs: dict[str, pd.Series | None] = {}
    for etf in ch.REFERENCE_TICKERS:
        c, _ = _load_close_vol(price_dir, etf)
        if c is not None and cutoff is not None:
            c = c[c.index <= cutoff]
        refs[etf] = c

    ref_stems = {safe_ticker_filename(e) for e in ch.REFERENCE_TICKERS}
    files = sorted(p for p in price_dir.glob("*.parquet") if p.stem not in ref_stems)
    if max_tickers:
        files = files[:max_tickers]
    if not files:
        raise SystemExit(f"No ticker parquets in {price_dir}")

    out: list[np.ndarray] = []
    n_tickers = 0
    for fp in files:
        ticker = fp.stem
        close, vol = _load_close_vol(price_dir, ticker)
        if close is None:
            continue
        if cutoff is not None:
            close = close[close.index <= cutoff]
            if vol is not None:
                vol = vol[vol.index <= cutoff]
        if len(close) < window + 2:
            continue

        sector_etf = ch.sector_to_etf(_load_sector(info_dir, ticker))
        market_close = refs.get(ch.MARKET_ETF)
        sector_close = refs.get(sector_etf)
        earnings, analyst = _load_series(series_dir, ticker)

        n_tickers += 1
        for end in range(window, len(close), stride):
            mat = ch.build_window_channels(
                close, spec, end_idx=end, window=window, volume=vol,
                market_close=market_close, sector_close=sector_close,
                earnings_hist=earnings, analyst_hist=analyst,
            )
            if mat.shape[0] != window:
                continue
            out.append(ch.standardize_window(mat))
            if max_windows and len(out) >= max_windows:
                logger.info(f"Reached max_windows={max_windows}")
                arr = np.stack(out).astype(np.float32)
                logger.info(f"Built {arr.shape[0]} windows from {n_tickers} tickers, shape={arr.shape}")
                return arr
    if not out:
        raise SystemExit("No windows built — check window length vs history / cutoff.")
    arr = np.stack(out).astype(np.float32)
    logger.info(f"Built {arr.shape[0]} windows from {n_tickers} tickers, shape={arr.shape}")
    return arr


# ── augmentations ───────────────────────────────────────────────────────────
def augment(x: torch.Tensor, mask_p=0.15, jitter=0.01, scale=0.05) -> torch.Tensor:
    out = x + torch.randn_like(x) * jitter
    s = 1.0 + torch.randn(x.size(0), 1, x.size(2), device=x.device) * scale
    out = out * s
    if mask_p > 0:
        m = (torch.rand(x.size(0), x.size(1), 1, device=x.device) > mask_p).float()
        out = out * m
    return out


# ── TS2Vec hierarchical contrastive loss ────────────────────────────────────
def instance_contrastive_loss(z1, z2):
    B = z1.size(0)
    if B == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=0).transpose(0, 1)
    sim = torch.matmul(z, z.transpose(1, 2))
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    i = torch.arange(B, device=z1.device)
    return (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2


def temporal_contrastive_loss(z1, z2):
    T = z1.size(1)
    if T == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=1)
    sim = torch.matmul(z, z.transpose(1, 2))
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    t = torch.arange(T, device=z1.device)
    return (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2


def hierarchical_contrastive_loss(z1, z2, alpha=0.5):
    loss = z1.new_tensor(0.0)
    d = 0
    while z1.size(1) > 1:
        loss = loss + alpha * instance_contrastive_loss(z1, z2)
        loss = loss + (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        loss = loss + alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / max(d, 1)


# ── training ────────────────────────────────────────────────────────────────
def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        logger.warning("CUDA requested but unavailable — using CPU.")

    spec = list(args.channels)
    cfg = EncoderConfig(
        n_channels=len(spec), window=args.window, hidden_dim=args.hidden_dim,
        depth=args.depth, kernel_size=args.kernel_size, embed_dim=args.embed_dim,
    )

    windows = build_windows(Path(args.corpus_dir), spec, args.window, args.stride,
                            args.cutoff_date, args.max_tickers, args.max_windows)
    data = torch.from_numpy(windows)
    n = data.size(0)

    model = TS2VecEncoder(cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    logger.info(f"Training: {n} windows, {len(spec)} channels {spec}, {args.epochs} epochs, "
                f"batch {args.batch_size}, device={device}, embed_dim={cfg.embed_dim}")
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        total, steps = 0.0, 0
        for s in range(0, n, args.batch_size):
            x = data[perm[s: s + args.batch_size]].to(device)
            if x.size(0) < 2:
                continue
            v1, v2 = model(augment(x)), model(augment(x))
            loss = hierarchical_contrastive_loss(v1, v2)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()); steps += 1
        logger.info(f"epoch {epoch + 1}/{args.epochs}  loss={total / max(steps, 1):.4f}")

    model.eval()
    version = hashlib.sha1(f"{cfg}|{spec}|{args.cutoff_date}|{time.time()}".encode()).hexdigest()[:12]
    meta = {
        "version": version,
        "pretrain_cutoff_date": args.cutoff_date,
        "channels": spec,                  # the contract production must rebuild
        "norm": "per_window",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_windows": int(n),
        "objective": "ts2vec_hierarchical_contrastive",
    }
    save_checkpoint(model, cfg, meta, args.out)
    logger.info(f"DONE — encoder v{version} ({len(spec)} channels) → {args.out}. "
                f"Upload to object storage and set ENCODER_URL on Render.")


def parse_args():
    p = argparse.ArgumentParser(description="Offline TS2Vec multi-channel price-encoder pre-training")
    p.add_argument("--corpus-dir", required=True, help="corpus CACHE_DIR (prices/ info/ series/)")
    p.add_argument("--out", default="artifacts/encoder.pt")
    p.add_argument("--cutoff-date", default=None,
                   help="YYYY-MM-DD; drop bars after this to keep downstream holdout honest")
    p.add_argument("--channels", nargs="+", default=ch.DEFAULT_SPEC,
                   help=f"channel spec (default Group A). All: {ch.ALL_CHANNELS}")
    p.add_argument("--window", type=int, default=252)
    p.add_argument("--stride", type=int, default=21)
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-tickers", type=int, default=None)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
