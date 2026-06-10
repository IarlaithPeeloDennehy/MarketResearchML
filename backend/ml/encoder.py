"""
encoder.py
──────────
Self-supervised temporal price encoder (TS2Vec-style contrastive).

This module is the *production-side* half of the offline pre-training feature:
it defines the encoder architecture, loads a frozen ``encoder.pt`` checkpoint,
and turns a single ticker's trailing price window into a fixed-length embedding.

The heavy lifting (training the encoder self-supervised on a large unlabeled
price corpus) happens OFFLINE on a GPU box in ``scripts/pretrain_encoder.py``.
Production never trains the encoder — it only runs ``embed()`` on CPU.

Design constraints (see plan):
  * torch is an OPTIONAL dependency. If it is not installed, this module still
    imports cleanly; ``load_encoder`` returns None and the whole app falls back
    to the price-ranks-only behaviour it had before pre-training existed.
  * The encoder is FROZEN at inference (``eval()``, no dropout, ``no_grad``),
    so a given (window) → (embedding) mapping is fully deterministic.
  * Embeddings are point-in-time: the caller passes only the price history up
    to the window start, so non-causal convolutions inside the encoder cannot
    leak future information — the future simply is not in the input tensor.

Checkpoint format (saved by the offline trainer, ``torch.save``):
    {
      "format_version": 1,
      "config":   EncoderConfig.__dict__,        # architecture
      "state_dict": <model weights>,
      "meta": {
        "version":             "<short content/version hash>",
        "pretrain_cutoff_date": "YYYY-MM-DD",     # no training data after this
        "channels":            ["logret", "vol"], # channel spec, for sanity
        "norm":                "per_window",       # normalization scheme
        "trained_at":          "<iso8601>",
      },
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional torch import ──────────────────────────────────────────────────
# The app must import and run even when torch is absent (e.g. local dev, or a
# deployment with USE_ENCODER off). All torch-dependent code is guarded.
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the no-torch fallback path
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


# ── Architecture defaults ───────────────────────────────────────────────────
DEFAULT_WINDOW    = 252      # trading days fed to the encoder
DEFAULT_EMBED_DIM = 32       # kept small on purpose — RF head trains on few rows
DEFAULT_CHANNELS  = ("logret", "vol")   # input channels built from a price series


@dataclass
class EncoderConfig:
    """Architecture-only config. Persisted inside the checkpoint."""
    n_channels: int = len(DEFAULT_CHANNELS)
    window:     int = DEFAULT_WINDOW
    hidden_dim: int = 64
    depth:      int = 6        # dilated conv blocks; receptive field grows 2**i
    kernel_size: int = 3
    embed_dim:  int = DEFAULT_EMBED_DIM


# ── Model definition (only when torch is available) ─────────────────────────
if _TORCH_AVAILABLE:
    import torch.nn as nn
    import torch.nn.functional as F

    class _SamePadConv(nn.Module):
        """1-D dilated conv with 'same' length output."""

        def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
            super().__init__()
            receptive = (kernel - 1) * dilation + 1
            padding = receptive // 2
            self.conv = nn.Conv1d(in_ch, out_ch, kernel,
                                  padding=padding, dilation=dilation)
            # even receptive field over-pads by one on the right; trim it.
            self._remove = 1 if receptive % 2 == 0 else 0

        def forward(self, x):
            out = self.conv(x)
            if self._remove:
                out = out[:, :, : -self._remove]
            return out

    class _ConvBlock(nn.Module):
        """Residual dilated conv block (TS2Vec-style)."""

        def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
            super().__init__()
            self.conv1 = _SamePadConv(in_ch, out_ch, kernel, dilation)
            self.conv2 = _SamePadConv(out_ch, out_ch, kernel, dilation)
            self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

        def forward(self, x):
            residual = x if self.proj is None else self.proj(x)
            x = F.gelu(self.conv1(x))
            x = self.conv2(x)
            return x + residual

    class _DilatedConvEncoder(nn.Module):
        def __init__(self, in_ch: int, channels: list[int], kernel: int):
            super().__init__()
            blocks = []
            prev = in_ch
            for i, ch in enumerate(channels):
                blocks.append(_ConvBlock(prev, ch, kernel, dilation=2 ** i))
                prev = ch
            self.net = nn.Sequential(*blocks)

        def forward(self, x):
            return self.net(x)

    class TS2VecEncoder(nn.Module):
        """Dilated-TCN encoder producing per-timestep representations.

        ``forward`` returns per-timestep reprs (used by the contrastive loss in
        the offline trainer). ``embed_window`` max-pools over time to produce one
        fixed-length embedding per window (used at inference).
        """

        def __init__(self, cfg: EncoderConfig):
            super().__init__()
            self.cfg = cfg
            self.input_fc = nn.Linear(cfg.n_channels, cfg.hidden_dim)
            channels = [cfg.hidden_dim] * cfg.depth + [cfg.embed_dim]
            self.feature_extractor = _DilatedConvEncoder(
                cfg.hidden_dim, channels, cfg.kernel_size
            )
            self.repr_dropout = nn.Dropout(p=0.1)  # disabled in eval()

        def forward(self, x):           # x: (B, T, C)
            x = self.input_fc(x)        # (B, T, hidden)
            x = x.transpose(1, 2)       # (B, hidden, T)
            x = self.feature_extractor(x)   # (B, embed_dim, T)
            x = self.repr_dropout(x)
            return x.transpose(1, 2)    # (B, T, embed_dim)

        def embed_window(self, x):      # (B, T, C) -> (B, embed_dim)
            reprs = self.forward(x)
            return reprs.max(dim=1).values


# ── Loaded-encoder wrapper (returned to the app) ─────────────────────────────
class LoadedEncoder:
    """Frozen, eval-mode encoder ready for CPU inference.

    Holds the torch model plus the metadata the rest of the app needs to stay
    consistent (embedding dim, version hash, channel spec). All embedding math
    lives here so callers never touch torch directly.
    """

    def __init__(self, model, cfg: EncoderConfig, meta: dict):
        self._model = model
        self.cfg = cfg
        self.meta = meta or {}
        self.embed_dim: int = cfg.embed_dim
        self.window: int = cfg.window
        self.version: str = str(self.meta.get("version", "unknown"))
        # The channel spec is the contract between this encoder and channels.py.
        # It comes from the checkpoint so a model and its inputs can never drift.
        self.channels = list(self.meta.get("channels", DEFAULT_CHANNELS))

    def embed_channels(self, raw_mat: np.ndarray) -> np.ndarray:
        """Embed a raw (T, C) channel matrix built by ``channels.build_window_channels``.

        Standardizes per-window (shared with the trainer via channels.standardize_window),
        keeps/pads the most recent ``window`` steps, then runs the frozen model.
        Returns a (embed_dim,) vector. Empty/short input → neutral zeros.
        """
        from . import channels  # local import avoids any import-time coupling

        raw_mat = np.asarray(raw_mat, dtype=np.float32)
        if raw_mat.size == 0 or raw_mat.shape[0] == 0:
            return np.zeros(self.embed_dim, dtype=np.float32)
        mat = channels.standardize_window(raw_mat)
        w = self.window
        if mat.shape[0] >= w:
            mat = mat[-w:]
        else:
            pad = np.zeros((w - mat.shape[0], mat.shape[1]), dtype=np.float32)
            mat = np.vstack([pad, mat])
        with torch.no_grad():
            x = torch.from_numpy(mat).unsqueeze(0)   # (1, T, C)
            emb = self._model.embed_window(x).squeeze(0).cpu().numpy()
        return emb.astype(np.float32)


# ── Public loader ───────────────────────────────────────────────────────────
def load_encoder(path: str | Path) -> Optional[LoadedEncoder]:
    """Load a frozen encoder checkpoint. Returns None on any failure.

    Failure modes (all non-fatal — caller falls back to price-ranks-only):
      * torch not installed
      * file missing / unreadable
      * malformed checkpoint
    """
    if not _TORCH_AVAILABLE:
        logger.info("Encoder requested but torch is not installed — skipping.")
        return None

    path = Path(path)
    if not path.exists():
        logger.info(f"Encoder checkpoint not found at {path} — skipping.")
        return None

    try:
        ckpt = torch.load(path, map_location="cpu")
        cfg_dict = ckpt.get("config", {})
        cfg = EncoderConfig(**{k: v for k, v in cfg_dict.items()
                               if k in EncoderConfig.__dataclass_fields__})
        model = TS2VecEncoder(cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        loaded = LoadedEncoder(model, cfg, ckpt.get("meta", {}))
        logger.info(
            f"Loaded encoder v{loaded.version} "
            f"(embed_dim={loaded.embed_dim}, window={loaded.window}, "
            f"channels={loaded.channels}, cutoff={loaded.meta.get('pretrain_cutoff_date')})"
        )
        return loaded
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(f"Failed to load encoder from {path}: {exc}", exc_info=True)
        return None


def save_checkpoint(model, cfg: EncoderConfig, meta: dict, path: str | Path) -> None:
    """Helper used by the offline trainer to persist a checkpoint."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is required to save an encoder checkpoint")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "config": asdict(cfg),
            "state_dict": model.state_dict(),
            "meta": meta,
        },
        path,
    )
    logger.info(f"Saved encoder checkpoint to {path}")
