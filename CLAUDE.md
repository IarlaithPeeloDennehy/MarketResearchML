# NUMKT — Claude Working Instructions

## Project
ML-powered stock analysis SPA. FastAPI backend + vanilla JS frontend served as static HTML. PostgreSQL via SQLModel/Alembic. Deployed on Render.

## Environment
- **OS:** Windows 11 — PowerShell syntax only. No bash heredocs, no `&&` chaining.
- **Python runtime:** 3.11.9 (see `backend/runtime.txt`)
- **Local Python:** mambaforge 3.10 — run tests locally with this, but code targets 3.11
- **Shell for git:** use `git commit -m @'...'@` heredoc syntax in PowerShell

## Key Files
| File | Lines | What it does |
|------|-------|-------------|
| `backend/main.py` | 776 | FastAPI app, all routes, startup, caches, rate limiting |
| `backend/ml/model.py` | ~900 | NUMKTEnsemble (RF+GB head), training pipeline, price-only + embedding feature generation |
| `backend/ml/backtest.py` | ~590 | Walk-forward backtest, real-return training loop |
| `backend/ml/data_fetcher.py` | 601 | Finnhub (US) + yfinance (UK/IE) fetch, parquet cache |
| `backend/ml/encoder.py` | ~230 | TS2Vec-style price encoder (torch, optional), `load_encoder`/`embed_channels`. Production-side of offline pre-training |
| `backend/ml/channels.py` | ~300 | **Single source of truth** for encoder input channels. `build_window_channels` used by BOTH offline trainer and production inference (parity-critical) |
| `backend/ml/embedding_features.py` | ~280 | Bridge: loads channel inputs by ticker → `channels.build_window_channels` → `emb_*` columns; cross-sectional standardize; in-mem cache; no-op when disabled |
| `backend/ml/series_cache.py` | ~60 | Persists Finnhub earnings/analyst **history** to `$CACHE_DIR/series/{ticker}.json` (Group B channels) |
| `scripts/pretrain_encoder.py` | ~280 | OFFLINE GPU trainer (TS2Vec contrastive); builds multi-channel windows via `channels.py` → `encoder.pt`. Not imported by the app |
| `scripts/build_pretrain_corpus.py` | ~110 | OFFLINE corpus builder; drives `data_fetcher` (Finnhub US only) into a corpus `CACHE_DIR` |
| `backend/auth/router.py` | 389 | Signup, login, logout, me, change-password, delete-account |
| `backend/auth/models.py` | ~104 | SQLModel tables: users, sessions, preferences, analyses, activity |
| `index.html` | 386 | Entire frontend — markup, JS, and inline styles in one file |

Reference by `file:function` not by file alone for large files.
e.g. `model.py:_labels_for_horizon`, `backtest.py:run_backtest`, `main.py:startup`

## Architecture
- US tickers → Finnhub REST API (`FINNHUB_API_KEY` env var); UK/IE (.L, .IR) → yfinance
- ML training is **price-only** (no fundamentals in historical windows — look-ahead bias fix)
- `PRICE_FEATURE_COLS` = 7 momentum/vol/RSI features; `FEATURE_COLS` = 21 (includes fundamentals for current snapshot only)
- **Pre-training = the offline self-supervised price encoder** (`scripts/pretrain_encoder.py`), trained on a GPU box. "Head training" (`_startup_train_head`, `run_backtest`) is the supervised RF+GB step — do NOT call that "pre-training".
- When an `encoder.pt` is loaded, point-in-time embeddings (`emb_0..emb_{D-1}`) **augment** the price-rank features; the RF+GB head consumes `[price ranks + embeddings]`. Embeddings are generated causally (only prices ≤ window end) and cross-sectionally standardized per snapshot.
- **Encoder channels** are defined ONCE in `channels.py` (`build_window_channels`) and used by both the offline trainer and production — they MUST stay identical or the frozen encoder gets garbage. The active spec is stored in the checkpoint's `meta["channels"]`. Channel set: base `logret`/`vol`; Group A `rel_volume`/`log_dollar_vol`/`excess_ret_vs_market`(SPY)/`rel_strength_vs_sector`(sector SPDR); Group B `earn_surprise_pulse`/`analyst_net_buy_ts` (from the `series/` cache, point-in-time gated). Missing source ⇒ that channel is neutral (zeros).
- Market/sector channels use **reference ETFs** (SPY + sector SPDRs, `channels.REFERENCE_TICKERS`); `main.startup` warms them in the background when the encoder is enabled.
- **Graceful fallback:** no torch / `USE_ENCODER=0` / no checkpoint ⇒ `emb_*` columns absent ⇒ exactly the pre-encoder price-ranks-only behaviour.
- Sessions: HttpOnly cookies, SHA-256 token hashes stored (never raw tokens)
- Alembic migrations run automatically on Render startup via Procfile (`alembic upgrade head`)
- Background tasks use `_fire_task()` pattern with strong refs in `_background_tasks` set

## Active Migrations
| Revision | Description |
|----------|-------------|
| 0001 | Initial schema |
| 0002 | sessions.token_hash UNIQUE constraint |

Next revision must be **0003**.

## Standing Rules

### Before any implementation
For tasks touching >3 files or any migration/deployment: **plan first, no code**.
List files affected, changes, and risks. Wait for approval before executing.

### Before declaring any feature complete
Run the full test suite and report results:
```
cd backend && python -m pytest tests/ -v --tb=short
```
Fix all failures before saying the work is done. Never declare done with failing tests.

### When swapping any integration or provider
Before writing code, grep the entire repo for every variant of the old name
(provider name, UI strings, log messages, env var names, comments).
Report the full list of references before making any changes.

### Output style
- No trailing summaries after completing tasks
- No explanations of what existing code does unless asked
- Diffs/edits preferred over full file rewrites
- One-line-per-finding for audits and reviews

## Testing
- Test files: `backend/tests/test_ml_integrity.py`, `test_cache_isolation.py`, `test_backtest_metrics.py`
- Run from `backend/` directory
- 21 tests currently — all should pass before any commit
- New features require new tests

## Deployment (Render)
- Procfile: `alembic upgrade head && uvicorn main:app ...`
- Required env vars: `DATABASE_URL`, `FINNHUB_API_KEY`, `SECRET_KEY`
- Email (when implemented): `RESEND_API_KEY`, `FROM_EMAIL`, `APP_URL`
- Encoder (optional): `USE_ENCODER` (default on), `ENCODER_URL` (object-storage URL of `encoder.pt`, downloaded to `$CACHE_DIR/models/` on startup), `ENCODER_VERSION` (optional expected version hash). All non-fatal — app runs without them.
- `torch` is **optional** and intentionally NOT in `backend/requirements.txt` (the app degrades to price-ranks-only without it). To enable the encoder, also install `backend/requirements-encoder.txt` (`torch==2.5.1+cpu`) and set `USE_ENCODER=1` + `ENCODER_URL`. **Memory risk:** torch + uvicorn + sklearn may exceed Render's 512MB free tier — size the instance up. `CACHE_DIR` must be a persistent disk to retain the encoder/model across deploys.
- Migrations run automatically — never manually run `alembic upgrade` against production
- Before any deploy-touching task: verify all required env vars are documented

## Auth Patterns
- Passwords: bcrypt with SHA-256 pre-hash (handles >72 byte truncation)
- Tokens: `secrets.token_hex(32)` → stored as `hashlib.sha256(token).hexdigest()`
- `is_active` = admin disable flag; `email_verified` = verification gate (pending implementation)
- Constant-time rejection on login (dummy hash on miss) — preserve this

## What NOT to do
- Do not use `asyncio.get_event_loop()` — use `get_running_loop()` inside async functions
- Do not use bare `asyncio.create_task()` — use `_fire_task()` from main.py
- Do not add `FEATURE_COLS` (fundamental features) to historical training windows
- Do not add `emb_*` columns from data after a window's end — embeddings must stay point-in-time (`point_in_time_embedding(..., end_idx=...)`)
- Do not train the encoder inside the app or on Render — pre-training is offline only (`scripts/pretrain_encoder.py`); production loads a frozen checkpoint
- Do not build encoder channels anywhere except `channels.build_window_channels` — duplicating channel logic breaks train/inference parity (the parity is covered by `tests/test_channels.py`)
- Do not route Finnhub series history through `_save_info` (it strips non-scalars) — use `series_cache.save`
- Adding/removing an encoder channel changes `n_channels` ⇒ the encoder MUST be retrained and the corpus rebuilt
- Do not skip the `_existing_tables()` guard in new Alembic migrations
- Do not commit `.env` files or hardcode API keys
- Do not use `git --no-verify` or `--no-gpg-sign`
