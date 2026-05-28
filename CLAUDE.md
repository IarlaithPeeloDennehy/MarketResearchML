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
| `backend/ml/model.py` | 863 | NUMKTEnsemble, training pipeline, price-only feature generation |
| `backend/ml/backtest.py` | 574 | Walk-forward backtest, real-return training loop |
| `backend/ml/data_fetcher.py` | 601 | Finnhub (US) + yfinance (UK/IE) fetch, parquet cache |
| `backend/auth/router.py` | 389 | Signup, login, logout, me, change-password, delete-account |
| `backend/auth/models.py` | ~104 | SQLModel tables: users, sessions, preferences, analyses, activity |
| `index.html` | 386 | Entire frontend — markup, JS, and inline styles in one file |

Reference by `file:function` not by file alone for large files.
e.g. `model.py:_labels_for_horizon`, `backtest.py:run_backtest`, `main.py:startup`

## Architecture
- US tickers → Finnhub REST API (`FINNHUB_API_KEY` env var); UK/IE (.L, .IR) → yfinance
- ML training is **price-only** (no fundamentals in historical windows — look-ahead bias fix)
- `PRICE_FEATURE_COLS` = 7 momentum/vol/RSI features; `FEATURE_COLS` = 21 (includes fundamentals for current snapshot only)
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
- Do not skip the `_existing_tables()` guard in new Alembic migrations
- Do not commit `.env` files or hardcode API keys
- Do not use `git --no-verify` or `--no-gpg-sign`
