# NUMKT — Multi-Market ML Stock Analyser

> A self-improving ensemble machine learning stock analysis tool covering US, UK (LSE) and Irish (Euronext Dublin) markets. Combines real-time fundamental data, Fama-French 5-factor quant analysis, and a walk-forward trained Random Forest / Gradient Boosting ensemble to produce ranked BUY, HOLD and SELL signals with plain-English rationale.

---

## Table of Contents

- [What It Does](#what-it-does)
- [Project Structure](#project-structure)
- [Features](#features)
- [Getting Started](#getting-started)
- [Authentication](#authentication)
- [Backend API Reference](#backend-api-reference)
- [How the ML Model Works](#how-the-ml-model-works)
- [Walk-Forward Backtest](#walk-forward-backtest)
- [Self-Improvement Loop](#self-improvement-loop)
- [Caching](#caching)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Security](#security)
- [Disclaimer](#disclaimer)

---

## What It Does

NUMKT fetches real-time price and fundamental data from Yahoo Finance for any set of stocks you choose, builds a 16-feature factor matrix, and trains an ensemble ML model to rank those stocks by their probability of outperforming the universe median over the next 12 months.

The result is a ranked signal table — STRONG BUY, BUY, HOLD or SELL — for each stock, together with a plain-English explanation of what's driving the call, a Fama-French 5-factor decomposition, and key fundamental metrics. A walk-forward backtest measures how well the model's factor rankings have predicted actual returns in your universe historically, and — critically — uses those real outcomes to retrain the model, so it improves with use.

Everything runs in a single web page served by the FastAPI backend. User accounts let you save analyses, set default preferences, and review your activity log.

---

## Project Structure

```
/
├── index.html                  # Single-page frontend — no build step
│
└── backend/
    ├── main.py                 # FastAPI app — all HTTP endpoints
    ├── requirements.txt
    ├── Procfile                # Deployment process file
    ├── runtime.txt             # Python version pin
    │
    ├── ml/
    │   ├── data_fetcher.py     # Yahoo Finance fetch + 7-day disk cache
    │   ├── feature_engineering.py  # 16-feature factor matrix + rank normalisation
    │   ├── model.py            # RF + GB ensemble; fit, save, load
    │   ├── backtest.py         # Walk-forward backtest + model training on real returns
    │   └── scoring.py          # Model probability → signal + FF5 + buy reasons
    │
    ├── auth/
    │   ├── models.py           # SQLModel table definitions (users, sessions, etc.)
    │   ├── router.py           # Register, login, logout, change-password, delete-account
    │   ├── dependencies.py     # get_current_user FastAPI dependency
    │   └── utils.py            # bcrypt hashing, token generation, expiry
    │
    ├── user/
    │   └── router.py           # Preferences, saved analyses, history, session list
    │
    ├── db/
    │   ├── base.py             # SQLModel metadata + table creation helper
    │   └── session.py          # get_db dependency
    │
    └── alembic/
        └── versions/
            └── 0001_initial_schema.py  # Initial migration
```

---

## Features

### Analysis

- Build a custom universe of 3–40 stocks from any mix of US, UK and Irish markets
- 80+ pre-loaded stocks to choose from, with live search
- Seven investor profiles, each weighting factors differently:
  - **Quality (QARP)** — ROE, margins, valuation discipline
  - **Value** — P/E, P/B, earnings yield
  - **Growth** — revenue growth, earnings acceleration
  - **Dividend** — yield, payout sustainability, balance sheet quality
  - **Momentum** — 3M and 12M price trend persistence
  - **Global Macro** — sector positioning relative to rates, FX and growth regime
  - **Activist / Deep Value** — depressed multiples, low P/B, re-rating potential
- Three risk levels (low / medium / high) that apply a beta-adjusted penalty to scores
- Adjustable ML parameters from the sidebar: lookback period, L2 regularisation (λ), tree depth, CV folds

### Signals and Output

- Ranked BUY / HOLD / SELL table with colour-coded confidence bars and institutional ownership
- Per-stock detail panel:
  - Composite score (ML probability minus risk penalty)
  - Fama-French 5-factor decomposition: HML, RMW, CMA, SMB, Momentum
  - Up to four plain-English buy/sell reasons derived from the stock's factor profile
  - Full fundamentals table: P/E, P/B, ROE, net margin, revenue growth, debt/equity, dividend yield, beta, 12M/3M momentum, RSI, institutional and insider ownership
- Summary grid: top pick, buy count, average confidence, CV accuracy
- Feature importance chart (real model weights after backtest, or profile weights before)
- Top-5 momentum simulation chart
- Sector exposure doughnut

### Backtest

- Walk-forward simulation across the full price history of your universe
- Configurable forward window (default 12M) and rebalance frequency (default 3M)
- Train / holdout split — the final ~20% of periods are never seen during training, giving an honest out-of-sample IC
- Metrics: annualised portfolio return, alpha vs equal-weight benchmark, Sharpe ratio, Calmar ratio, max drawdown, hit rate, IC (mean and std), ICIR, beta, average monthly turnover
- Per-period prediction examples showing model scores vs actual outcomes for the holdout set

### User Accounts

- Register with email and password; bcrypt-hashed at rest
- Server-side sessions via HttpOnly cookies — no tokens stored in the browser
- Save any analysis by name — retrieve the full results later
- User preferences: default profile, risk level, lookback period, default ticker list
- Active session list with IP address and user-agent — revoke any session individually
- Activity log: every login, analysis, backtest, password change and account deletion is recorded with timestamp and IP

---

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL (local or hosted)
- A modern browser

### Installation

```bash
git clone https://github.com/iarlaithpeelodennehy/marketresearchml.git
cd marketresearchml

cd backend
pip install -r requirements.txt
```

### Environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string, e.g. `postgresql://user:pass@localhost:5432/numkt` |
| `SECRET_KEY` | Yes | Random hex string used for token signing. Generate with `python -c "import secrets; print(secrets.token_hex(32))"` |
| `ALLOWED_ORIGINS` | No | Comma-separated list of allowed CORS origins. Leave empty when the frontend is served from this same app (the default). |
| `CACHE_DIR` | No | Path to the price/info cache directory. Defaults to `backend/cache/`. Set this to a persistent volume on cloud hosts. |

### Database setup

Run the Alembic migration to create all tables:

```bash
cd backend
alembic upgrade head
```

### Running the app

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in your browser. The frontend is served directly from the FastAPI app — no separate frontend server is needed.

---

## Authentication

All analysis and cache/model endpoints require a logged-in session. The auth flow:

1. Register at `/auth/register` with email + password (minimum 8 characters)
2. Log in at `/auth/login` — the server sets an `HttpOnly; SameSite=Lax` session cookie
3. All subsequent requests carry the cookie automatically — no manual token handling
4. Log out at `/auth/logout` to revoke the session server-side
5. Change password at `/auth/change-password` (requires current password)
6. Delete account at `/auth/delete-account` (requires password confirmation — permanently removes all data)

Sessions expire after 30 days of inactivity. You can view and revoke individual active sessions from the account panel.

---

## Backend API Reference

All endpoints except `/`, `/health`, `/auth/register` and `/auth/login` require authentication via session cookie.

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/` | No | Serves the frontend HTML |
| `GET` | `/health` | No | Health check — `{"status": "ok"}` |
| `GET` | `/api/status` | No | Service info and saved model summary |
| `POST` | `/auth/register` | No | Create account |
| `POST` | `/auth/login` | No | Log in, sets session cookie |
| `POST` | `/auth/logout` | Yes | Revoke current session |
| `POST` | `/auth/change-password` | Yes | Change password |
| `DELETE` | `/auth/delete-account` | Yes | Permanently delete account |
| `GET` | `/user/preferences` | Yes | Fetch user preferences |
| `PUT` | `/user/preferences` | Yes | Update user preferences |
| `GET` | `/user/analyses` | Yes | List saved analyses |
| `POST` | `/user/analyses` | Yes | Save a new analysis |
| `GET` | `/user/analyses/{id}` | Yes | Fetch one saved analysis |
| `DELETE` | `/user/analyses/{id}` | Yes | Delete a saved analysis |
| `GET` | `/user/history` | Yes | Paginated activity log |
| `GET` | `/user/sessions` | Yes | List active sessions |
| `DELETE` | `/user/sessions/{id}` | Yes | Revoke a session |
| `POST` | `/analyse` | Yes | Run ML analysis on a ticker list |
| `POST` | `/backtest` | Yes | Walk-forward backtest + optional model training |
| `GET` | `/cache/status` | Yes | List cached tickers and freshness |
| `DELETE` | `/cache/data` | Yes | Clear price cache (all or one ticker) |
| `DELETE` | `/cache/models` | Yes | Clear in-memory model cache |
| `GET` | `/model/info` | Yes | Info on saved and active models |
| `GET` | `/stock/{ticker}` | Yes | Raw data for a single ticker |

### POST `/analyse`

```json
{
  "tickers": ["AAPL", "MSFT", "NVDA", "SHEL.L", "CRH.IR"],
  "profile": "quality",
  "risk": "medium",
  "lookback_years": 5,
  "cv_folds": 5,
  "lambda_reg": 0.10
}
```

**Profiles:** `quality`, `value`, `growth`, `dividend`, `momentum`, `macro`, `activist`
**Risk:** `low`, `medium`, `high`

### POST `/backtest`

```json
{
  "tickers": ["AAPL", "MSFT", "NVDA", "JPM", "SHEL.L"],
  "profile": "quality",
  "lookback_years": 5,
  "forward_months": 12,
  "rebalance_months": 3,
  "train_model": true
}
```

Setting `train_model: true` (the default) trains the model on real historical returns after the backtest and saves it to disk. The next `/analyse` call will use this model automatically.

### Ticker format

- **US stocks:** `AAPL`, `MSFT`, `NVDA`
- **UK stocks (LSE):** `SHEL.L`, `AZN.L`, `HSBA.L`
- **Irish stocks (Euronext Dublin):** `AIB.IR`, `CRH.IR`, `BNK.IR`
- Any ticker valid on Yahoo Finance works

---

## How the ML Model Works

The model is a blended ensemble: Random Forest (60%) + Gradient Boosting (40%).

### Features

16 features are extracted per stock and rank-normalised to [0, 1] within the universe before training. Rank normalisation makes the model robust to outliers and scale differences between stocks.

| Category | Features |
|----------|----------|
| Valuation | P/E ratio, P/B ratio |
| Profitability | ROE, net margin |
| Growth | Revenue growth (YoY) |
| Safety | Debt/equity, dividend yield |
| Momentum | 1M, 3M, 6M, 12M price return |
| Volatility | 60-day annualised realised vol, beta |
| Technical | RSI-14, price vs 52-week high |
| Size | Log market cap |

### Training target

The model learns to predict which stocks will beat the equal-weight universe median return over a forward window. This is a binary label (1 = outperformed, 0 = underperformed), which avoids the noise of predicting absolute return levels.

### Cross-validation

TimeSeriesSplit is used — each fold's test set lies strictly after its training set, preventing any future data from leaking into training. Out-of-fold (OOF) predictions are used to compute training-period IC, which is a more honest estimate of skill than in-sample accuracy.

### Scoring

The model outputs a probability (0–1). This is adjusted for risk profile:

- **Risk penalty:** high-beta stocks are penalised in `low` and `medium` risk profiles
- **Composite score:** `clip(ML probability − risk penalty, 0.05, 0.97)`

**Signal thresholds:**

| Composite score | Signal |
|-----------------|--------|
| > 68% | STRONG BUY |
| > 54% | BUY |
| > 38% | HOLD |
| ≤ 38% | SELL |

### Fama-French 5-Factor Decomposition

Every stock receives a factor decomposition for display purposes: HML (value), RMW (quality), CMA (investment conservatism), SMB (size) and Momentum. These are derived from the stock's current fundamental snapshot and presented as estimated factor contributions in percentage terms.

---

## Walk-Forward Backtest

The backtest simulates how the model would have performed if run historically on your universe.

**Process:**

1. Price history is split into non-overlapping periods (default: 3-month rebalance, 12-month forward window)
2. At each period start, factor features are computed from price data available at that time only — no lookahead
3. Actual forward returns are computed for each stock over the next forward window
4. Label = 1 if the stock beat the universe median return that period, else 0
5. The final ~20% of periods are held out and never used for training

**Training:**

After collecting all (features, label) pairs, the model is trained on the training periods using 5-fold cross-validation. OOF predictions from the training set, and direct predictions on the holdout set, are used to compute all reported metrics.

**Backtest metrics explained:**

| Metric | What it means | Good threshold |
|--------|---------------|----------------|
| Hit Rate | % of periods where the model's top-half stocks beat the bottom half | > 55% |
| IC (Information Coefficient) | Spearman rank correlation between model scores and actual returns | > 0.05 |
| ICIR | IC / std(IC) — measures consistency of IC across periods | > 0.50 |
| Sharpe Ratio | Annualised return of top-quintile portfolio per unit of risk | > 1.0 |
| Max Drawdown | Worst peak-to-trough loss of the simulated portfolio | < 20% |
| Calmar Ratio | Annualised return / max drawdown | > 0.50 |
| Alpha | Excess return vs equal-weight benchmark | Positive = model adding value |
| Holdout IC | IC computed on the held-out periods the model never saw | Honest out-of-sample estimate |

> **Small universe note:** With fewer than 20 stocks, confidence intervals on IC and hit rate are wide. Use these metrics as directional signals rather than precise estimates. For best results: 10+ tickers, 3+ years of lookback, at least one completed backtest before acting on signals.

---

## Self-Improvement Loop

NUMKT is designed to improve the more historical data you feed it.

```
First /analyse   →  synthetic quality labels (factor-weighted heuristic)
                    good baseline, no market data needed

First /backtest  →  walks through price history, collects real return labels
                    trains model on what actually predicted outperformance
                    saves trained model to disk

All future /analyse calls  →  load saved model  →  predictions grounded
                                                    in real market outcomes
```

Each subsequent backtest with more tickers or longer history adds more training observations and retrains the model. The more real data the model has seen, the less it relies on the initial heuristic labels.

---

## Caching

The backend caches all Yahoo Finance data to disk to avoid hitting rate limits on repeat runs.

| Path | Contents | Format |
|------|----------|--------|
| `cache/prices/` | Daily OHLCV price history | `.parquet` — one file per ticker |
| `cache/info/` | Fundamental snapshot from Yahoo | `.json` — one file per ticker |
| `cache/models/` | Trained ensemble models | `.joblib` + `_meta.json` |

**Cache behaviour:**
- Cache is considered fresh for **7 days**
- Cache hit: loads from disk in ~50ms, no Yahoo request
- Cache miss: fetches from Yahoo (~3s per ticker) and saves to disk
- Tickers are fetched **sequentially with a 2s gap** when hitting Yahoo to avoid rate limiting
- Cached tickers skip the delay entirely
- If Yahoo returns no data on a fresh fetch, stale cache is used as a fallback rather than dropping the ticker

**Cache management (requires authentication):**

```bash
# What's cached
curl -b cookies.txt http://localhost:8000/cache/status

# Clear all price data (forces re-fetch from Yahoo)
curl -b cookies.txt -X DELETE http://localhost:8000/cache/data

# Clear one ticker
curl -b cookies.txt -X DELETE "http://localhost:8000/cache/data?ticker=AAPL"

# Clear in-memory model (forces retrain on next /analyse)
curl -b cookies.txt -X DELETE http://localhost:8000/cache/models
```

Set the `CACHE_DIR` environment variable to redirect the cache to a persistent volume if your host has an ephemeral filesystem (e.g. Render free tier).

---

## Deployment

The app is configured for Render via `render.yaml`. The frontend is served directly from FastAPI, so you only need to deploy the backend service.

**Render (one-click deploy):**

1. Fork the repo and connect it to Render
2. Render will read `render.yaml` and create the web service and database
3. Set `SECRET_KEY` as an environment variable (or let `render.yaml` generate it)
4. Run `alembic upgrade head` once after the first deploy (or add it as a build command)

**Other hosts:**

Any host that supports Python 3.11+ and PostgreSQL works. The `Procfile` contains the start command:

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

**Environment variables to set in production:**

```
DATABASE_URL=postgresql://...
SECRET_KEY=<32-byte hex>
ALLOWED_ORIGINS=https://your-domain.com   # only if frontend is on a separate domain
CACHE_DIR=/var/data/numkt-cache            # if using a persistent disk mount
```

---

## Configuration

| Location | Variable | Default | Description |
|----------|----------|---------|-------------|
| `index.html` | `BACKEND_URL` | `''` (same-origin) | Set to an absolute URL only if running a separate frontend domain |
| `.env` | `DATABASE_URL` | — | PostgreSQL connection string |
| `.env` | `SECRET_KEY` | — | Token signing key |
| `.env` | `CACHE_DIR` | `backend/cache/` | Price/info cache directory |
| `data_fetcher.py` | `CACHE_MAX_AGE_HOURS` | `168` (7 days) | Hours before cached data is considered stale |
| `data_fetcher.py` | `MIN_BARS` | `60` | Minimum price bars to treat a ticker as valid |
| `model.py` | `rf_weight` | `0.60` | Random Forest weight in the ensemble blend |
| Frontend sidebar | Lookback period | 5 years | Years of Yahoo Finance history to fetch |
| Frontend sidebar | L2 λ | 0.10 | Regularisation strength — higher = simpler model, less overfitting |
| Frontend sidebar | CV folds | 5 | TimeSeriesSplit folds for cross-validation |

---

## Security

- **Passwords** — bcrypt with per-password salts; never stored in plaintext
- **Sessions** — server-side session tokens stored as SHA-256 hashes in the database; the raw token lives only in an `HttpOnly; SameSite=Lax` cookie
- **Rate limiting** — `/analyse` is limited to 10 requests/minute per IP; `/backtest` to 3/minute (slowapi)
- **Security headers** — `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy`, `Content-Security-Policy`, and `Strict-Transport-Security` (HTTPS only) on every response
- **API docs disabled** — `/docs` and `/redoc` are not exposed in production
- **Authentication required** — all analysis, cache and model endpoints require a valid session; unauthenticated requests receive a 401
- **Timing-safe auth** — login always runs the bcrypt comparison (against a dummy hash if the user doesn't exist) to prevent user enumeration via response time

---

## Disclaimer

NUMKT is for educational and informational purposes only. The ML model uses historical market data and statistical factor heuristics. Signals do not constitute financial advice. Past model performance, simulated or otherwise, does not predict future investment results. Always consult a qualified financial adviser before making any investment decision.
