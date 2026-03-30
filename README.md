# NUMKT — Multi-Market ML Stock Analyser

> A self-improving ensemble ML stock analysis tool covering US, UK and Irish markets. Layers fundamental factors, Fama-French 5-factor quant analysis, macro overlays, and hedge fund 13F positioning signals to produce clear BUY, HOLD and SELL recommendations.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the App](#running-the-app)
- [Backend API Reference](#backend-api-reference)
- [How the ML Model Works](#how-the-ml-model-works)
- [Self-Improvement Loop](#self-improvement-loop)
- [Connecting a Live Backend](#connecting-a-live-backend)
- [Offline Mode](#offline-mode)
- [Caching](#caching)
- [Configuration](#configuration)
- [Disclaimer](#disclaimer)

---

## Overview

NUMKT has two modes:

**Offline mode** — open `index.html` directly in a browser. The frontend runs a local factor model using static fundamental data for 80+ stocks across US, UK and Irish markets. No backend required.

**Live mode** — connect the Python FastAPI backend to fetch real-time price and fundamental data from Yahoo Finance, train the ensemble model on actual historical returns, and run a walk-forward backtest to measure model quality.

---

## Project Structure

```
/
├── index.html              # Frontend — self-contained single-page app
│
└── backend/
    ├── main.py             # FastAPI application — entry point
    ├── requirements.txt    # Python dependencies
    ├── Procfile            # Process file for Railway / Heroku deployment
    ├── runtime.txt         # Python version pin for deployment
    └── ml/
        ├── __init__.py
        ├── data_fetcher.py         # Yahoo Finance fetcher with disk cache
        ├── feature_engineering.py  # Builds feature matrix from raw data
        ├── model.py                # RF + GB ensemble (trains on real returns)
        ├── backtest.py             # Walk-forward backtest + model training
        └── scoring.py              # Converts model output to BUY/HOLD/SELL signals
```

---

## Features

**Frontend**
- 80+ pre-loaded stocks across US, UK (LSE) and Irish (Euronext Dublin) markets
- Search and build a custom stock universe
- Seven investor profiles: Value, Growth, Dividend, Momentum, QARP, Global Macro, Activist
- Adjustable ML parameters: lookback period, L2 regularisation (λ), tree depth, CV folds, HF signal weight
- Toggle model layers on/off: HF 13F signals, macro overlay, Fama-French 5-factor, geopolitical risk
- BUY / HOLD / SELL signal table with colour-coded confidence bars
- Per-stock detail panel: factor decomposition, FF5 breakdown, HF positioning, macro note
- Feature importance chart, momentum simulation, sector exposure doughnut
- Walk-forward backtest panel with Hit Rate, IC, Sharpe, Drawdown, Alpha

**Backend**
- Real-time price and fundamental data via Yahoo Finance (`yfinance`)
- 24-hour disk cache — avoids rate limits on repeated runs
- Random Forest + Gradient Boosting ensemble trained on actual forward returns
- TimeSeriesSplit cross-validation (no lookahead bias)
- Model persists to disk between server restarts
- After each backtest, the model retrains on real outcomes and replaces synthetic labels
- REST API with health check, cache management and model introspection endpoints

---

## Getting Started

### Prerequisites

- **Python 3.11+** (for the backend)
- **pip**
- A modern browser (Chrome, Firefox, Safari, Edge) for the frontend

### Installation

**1. Clone the repo**

```bash
git clone https://github.com/your-org/numkt.git
cd numkt
```

**2. Install Python dependencies**

```bash
cd backend
pip install -r requirements.txt
```

`requirements.txt` should contain:

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
yfinance>=0.2.40
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.4.0
scipy>=1.12.0
joblib>=1.3.0
pyarrow>=15.0.0     # required for parquet cache
pydantic>=2.0.0
httpx>=0.27.0
```

> **Note:** `yfinance` frequently releases breaking changes. Pin to a known-good version if you hit data fetching issues (e.g. `yfinance==0.2.40`).

### Running the App

**Start the backend:**

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive Swagger UI.

**Open the frontend:**

Open `index.html` in your browser. The frontend is a self-contained HTML file with no build step required.

**Connect frontend to backend:**

In `index.html`, find this line near the top of the `<script>` block and update it to match your backend URL:

```javascript
const BACKEND_URL = 'http://localhost:8000';
```

Leave it as an empty string (`''`) to run in offline mode with local static data.

---

## Backend API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info + saved model summary |
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/analyse` | Run ML analysis on a list of tickers |
| `POST` | `/backtest` | Walk-forward backtest + optional model training |
| `GET` | `/cache/status` | List all cached tickers and freshness |
| `DELETE` | `/cache/data` | Clear price cache (all or one ticker) |
| `DELETE` | `/cache/models` | Clear in-memory model cache |
| `GET` | `/model/info` | Info on saved and active models |
| `GET` | `/stock/{ticker}` | Fetch raw data for a single ticker |

### POST `/analyse`

```json
{
  "tickers": ["AAPL", "MSFT", "NVDA"],
  "profile": "quality",
  "risk": "medium",
  "lookback_years": 5,
  "cv_folds": 5,
  "lambda_reg": 0.10,
  "use_macro": true,
  "use_hf_signals": true
}
```

**Profiles:** `value`, `growth`, `dividend`, `momentum`, `quality`, `macro`, `activist`  
**Risk:** `low`, `medium`, `high`

### POST `/backtest`

```json
{
  "tickers": ["AAPL", "MSFT", "NVDA", "JPM", "BRK.B"],
  "profile": "quality",
  "lookback_years": 5,
  "forward_months": 12,
  "rebalance_months": 3,
  "train_model": true
}
```

Setting `train_model: true` (the default) triggers model training on real historical returns after the backtest completes. The trained model is saved to disk and used automatically on the next `/analyse` call.

---

## How the ML Model Works

The model is a blended ensemble of Random Forest (60%) and Gradient Boosting (40%).

**Feature engineering** (`feature_engineering.py`) extracts 16 cross-sectionally rank-normalised features per stock:

| Category | Features |
|----------|----------|
| Valuation | P/E ratio, P/B ratio |
| Profitability | ROE, net margin |
| Growth | Revenue growth |
| Safety | Debt/equity, dividend yield |
| Momentum | 1M, 3M, 6M, 12M price return |
| Volatility | 60-day realised vol, beta |
| Technical | RSI-14, price vs 52-week high |
| Size | Log market cap |

Each feature is rank-normalised to [0, 1] within the current universe before being fed to the model. This makes predictions robust to outlier values and scale differences between stocks.

**Scoring** (`scoring.py`) adds three overlays on top of the model probability:

- **Macro adjustment** — sector-level adjustments based on Fed rate, VIX, GBP/USD, EUR/USD
- **HF signal adjustment** — sector-level hedge fund 13F positioning (accumulating / trimming)
- **Risk penalty** — penalises high-beta stocks in conservative risk profiles

**Signal thresholds:**

| Score | Signal |
|-------|--------|
| > 68% | STRONG BUY |
| > 54% | BUY |
| > 38% | HOLD |
| ≤ 38% | SELL |

---

## Self-Improvement Loop

NUMKT is designed to get better the more you use it.

1. **First run:** `/analyse` trains on synthetic quality labels (factor-based, no real returns)
2. **Run backtest:** `/backtest` with `train_model: true` collects (features, actual return) pairs across all historical periods
3. **Model trains on real outcomes:** which factor combinations actually predicted outperformance in your universe
4. **Model saved to disk:** survives server restarts
5. **All future `/analyse` calls use the real-return model** instead of synthetic labels

The more tickers and historical data you feed into the backtest, the more training observations are collected and the better the model becomes.

```
First /analyse  →  synthetic labels  →  good starting point
First /backtest →  real return data  →  model learns from history
Future /analyse →  trained model     →  increasingly accurate signals
```

---

## Connecting a Live Backend

To deploy on Railway (or any cloud host):

**1. Set the `BACKEND_URL` in `index.html`:**

```javascript
const BACKEND_URL = 'https://your-numkt-backend.up.railway.app';
```

**2. Ensure CORS is open** — the backend already sets `allow_origins=["*"]` for development. For production, restrict this to your frontend's domain:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    ...
)
```

**3. Verify connectivity:**

```bash
curl https://your-numkt-backend.up.railway.app/health
# Expected: {"status": "ok"}
```

The frontend header badge will show `LIVE DATA` (green) when the backend is reachable, and `OFFLINE` (grey) when it falls back to the local model.

---

## Offline Mode

Set `BACKEND_URL` to an empty string in `index.html`:

```javascript
const BACKEND_URL = '';
```

In offline mode the frontend uses static fundamental data embedded directly in `index.html` for 80+ stocks. All factor scoring, Fama-French decomposition, macro overlays and HF signals run locally in the browser. No internet connection is required.

Offline mode is useful for demos, development and situations where live market data is not needed.

---

## Caching

The backend caches all Yahoo Finance data to disk under `backend/cache/`:

| Path | Contents | Format |
|------|----------|--------|
| `cache/prices/` | Daily OHLCV price history | `.parquet` (one file per ticker) |
| `cache/info/` | Fundamental data from Yahoo | `.json` (one file per ticker) |
| `cache/models/` | Trained ensemble models | `.joblib` + `_meta.json` |

**Cache behaviour:**
- Cache is considered fresh for **24 hours**
- On a cache hit: data loads from disk in ~50ms with no Yahoo request
- On a cache miss: data is fetched from Yahoo (~3s per ticker) and saved to disk
- Tickers are fetched **sequentially** (1 per 2s) when hitting Yahoo to avoid rate limiting
- Already-cached tickers skip the delay entirely

**Clearing the cache:**

```bash
# Clear all price data
curl -X DELETE http://localhost:8000/cache/data

# Clear one ticker
curl -X DELETE "http://localhost:8000/cache/data?ticker=AAPL"

# Clear in-memory model (forces retrain on next /analyse)
curl -X DELETE http://localhost:8000/cache/models
```

---

## Configuration

| Location | Variable | Default | Description |
|----------|----------|---------|-------------|
| `index.html` | `BACKEND_URL` | `'http://localhost:8000'` | Backend URL. Set to `''` for offline mode |
| `data_fetcher.py` | `CACHE_MAX_AGE_HOURS` | `24` | Hours before cached data is considered stale |
| `data_fetcher.py` | `MIN_BARS` | `60` | Minimum price bars for a ticker to be considered valid |
| `model.py` | `rf_weight` | `0.60` | Weight given to Random Forest in the ensemble blend |
| `scoring.py` | `MACRO` | see file | Current macro environment values (Fed rate, VIX, FX rates) |
| Frontend sidebar | Lookback period | 3 years | How many years of data to fetch and use |
| Frontend sidebar | L2 λ | 0.10 | Regularisation strength — higher = simpler model |
| Frontend sidebar | CV folds | 5 | Number of TimeSeriesSplit folds for cross-validation |
| Frontend sidebar | HF signal weight | 20% | How much weight the HF positioning overlay adds |

---

## Disclaimer

NUMKT is for educational and informational purposes only. The ML model uses historical market data and factor-based heuristics. Signals do not constitute financial advice. Past model performance does not predict future investment results. Always consult a qualified financial adviser before making any investment decision.
