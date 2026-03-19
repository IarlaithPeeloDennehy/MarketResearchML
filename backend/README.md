# NUMKT ML Backend

Real Python ML backend for the NUMKT multi-market stock analyser.

Fetches live data from Yahoo Finance, trains a Random Forest + Gradient
Boosting ensemble with proper TimeSeriesSplit cross-validation, and
returns scored BUY/HOLD/SELL signals with industry-standard backtest metrics.

---

## How it connects to your GitHub Pages frontend

```
GitHub Pages          Railway (this repo)       Yahoo Finance
NUMKT.html      →     /analyse   (FastAPI)  →   yfinance (free)
                →     /backtest
                ←     JSON signals + metrics
```

Your HTML already has everything wired up. You just need to:
1. Deploy this backend to Railway
2. Paste the Railway URL into NUMKT.html (one line change)

---

## Step 1 — Add this backend to your existing GitHub repo

You have two choices:

### Option A — Same repo as your frontend (recommended)
Put the backend in a subfolder of your existing repo:

```
your-github-repo/
├── NUMKT.html          ← your existing frontend (GitHub Pages serves this)
├── backend/            ← add this folder
│   ├── main.py
│   ├── requirements.txt
│   ├── Procfile
│   ├── runtime.txt
│   └── ml/
│       ├── __init__.py
│       ├── data_fetcher.py
│       ├── feature_engineering.py
│       ├── model.py
│       ├── scoring.py
│       └── backtest.py
```

Copy the contents of this zip into a `backend/` folder in your repo,
commit and push to GitHub.

### Option B — Separate repo
Create a new GitHub repo just for the backend and push these files to it.
Railway can deploy from either repo.

---

## Step 2 — Deploy to Railway (free tier)

Railway gives you 500 free hours/month — enough for personal use.

1. Go to **railway.app** and sign up (use "Sign in with GitHub")

2. Click **New Project** → **Deploy from GitHub repo**

3. Select your repo (and set the **Root Directory** to `backend/`
   if you used Option A above)

4. Railway auto-detects the Procfile and starts deploying

5. Once deployed, click your service → **Settings** → **Networking**
   → **Generate Domain**. Copy the URL — it looks like:
   `https://numkt-backend-production.up.railway.app`

---

## Step 3 — Connect frontend to backend (one line)

Open **NUMKT.html** in your editor. Near the top of the `<script>` block, find:

```javascript
const BACKEND_URL = ''; // replace with Railway URL after deploy
```

Replace with your Railway URL:

```javascript
const BACKEND_URL = 'https://numkt-backend-production.up.railway.app';
```

Commit and push. Your GitHub Pages site will now call the real ML backend.

---

## Step 4 — Verify it works

Open your GitHub Pages URL. Run an analysis. You should see:

- **LIVE DATA** badge in the header (green) instead of OFFLINE
- A teal dot (●) next to each ticker in the results table
- Real CV accuracy from the actual trained model (not simulated)
- Feature importance bars showing what the RF+GB model actually learned
- **Run Backtest** button now produces real walk-forward metrics

---

## Local development

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000/docs for the interactive API docs (FastAPI Swagger UI).

To test locally from your frontend, temporarily set:
```javascript
const BACKEND_URL = 'http://localhost:8000';
```

Then open NUMKT.html directly in your browser (or via a local server).

---

## API Endpoints

| Method | Endpoint        | Description                                    |
|--------|----------------|------------------------------------------------|
| GET    | /               | Service info                                   |
| GET    | /health         | Health check (used by frontend to detect live) |
| POST   | /analyse        | Main ML analysis — returns ranked signals      |
| POST   | /backtest       | Walk-forward backtest — returns quality metrics|
| GET    | /stock/{ticker} | Raw fundamentals for one ticker                |
| GET    | /model/info     | Cached model info                              |
| DELETE | /model/cache    | Clear model cache (force retrain)              |

### /analyse request body
```json
{
  "tickers": ["AAPL", "MSFT", "NVDA", "SHEL.L", "CRH"],
  "profile": "quality",
  "risk": "medium",
  "lookback_years": 3,
  "cv_folds": 5,
  "lambda_reg": 0.10,
  "use_macro": true,
  "use_hf_signals": true
}
```

### /backtest request body
```json
{
  "tickers": ["AAPL", "MSFT", "NVDA", "JPM", "JNJ", "SHEL.L"],
  "profile": "quality",
  "lookback_years": 5,
  "forward_months": 12
}
```

---

## Backtest metrics explained

| Metric          | What it means                          | Good threshold      |
|----------------|----------------------------------------|---------------------|
| Hit Rate        | % of BUY periods that beat benchmark  | > 55%               |
| IC (Info. Coef.)| Rank correlation: score vs returns     | > 0.05 meaningful   |
| ICIR            | IC / std(IC) — consistency            | > 0.5 consistent    |
| Sharpe Ratio    | Return per unit of risk               | > 1.0 good          |
| Max Drawdown    | Worst peak-to-trough loss             | < 20% acceptable    |
| Calmar Ratio    | Annualised return / max drawdown      | > 0.5 good          |
| Alpha           | Excess return vs equal-weight universe| Positive = value add|

---

## Supported tickers

Any ticker valid on Yahoo Finance works. Format:
- **US stocks**: `AAPL`, `MSFT`, `GOOGL`
- **UK stocks**: `SHEL.L`, `AZN.L`, `HSBA.L` (add `.L` suffix)
- **Irish stocks**: `AIB.IR`, `BNK.IR` (add `.IR` suffix)
- **ETFs**: `SPY`, `QQQ`, `VUSA.L`

---

## Notes on model accuracy

With a small universe (<20 stocks), the backtest confidence intervals are
wide — treat IC and hit rate as directional signals, not precise estimates.

The model is trained on cross-sectional factor rankings, not price prediction.
It learns which factor combinations (value, quality, momentum) have historically
been associated with relative outperformance within the universe provided.

For best results: use 10+ tickers, 3+ years lookback, and run the backtest
before acting on any signal.
