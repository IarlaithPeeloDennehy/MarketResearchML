# NUMKT ML Backend




```
your-github-repo/
├── index.html          
├── backend/            
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
