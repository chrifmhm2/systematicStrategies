# QuantForge

A full-stack quantitative finance platform — backtest systematic strategies on live market data, analyse risk, price options, and simulate delta hedging. Built from scratch in Python + React.

---

## What you get

| Page | What it does |
|------|-------------|
| **Home** | Live equity-curve preview (3 strategies auto-run on load) |
| **Backtest** | Run any of the 8 strategies on real or simulated data, full metrics + charts |
| **Comparison** | 2–4 strategies side by side — overlaid curves, metric table |
| **Risk** | VaR 95%, CVaR, Sharpe, Sortino, Calmar, max drawdown, Beta, Alpha |
| **Delta Hedging** | Replicate basket option payoffs across Monte Carlo paths |
| **Option Pricing** | Black-Scholes or Monte Carlo, full Greeks, strike sweep |

---

## Prerequisites

| Tool | Minimum | Notes |
|------|---------|-------|
| Python | 3.10 | 3.12 recommended |
| Node.js | 18 | Use [nvm](https://github.com/nvm-sh/nvm) if your system node is old |
| npm | 9+ | Comes with Node |
| git | any | |

Internet access is required on first run — the backend fetches price data from Yahoo Finance.

---

## Quick start (full stack)

### 1 — Clone

```bash
git clone <repo-url> systematicStrategies
cd systematicStrategies
```

### 2 — Backend

```bash
cd backend

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install the package and all dependencies
pip install -e ".[dev]"

# Start the API server
uvicorn main:app --reload --port 8000
```

The API will be live at **http://localhost:8000**
Interactive docs (Swagger UI) at **http://localhost:8000/docs**

### 3 — Frontend (new terminal)

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** — the app proxies all `/api` requests to the backend on `:8000`, so no CORS setup is needed.

> **Node version issue?**
> If `npm install` fails because your system Node is too old (e.g. v12), install a recent version via nvm:
> ```bash
> nvm install 24
> nvm use 24
> npm install
> npm run dev
> ```

---

## Platform tour

Once both servers are running, visit **http://localhost:5173**:

1. **Home page** loads immediately and auto-runs a live backtest comparison — you will see an equity curve appear after a few seconds while data is fetched from Yahoo Finance.

2. Go to **Backtest** to run a single strategy:
   - Pick a strategy from the dropdown (grouped by family)
   - Add asset tickers (e.g. `AAPL`, `MSFT`, `GOOGL`)
   - Set a date range and click **Run Backtest**
   - Results: 8 metric cards, equity curve, drawdown, portfolio weights, trades log

3. Go to **Comparison** to run 2–4 strategies on the same universe in one shot.

4. Go to **Risk** for the full 14-metric risk report + return distribution histogram.

5. Go to **Hedging** to simulate delta (or delta-gamma) hedging of a basket option.

6. Go to **Pricing** for Black-Scholes / Monte Carlo pricing, Greeks, and a 21-point strike sweep.

---

## Available strategies

| Family | ID | Description |
|--------|----|-------------|
| allocation | `equal_weight` | Rebalance to equal weights every period |
| allocation | `min_variance` | Minimum variance portfolio (QP) |
| allocation | `max_sharpe` | Maximum Sharpe ratio portfolio (QP) |
| allocation | `risk_parity` | Equal risk contribution |
| signal | `momentum` | Buy top-N assets by trailing return |
| signal | `mean_reversion` | Buy bottom-N assets (reversal) |
| hedging | `delta_hedge` | Delta-neutral replication |
| hedging | `delta_gamma_hedge` | Delta + gamma neutral replication |

---

## Project structure

```
systematicStrategies/
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Settings / env vars
│   ├── core/                # Pure Python quant engine (no web deps)
│   │   ├── models/          # DataFeed, Portfolio, Options, Results
│   │   ├── data/            # IDataProvider, Simulated, Yahoo, CSV
│   │   ├── pricing/         # Black-Scholes, Monte Carlo
│   │   ├── strategies/      # IStrategy, StrategyRegistry, 8 strategies
│   │   ├── backtester/      # BacktestEngine, rebalancing, costs
│   │   └── risk/            # PerformanceMetrics, VaR, Greeks
│   ├── api/
│   │   ├── schemas.py       # Pydantic request / response models
│   │   └── routes/          # strategies, backtest, hedging, risk, pricing
│   └── tests/               # pytest suite (116 tests, 80%+ coverage)
└── frontend/
    └── src/
        ├── api/             # Axios client + TypeScript types
        ├── components/      # layout/, charts/, common/
        ├── pages/           # 6 route-level pages
        ├── hooks/           # useBacktest, useStrategies, useRiskMetrics
        └── utils/           # formatters, colors
```

---

## Development

### Backend tests

```bash
cd backend
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=core --cov-fail-under=80

# Lint
ruff check .
```

### Frontend build check

```bash
cd frontend
npm run build    # TypeScript compile + Vite bundle (must be 0 errors)
npm run lint     # ESLint
```

### API docs

With the backend running, visit **http://localhost:8000/docs** for the full Swagger UI — you can call every endpoint directly from the browser without touching the frontend.

---

## Adding a new strategy

1. Create `backend/core/strategies/<family>/<your_strategy>.py`
2. Decorate the class with `@StrategyRegistry.register`
3. Implement `compute_weights(prices_df) -> pd.Series`
4. Define `param_schema` as a class variable (the UI will auto-render a form for it)
5. Restart uvicorn — the strategy appears immediately in the dropdown

No frontend code changes required.

---

## Status

| Phase | What it builds | Status |
|-------|---------------|--------|
| 1 | Core models, Black-Scholes, Monte Carlo, simulated data | ✅ Done |
| 2 | 8 strategies, StrategyRegistry | ✅ Done |
| 3 | Backtest engine, rebalancing oracles, Yahoo data | ✅ Done |
| 4 | Risk metrics, VaR/CVaR, Greeks calculator | ✅ Done |
| 5 | FastAPI + all 6 route files + Pydantic schemas | ✅ Done |
| 6 | React dashboard — 6 pages, charts, dynamic forms | ✅ Done |
| 7 | Docker + docker-compose, CI/CD, deployment | 🔲 Planned |

---

## Key design principles

**No look-ahead bias** — `BacktestEngine` passes only `prices.loc[:t]` to the strategy at date `t`.

**Self-financing** — portfolio value is identical before and after rebalancing; transaction costs come out of cash.

**Data-source independence** — strategies code against `IDataProvider`; swap Yahoo for simulated data with one parameter.

**Plugin architecture** — one decorated class = one new strategy; `param_schema` surfaces automatically in the UI.

**`core/` has zero web dependencies** — use the quant engine standalone in a notebook or CLI, no FastAPI needed.

---

## License

MIT
