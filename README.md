# QuantForge

A production-grade quantitative finance platform for pricing options, backtesting systematic strategies, and computing risk analytics — built in Python and React.

---

## Overview

QuantForge is a full-stack quant engine inspired by an academic delta-hedging project (Ensimag). It reimplements and extends the original .NET system in Python, adding a modern REST API and interactive dashboard.

```
systematicStrategies/
├── backend/          # Python quant engine + FastAPI
│   ├── core/         # Pure Python — no web deps
│   │   ├── models/   # Data models (Portfolio, Options, Results…)
│   │   ├── data/     # Data providers (Simulated, CSV, Yahoo)
│   │   ├── pricing/  # Black-Scholes, Monte Carlo
│   │   ├── strategies/  # 8 systematic strategies  ← Phase 2
│   │   ├── backtester/  # Backtest engine          ← Phase 3
│   │   └── risk/        # VaR, Greeks, metrics     ← Phase 4
│   └── api/          # FastAPI routes               ← Phase 5
├── frontend/         # React 18 dashboard           ← Phase 6
└── docs/
    └── PRD.md        # Full product requirements
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Quant Engine | Python 3.10+, NumPy, SciPy, pandas |
| API | FastAPI + uvicorn |
| Frontend | React 18 + TypeScript + Vite + Tailwind CSS + Recharts |
| Data | yfinance (real), GBM simulation (synthetic), CSV (Ensimag format) |
| Testing | pytest + pytest-cov |
| Containers | Docker + docker-compose ← Phase 7 |

---

## Current Status — Phase 1 complete

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Core engine foundation | ✅ Done |
| Phase 2 | Strategy framework (8 strategies) | 🔲 Planned |
| Phase 3 | Backtesting engine | 🔲 Planned |
| Phase 4 | Risk analytics (VaR, Greeks) | 🔲 Planned |
| Phase 5 | FastAPI backend | 🔲 Planned |
| Phase 6 | React frontend | 🔲 Planned |
| Phase 7 | Docker + CI/CD + deployment | 🔲 Planned |

### What's implemented in Phase 1

**Data Models** (`core/models/`)
- `DataFeed`, `OHLCV` — market data primitives
- `Position`, `Portfolio` — portfolio snapshot with mark-to-market valuation
- `VanillaOption`, `BasketOption` — option contracts
- `PricingResult`, `BacktestResult` — output containers

**Data Layer** (`core/data/`)
- `IDataProvider` — abstract interface (strategies never depend on concrete providers)
- `SimulatedDataProvider` — correlated GBM paths via Cholesky decomposition
- `CsvDataProvider` — Ensimag-format CSV loader (`Id`, `DateOfPrice`, `Value`)

**Pricing Utilities** (`core/pricing/`)
- `cholesky_decompose` — Cholesky factor with positive-definite guard
- `generate_correlated_normals` — correlated N(0,1) draws

**Black-Scholes Model** (`core/pricing/black_scholes.py`)
- `call_price`, `put_price` (put-call parity)
- All Greeks: `delta`, `gamma`, `vega`, `theta`, `rho`
- `implied_volatility` via Newton-Raphson

---

## Quickstart

### Prerequisites
- Python 3.10+
- pip or uv

### Backend setup

```bash
cd backend

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=core --cov-report=term-missing
```

### REPL quick demo

```python
from core.pricing.black_scholes import BlackScholesModel
from core.pricing.monte_carlo import MonteCarloPricer   # Phase 1 in progress
from core.data import SimulatedDataProvider
import numpy as np
from datetime import date

# Black-Scholes — ATM call
print(BlackScholesModel.call_price(100, 100, 1, 0.05, 0.2))  # ~10.45

# Simulate correlated price paths
provider = SimulatedDataProvider(
    spots={"AAPL": 150.0, "MSFT": 300.0},
    volatilities={"AAPL": 0.25, "MSFT": 0.20},
    correlation=np.array([[1.0, 0.6], [0.6, 1.0]]),
    seed=42,
)
df = provider.get_prices(["AAPL", "MSFT"], date(2024, 1, 2), date(2024, 12, 31))
print(df.head())
```

---

## Key Design Principles

**No look-ahead bias**
`BacktestEngine` only passes `prices.loc[:t]` to `strategy.compute_weights()` at date `t`.

**Self-financing**
Portfolio value before and after rebalancing must be equal — transaction costs are deducted from cash, not created from thin air.

**Data source independence**
Strategies and the backtester always code against `IDataProvider`, never against `YahooDataProvider` or `SimulatedDataProvider` directly. Swap data sources with zero strategy code change.

**Strategy as a plugin**
Registering a new strategy requires only creating a file and decorating the class with `@StrategyRegistry.register`.

**`core/` has zero web dependencies**
The quant engine works standalone in a notebook or CLI — no FastAPI, no uvicorn.

---

## Strategies (Phase 2)

| Family | Strategy |
|--------|----------|
| Hedging | Delta Hedge, Delta-Gamma Hedge |
| Allocation | Equal Weight, Min Variance, Max Sharpe, Risk Parity |
| Signal | Momentum, Mean Reversion |

---

## License

MIT
