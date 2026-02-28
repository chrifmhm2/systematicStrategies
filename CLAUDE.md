# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Session Continuity

At the **start of every new session**, before doing anything else:
1. Read all files in `.claude/sessions/` in order (session1, session2, …)
2. Load the context — what was built, decisions made, current state, what comes next
3. Confirm to the user which session you loaded and what the next step is

Session files follow the naming convention:
```
.claude/sessions/session1_phase1_complete.md
.claude/sessions/session2_phase2_complete.md
...
```

---

## Learning Convention

When the user prefixes a message with **`[Learn]`**, treat it as a learning question:
1. Answer it **briefly and clearly** in the chat
2. Save the Q&A to `.claude/learn/<topic>.md` (create the file if new, append if it already exists)
3. Keep entries concise — one question, one short answer, a code snippet only if essential

---

## Repository Layout

```
systematicStrategies/
├── 01_Projet_DotNET/        # Original Ensimag academic project (.NET 8.0 / C#)
├── backend/                 # QuantForge quant engine + FastAPI (Python 3.12)
├── frontend/                # QuantForge dashboard (React 18 + TypeScript + Tailwind)
├── docs/
│   └── PRD.md               # Full product requirements for QuantForge
└── .claude/
    ├── phases/              # Incremental implementation plans (Phase 1–7)
    └── sessions/            # Session summaries / context snapshots
```

---

## QuantForge — New Python/React Platform

This is the active project being built. Full spec is in [docs/PRD.md](docs/PRD.md). Implementation is split into 7 phases tracked in `.claude/phases/`.

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Quant Engine | Python 3.12, NumPy, SciPy, pandas |
| API | FastAPI + uvicorn |
| Frontend | React 18 + TypeScript + Vite + Tailwind CSS + Recharts |
| Data | yfinance (real), GBM simulation (synthetic) |
| Testing | pytest + React Testing Library |
| Containers | Docker + docker-compose |
| CI/CD | GitHub Actions |

### Build & Run (QuantForge)

```bash
# Backend — install and run dev server
cd backend
pip install -e ".[dev]"
uvicorn main:app --reload --port 8000
# API docs at http://localhost:8000/docs

# Backend — run tests
pytest tests/ -v
pytest tests/ --cov=core --cov-fail-under=80

# Backend — lint
ruff check .

# Frontend — install and run dev server
cd frontend
npm install
npm run dev      # http://localhost:5173 (proxies /api → localhost:8000)
npm run build
npm run lint

# Full stack — Docker
docker-compose up --build
# Frontend: http://localhost:3000 | Backend docs: http://localhost:8000/docs
```

### QuantForge Architecture

```
backend/
├── main.py                  # FastAPI entry point
├── config.py                # Settings / env vars
├── core/                    # Pure Python quant engine (no web deps)
│   ├── models/              # DataFeed, Portfolio, BasketOption, PricingResult, BacktestResult
│   ├── data/                # IDataProvider, SimulatedDataProvider, YahooDataProvider, CsvDataProvider
│   ├── pricing/             # BlackScholesModel, MonteCarloPricer, utils
│   ├── strategies/          # IStrategy, StrategyRegistry, 8 concrete strategies
│   │   ├── hedging/         # DeltaHedgeStrategy, DeltaGammaHedgeStrategy
│   │   ├── allocation/      # EqualWeight, MinVariance, MaxSharpe, RiskParity
│   │   └── signal/          # Momentum, MeanReversion
│   ├── backtester/          # BacktestEngine, RebalancingOracle, TransactionCostModel
│   └── risk/                # PerformanceMetrics, VaRCalculator, GreeksCalculator
└── api/
    ├── schemas.py            # Pydantic request/response models
    └── routes/              # strategies, backtest, hedging, risk, data, pricing

frontend/src/
├── api/                     # Axios client + TypeScript types
├── components/              # layout/, charts/, forms/, common/
├── pages/                   # 6 route-level pages
├── hooks/                   # useBacktest, useStrategies, useRiskMetrics
└── utils/                   # formatters, colors
```

### Key Design Constraints

- **No look-ahead bias**: `BacktestEngine` only passes `prices.loc[:t]` to `strategy.compute_weights()` at date `t`.
- **Self-financing**: portfolio value before and after rebalancing must be equal (transaction costs deducted from cash, not from thin air).
- **Data source independence**: strategies and the backtester always code against `IDataProvider`, never `YahooDataProvider` or `SimulatedDataProvider` directly.
- **Strategy as a plugin**: registering a new strategy requires only creating a file and decorating the class with `@StrategyRegistry.register`.
- **`core/` has zero web deps**: it must work standalone in a notebook or CLI.

### Phase Plan

Incremental implementation order tracked in `.claude/phases/`:

| File | Phase | What it builds |
|------|-------|----------------|
| `phase1_core_engine_foundation.md` | 1 | Data models, Black-Scholes, Monte Carlo, simulated data provider |
| `phase2_strategy_framework.md` | 2 | `IStrategy`, registry, all 8 strategies |
| `phase3_backtesting_engine.md` | 3 | Backtest loop, rebalancing oracles, transaction costs, Yahoo data |
| `phase4_risk_analytics.md` | 4 | All metrics, VaR/CVaR, Greeks surface |
| `phase5_fastapi_backend.md` | 5 | FastAPI app, all 6 route files, Pydantic schemas |
| `phase6_react_frontend.md` | 6 | React app, all 6 pages, charts, forms, hooks |
| `phase7_polish_and_deploy.md` | 7 | Docker, CI/CD, Railway + Vercel deployment, README |

Each TODO has a unique label (`[P1-01]`, `[P2-15a]`, …) for precise reference.

---

## Original .NET Project (Ensimag Academic)

Located in `01_Projet_DotNET/`. Source code under `01_Projet_DotNET/src/`.

### Build & Run (.NET)

```bash
dotnet build 01_Projet_DotNET/src/BacktestConsole.sln
dotnet build 01_Projet_DotNET/src/BacktestConsole.sln -c Release

BacktestConsole.exe <params.json> <market-data.csv> <output.json>
dotnet run --project 01_Projet_DotNET/src/GrpcBacktestServer/
dotnet run --project 01_Projet_DotNET/src/GrpcEvaluation/GrpcEvaluation/ <params.json> <market-data.csv> <output.json>

# Automated evaluation (Python)
python.exe 01_Projet_DotNET/tests/EvaluationScript/generate_backtest_results.py \
  --sln=<abs-path> --tests=<abs-path> --out=<abs-path> --build=<abs-path> --force
```

Test data in `01_Projet_DotNET/data/Test_*_*/` (each folder: `params_*.json`, `data_*.csv`, `resultat.json`).

### .NET Projects

| Project | Type | Role |
|---|---|---|
| `HeadingCalculLibrary` | Library | Core hedging algorithm, rebalancing, portfolio state |
| `BacktestConsole` | Executable | CLI: reads files → runs backtest → writes JSON |
| `GrpcBacktestServer` | ASP.NET gRPC Server | `BacktestRunner.RunBacktest` RPC on `localhost:5000` |
| `GrpcEvaluation` | gRPC Client | Same pipeline as console but via gRPC |

### .NET Data Flow

```
params.json + market-data.csv
    ↓ JSONToTestParameters / CsvToListDataFeed
AlgoSystematicStrategie.ExecuterAlogSystematicStrategie()
    ├─ Pricer.Price()              [PricingLibrary 2.0.5 — external NuGet]
    ├─ DetectRebalancingFactory.IsRebalancing()
    └─ Portfolio.GetPortfolioValue()
    ↓
ListOuputToJSON → output.json
```

### .NET Notes

- `ConvertisseurFile/` is duplicated between `BacktestConsole` and `GrpcEvaluation` — sync changes in both.
- `PricingLibrary` 2.0.5 is external; source not in repo.
- Architecture diagrams in `01_Projet_DotNET/res/conception/`.
