# Product Requirements Document (PRD)
# QuantForge — Systematic Strategies & Hedging Platform

> **Purpose of this document**: This PRD is designed to be given to Claude Code (or any AI coding agent) in VS Code to generate the full project structure, implement all modules, and create a production-ready application. Every section contains precise technical specifications, file paths, interfaces, and acceptance criteria.

---

## 1. Project Overview

### 1.1 What Is This?

**QuantForge** is a professional-grade quantitative finance platform that combines:

1. A **systematic strategy backtesting engine** — supporting 8+ portfolio strategies (hedging, allocation, signal-based)
2. A **derivatives pricing & hedging simulator** — forward-testing and back-testing delta-hedging portfolios for basket options under Black-Scholes
3. A **risk analytics module** — computing industry-standard risk metrics (Sharpe, VaR, Greeks, etc.)
4. A **beautiful interactive web dashboard** — where anyone (including non-technical recruiters) can explore strategies, run backtests, and compare performance

### 1.2 Origin & Motivation

This project is a **production-grade reimagining** of an academic project from Ensimag (Grenoble INP) — ".NET for Systematic Strategies" (2024). The original was a 1-week C# project focused on delta-hedging a single basket call option with simulated data. This version rebuilds everything from scratch in Python with a React frontend, extending it into a full quantitative platform that demonstrates hedge-fund-level engineering and financial knowledge.

### 1.3 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Quant Engine** | Python 3.12+, NumPy, SciPy, pandas |
| **API** | FastAPI (async, auto-generated OpenAPI docs) |
| **Frontend** | React 18 + TypeScript + Tailwind CSS + Recharts |
| **Data** | Yahoo Finance (yfinance), simulated Black-Scholes data |
| **Testing** | pytest, React Testing Library |
| **Deployment** | Vercel (frontend), Railway or Render (backend) |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker + docker-compose |

### 1.4 Repository Structure

```
quantforge/
├── README.md
├── docker-compose.yml
├── .github/
│   └── workflows/
│       ├── backend-ci.yml
│       └── frontend-ci.yml
│
├── backend/                          # Python FastAPI application
│   ├── pyproject.toml                # Dependencies (use uv or pip)
│   ├── Dockerfile
│   ├── main.py                       # FastAPI app entry point
│   ├── config.py                     # Settings, environment variables
│   │
│   ├── core/                         # Quant engine (pure Python, no web dependency)
│   │   ├── __init__.py
│   │   ├── models/                   # Data models & schemas
│   │   │   ├── __init__.py
│   │   │   ├── market_data.py        # MarketData, OHLCV, DataFeed
│   │   │   ├── portfolio.py          # Portfolio, Position, Trade
│   │   │   ├── options.py            # Option, BasketOption, VanillaOption
│   │   │   └── results.py           # BacktestResult, RiskMetrics, PricingResult
│   │   │
│   │   ├── data/                     # Data providers (abstracted)
│   │   │   ├── __init__.py
│   │   │   ├── base.py              # Abstract IDataProvider interface
│   │   │   ├── yahoo.py             # Yahoo Finance real market data
│   │   │   ├── simulated.py         # Black-Scholes simulated data generator
│   │   │   └── csv_loader.py        # Load from CSV files
│   │   │
│   │   ├── pricing/                  # Option pricing engine
│   │   │   ├── __init__.py
│   │   │   ├── black_scholes.py     # BS closed-form + Greeks
│   │   │   ├── monte_carlo.py       # MC pricer (multi-asset, variance reduction)
│   │   │   └── utils.py             # Cholesky decomposition, random generation
│   │   │
│   │   ├── strategies/               # All systematic strategies (pluggable)
│   │   │   ├── __init__.py
│   │   │   ├── base.py              # Abstract IStrategy interface
│   │   │   ├── registry.py          # Strategy registry (discover & list all strategies)
│   │   │   │
│   │   │   ├── hedging/             # === HEDGING STRATEGIES ===
│   │   │   │   ├── __init__.py
│   │   │   │   ├── delta_hedge.py   # Delta hedging on basket options
│   │   │   │   └── delta_gamma_hedge.py
│   │   │   │
│   │   │   ├── allocation/          # === PORTFOLIO ALLOCATION ===
│   │   │   │   ├── __init__.py
│   │   │   │   ├── equal_weight.py  # 1/N equal weight
│   │   │   │   ├── min_variance.py  # Markowitz min-variance
│   │   │   │   ├── max_sharpe.py    # Tangency portfolio
│   │   │   │   └── risk_parity.py   # Risk parity (equal risk contribution)
│   │   │   │
│   │   │   └── signal/              # === SIGNAL-BASED ===
│   │   │       ├── __init__.py
│   │   │       ├── momentum.py      # Cross-sectional momentum
│   │   │       └── mean_reversion.py # Z-score / Bollinger band mean reversion
│   │   │
│   │   ├── backtester/               # Backtesting engine
│   │   │   ├── __init__.py
│   │   │   ├── engine.py            # Core backtest loop
│   │   │   ├── rebalancing.py       # Rebalancing oracles (periodic, threshold, etc.)
│   │   │   └── costs.py             # Transaction cost models
│   │   │
│   │   └── risk/                     # Risk analytics
│   │       ├── __init__.py
│   │       ├── metrics.py           # Sharpe, Sortino, max drawdown, Calmar, etc.
│   │       ├── var.py               # VaR, CVaR (historical, parametric, MC)
│   │       └── greeks.py            # Greeks computation (delta, gamma, vega, theta, rho)
│   │
│   ├── api/                          # FastAPI routes
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── strategies.py        # GET /strategies, GET /strategies/{id}
│   │   │   ├── backtest.py          # POST /backtest
│   │   │   ├── hedging.py           # POST /hedging/simulate
│   │   │   ├── risk.py              # POST /risk/analyze
│   │   │   ├── data.py              # GET /data/assets, GET /data/prices
│   │   │   └── pricing.py           # POST /pricing/option
│   │   └── schemas.py               # Pydantic request/response models
│   │
│   └── tests/
│       ├── __init__.py
│       ├── test_black_scholes.py
│       ├── test_monte_carlo.py
│       ├── test_strategies.py
│       ├── test_backtester.py
│       ├── test_risk_metrics.py
│       └── test_api.py
│
├── frontend/                         # React + TypeScript application
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── vite.config.ts
│   ├── Dockerfile
│   ├── index.html
│   │
│   ├── public/
│   │   └── favicon.ico
│   │
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── index.css                 # Tailwind imports
│       │
│       ├── api/                      # API client
│       │   ├── client.ts            # Axios/fetch wrapper
│       │   └── types.ts            # TypeScript types matching backend schemas
│       │
│       ├── components/               # Reusable UI components
│       │   ├── layout/
│       │   │   ├── Navbar.tsx
│       │   │   ├── Sidebar.tsx
│       │   │   └── Footer.tsx
│       │   ├── charts/
│       │   │   ├── EquityCurve.tsx
│       │   │   ├── DrawdownChart.tsx
│       │   │   ├── CompositionChart.tsx
│       │   │   ├── CorrelationHeatmap.tsx
│       │   │   ├── GreeksSurface.tsx
│       │   │   └── RollingMetrics.tsx
│       │   ├── forms/
│       │   │   ├── StrategyConfigurator.tsx
│       │   │   ├── BacktestForm.tsx
│       │   │   ├── HedgingForm.tsx
│       │   │   └── AssetSelector.tsx
│       │   └── common/
│       │       ├── MetricCard.tsx
│       │       ├── LoadingSpinner.tsx
│       │       ├── ErrorBanner.tsx
│       │       └── DataTable.tsx
│       │
│       ├── pages/                    # Route-level pages
│       │   ├── HomePage.tsx
│       │   ├── StrategyExplorerPage.tsx
│       │   ├── BacktestResultsPage.tsx
│       │   ├── StrategyComparisonPage.tsx
│       │   ├── RiskAnalyticsPage.tsx
│       │   └── HedgingSimulatorPage.tsx
│       │
│       ├── hooks/                    # Custom React hooks
│       │   ├── useBacktest.ts
│       │   ├── useStrategies.ts
│       │   └── useRiskMetrics.ts
│       │
│       └── utils/
│           ├── formatters.ts        # Number/date formatting
│           └── colors.ts            # Chart color palette
│
└── docs/
    ├── ARCHITECTURE.md
    ├── STRATEGIES.md                 # Detailed explanation of each strategy
    └── API.md                        # API documentation
```

---

## 2. Core Quant Engine Specifications

### 2.1 Data Layer (`core/data/`)

#### Abstract Interface (`base.py`)

```python
from abc import ABC, abstractmethod
from datetime import date
import pandas as pd

class IDataProvider(ABC):
    """Abstract data provider. The backtester and strategies
    depend on this interface, NEVER on a concrete implementation.
    This is a key design principle from the original project."""

    @abstractmethod
    def get_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Return a DataFrame with DatetimeIndex and one column per symbol,
        containing adjusted close prices."""
        ...

    @abstractmethod
    def get_risk_free_rate(self, date: date) -> float:
        """Return the annualized risk-free rate for the given date."""
        ...
```

#### Yahoo Finance Provider (`yahoo.py`)

- Uses `yfinance` library to fetch real OHLCV data
- Implements `IDataProvider`
- Caches data locally (in-memory or SQLite) to avoid repeated API calls
- Default universe: top 20 S&P 500 stocks by market cap (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK-B, JPM, V, JNJ, WMT, PG, MA, HD, UNH, DIS, BAC, XOM, PFE)
- Risk-free rate: fetched from ^IRX (13-week Treasury bill) or hardcoded fallback

#### Simulated Data Provider (`simulated.py`)

- Generates synthetic price paths using geometric Brownian motion (multi-asset, correlated)
- Parameters: initial prices `S0[]`, volatilities `σ[]`, correlation matrix `Σ`, drift `μ[]`, risk-free rate `r`, number of time steps, time horizon `T`
- Uses Cholesky decomposition of the correlation matrix for correlated paths
- Implements `IDataProvider` with the same interface
- This is the provider used for forward-testing hedging strategies (matching the original Ensimag project)

### 2.2 Pricing Engine (`core/pricing/`)

#### Black-Scholes Closed-Form (`black_scholes.py`)

Implement the following for European vanillas:

```python
class BlackScholesModel:
    """Closed-form BS pricing and Greeks."""

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float: ...

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float: ...

    @staticmethod
    def delta(S, K, T, r, sigma, option_type: str) -> float: ...

    @staticmethod
    def gamma(S, K, T, r, sigma) -> float: ...

    @staticmethod
    def vega(S, K, T, r, sigma) -> float: ...

    @staticmethod
    def theta(S, K, T, r, sigma, option_type: str) -> float: ...

    @staticmethod
    def rho(S, K, T, r, sigma, option_type: str) -> float: ...

    @staticmethod
    def implied_volatility(price, S, K, T, r, option_type: str) -> float:
        """Newton-Raphson solver for implied vol."""
        ...
```

#### Monte Carlo Pricer (`monte_carlo.py`)

```python
class MonteCarloPricer:
    """Multi-asset MC pricer with variance reduction."""

    def __init__(
        self,
        n_simulations: int = 100_000,
        seed: int | None = None,
        variance_reduction: str = "antithetic"  # "none", "antithetic", "control_variate"
    ): ...

    def price_basket_option(
        self,
        spots: np.ndarray,         # (n_assets,) current prices
        weights: np.ndarray,       # (n_assets,) basket weights ω_i
        strike: float,             # K
        maturity: float,           # T in years
        risk_free_rate: float,     # r
        volatilities: np.ndarray,  # (n_assets,) σ_i
        correlation: np.ndarray,   # (n_assets, n_assets) correlation matrix
    ) -> PricingResult:
        """Price a basket call: (Σ ω_i S_i^T - K)^+
        Returns price, std error, confidence interval, and deltas."""
        ...

    def compute_deltas(
        self,
        # same params as above, plus:
        bump_size: float = 0.01    # 1% bump for finite difference
    ) -> np.ndarray:
        """Compute delta of each underlying via finite differences."""
        ...
```

**Variance reduction techniques to implement:**
- **Antithetic variates**: for each random draw Z, also use -Z
- **Control variates**: use the geometric basket (has closed-form) as control

### 2.3 Strategy Framework (`core/strategies/`)

#### Abstract Interface (`base.py`)

This is the **most important design decision** in the project. Every strategy implements this interface.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
import numpy as np

@dataclass
class StrategyConfig:
    """Parameters that configure a strategy. Each strategy subclass
    defines its own config with additional fields."""
    name: str
    description: str
    rebalancing_frequency: str = "weekly"  # "daily", "weekly", "monthly", "threshold"
    transaction_cost_bps: float = 10.0     # basis points per trade

@dataclass
class PortfolioWeights:
    """Output of a strategy's weight computation."""
    weights: dict[str, float]   # symbol -> weight (0 to 1, sum to 1 for long-only)
    cash_weight: float          # remaining weight in risk-free asset
    timestamp: date

class IStrategy(ABC):
    """Abstract systematic strategy interface.

    A systematic strategy determines:
    1. WHEN to rebalance (RebalancingTime)
    2. HOW to compute portfolio composition (UpdateCompo)

    Constraint: self-financing — portfolio value before and after
    rebalancing must be equal. No look-ahead bias allowed.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config

    @abstractmethod
    def compute_weights(
        self,
        current_date: date,
        price_history: pd.DataFrame,   # all prices up to current_date (NO future data)
        current_portfolio: PortfolioWeights | None,
    ) -> PortfolioWeights:
        """Compute new target weights. Must only use data <= current_date."""
        ...

    @property
    @abstractmethod
    def required_history_days(self) -> int:
        """Minimum number of historical trading days needed before
        this strategy can start computing weights."""
        ...

    @classmethod
    @abstractmethod
    def get_param_schema(cls) -> dict:
        """Return a JSON-schema-like dict describing configurable parameters.
        Used by the frontend to dynamically render configuration forms."""
        ...
```

#### Strategy Registry (`registry.py`)

```python
class StrategyRegistry:
    """Auto-discovers and registers all strategy implementations.
    Used by the API to list available strategies and instantiate them."""

    _strategies: dict[str, type[IStrategy]] = {}

    @classmethod
    def register(cls, strategy_class: type[IStrategy]):
        """Decorator to register a strategy."""
        cls._strategies[strategy_class.__name__] = strategy_class
        return strategy_class

    @classmethod
    def list_strategies(cls) -> list[dict]:
        """Return metadata about all registered strategies."""
        ...

    @classmethod
    def create(cls, name: str, config: dict) -> IStrategy:
        """Instantiate a strategy by name with given config."""
        ...
```

#### Strategy Implementations

**1. Delta Hedging (`hedging/delta_hedge.py`)**

The core of the original Ensimag project. Given a basket call option:
- **Payoff**: `(Σ ω_i S_i^T - K)^+`
- At each rebalancing date, use the MC pricer to compute deltas
- Hold `δ_i` shares of asset `i`, rest in risk-free asset
- Track portfolio value vs. theoretical option price over time
- Key output: **tracking error** = std(V_portfolio - V_option) / V_option

Config params: `option_weights`, `strike`, `maturity`, `n_simulations`, `rebalancing_frequency`

**2. Delta-Gamma Hedging (`hedging/delta_gamma_hedge.py`)**

Extends delta hedging by also matching gamma using a second option instrument. Shows more sophisticated risk management.

**3. Equal Weight (`allocation/equal_weight.py`)**

- Invest `V_t / (n * S_i^t)` shares of each asset at each rebalancing
- From the course slides (Example 1: uniform strategy)
- Config: `rebalancing_frequency` (default: weekly on Mondays)

**4. Minimum Variance (`allocation/min_variance.py`)**

- Solve: `min ω^T Σ ω` subject to `Σω_i = 1`, `ω_i >= 0`
- Uses `scipy.optimize.minimize` with SLSQP
- Covariance matrix estimated from rolling window of returns
- Config: `lookback_window` (days), `rebalancing_frequency`

**5. Maximum Sharpe (`allocation/max_sharpe.py`)**

- Solve: `max (ω^T μ - r) / sqrt(ω^T Σ ω)` subject to `Σω_i = 1`, `ω_i >= 0`
- Also known as the tangency portfolio
- Config: `lookback_window`, `risk_free_rate_override` (optional), `rebalancing_frequency`

**6. Risk Parity (`allocation/risk_parity.py`)**

- Each asset contributes equally to total portfolio risk
- Solve: `min Σ (RC_i - 1/n)^2` where `RC_i = ω_i * (Σω)_i / (ω^T Σ ω)`
- Config: `lookback_window`, `rebalancing_frequency`

**7. Momentum (`signal/momentum.py`)**

- Cross-sectional momentum: rank assets by trailing return
- Go long top `k` assets, equal-weight among winners
- Optional: long top k, short bottom k (long-short version)
- Config: `lookback_period` (default: 252 days), `top_k` (number of winners), `long_only` (bool), `rebalancing_frequency`

**8. Mean Reversion (`signal/mean_reversion.py`)**

- Compute z-score of each asset: `z_i = (S_i - MA_i) / std_i`
- Buy assets with z < -threshold, sell/avoid assets with z > +threshold
- Weight inversely proportional to z-score magnitude
- Config: `lookback_window` (default: 20 days), `z_threshold` (default: 2.0), `rebalancing_frequency`

### 2.4 Backtesting Engine (`core/backtester/`)

#### Core Loop (`engine.py`)

This directly mirrors the pseudocode from the course slides:

```python
@dataclass
class BacktestConfig:
    initial_value: float = 100_000.0
    start_date: date = None
    end_date: date = None
    data_provider: str = "yahoo"      # "yahoo" or "simulated"
    symbols: list[str] = None
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0

@dataclass
class BacktestResult:
    """Complete output of a backtest run."""
    portfolio_values: pd.Series          # date -> portfolio value
    benchmark_values: pd.Series | None   # date -> benchmark (e.g., equal-weight)
    weights_history: pd.DataFrame        # date x symbol -> weight at each rebalancing
    trades_log: list[dict]               # every trade executed
    risk_metrics: dict                   # computed at the end
    config: BacktestConfig
    strategy_name: str
    computation_time_ms: float

class BacktestEngine:
    """
    Main backtest loop. Implements Algorithm 2 from the course:

    1. Initialize portfolio with V_t0
    2. For each date t > t0:
       a. Update portfolio value: V_t = Σ q_i * S_i^t + cash * (1 + r*dt)
       b. If rebalancing_time(t): compute new composition
    3. Return portfolio value series

    CONSTRAINTS:
    - Self-financing: value before/after rebalancing is the same
    - No look-ahead bias: strategy only sees data up to current date
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(
        self,
        strategy: IStrategy,
        data_provider: IDataProvider,
    ) -> BacktestResult:
        """Execute backtest and return full results."""
        ...
```

#### Rebalancing Oracles (`rebalancing.py`)

```python
class RebalancingOracle(ABC):
    @abstractmethod
    def should_rebalance(self, current_date: date, portfolio: Portfolio) -> bool: ...

class PeriodicRebalancing(RebalancingOracle):
    """Rebalance every N trading days, or on specific weekdays."""
    def __init__(self, frequency: str): ...  # "daily", "weekly", "monthly"

class ThresholdRebalancing(RebalancingOracle):
    """Rebalance when any weight drifts beyond a threshold from target."""
    def __init__(self, threshold: float = 0.05): ...
```

#### Transaction Costs (`costs.py`)

```python
class TransactionCostModel:
    """Realistic cost model. Apply after each trade."""

    def __init__(
        self,
        commission_bps: float = 10.0,    # 10 bps = 0.1%
        slippage_bps: float = 5.0,       # market impact
        min_commission: float = 1.0,     # minimum $1 per trade
    ):
        ...

    def compute_cost(self, trade_value: float) -> float:
        """Return total cost for a trade of given notional value."""
        ...
```

### 2.5 Risk Analytics (`core/risk/`)

#### Performance Metrics (`metrics.py`)

Implement all of the following. Each takes a `pd.Series` of portfolio values or returns:

| Metric | Formula / Description |
|--------|----------------------|
| **Total Return** | `(V_T - V_0) / V_0` |
| **Annualized Return** | `(1 + total_return)^(252/N) - 1` |
| **Annualized Volatility** | `std(daily_returns) * sqrt(252)` |
| **Sharpe Ratio** | `(ann_return - r_f) / ann_vol` |
| **Sortino Ratio** | `(ann_return - r_f) / downside_vol` |
| **Max Drawdown** | `max(peak - trough) / peak` |
| **Calmar Ratio** | `ann_return / abs(max_drawdown)` |
| **Win Rate** | `% of positive return days` |
| **Profit Factor** | `sum(gains) / abs(sum(losses))` |
| **Tracking Error** | `std(portfolio_return - benchmark_return) * sqrt(252)` |
| **Information Ratio** | `(ann_return - bench_return) / tracking_error` |
| **Turnover** | Average total absolute weight change per rebalancing |

#### Value at Risk (`var.py`)

```python
class VaRCalculator:
    def historical_var(self, returns: pd.Series, confidence: float = 0.95) -> float: ...
    def parametric_var(self, returns: pd.Series, confidence: float = 0.95) -> float: ...
    def historical_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float: ...
    def rolling_var(self, returns: pd.Series, window: int = 252, confidence: float = 0.95) -> pd.Series: ...
```

#### Greeks (`greeks.py`)

For hedging strategies, compute and expose:

```python
class GreeksCalculator:
    """Compute Greeks for the hedged option at each time step."""

    def compute_greeks_surface(
        self,
        spot_range: np.ndarray,    # range of spot prices
        vol_range: np.ndarray,     # range of volatilities
        strike: float,
        maturity: float,
        risk_free_rate: float,
    ) -> dict:
        """Return a grid of delta, gamma, vega, theta for
        (spot, vol) pairs. Used for 3D surface visualization."""
        ...
```

---

## 3. API Specifications

### 3.1 FastAPI Application (`main.py`)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="QuantForge API",
    description="Systematic strategies backtesting & hedging platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 3.2 API Endpoints

#### `GET /api/strategies`

List all available strategies with their metadata and configurable parameters.

**Response:**
```json
{
  "strategies": [
    {
      "id": "delta_hedge",
      "name": "Delta Hedging",
      "family": "hedging",
      "description": "Replicates a basket call option using delta hedging",
      "params": {
        "strike": { "type": "number", "default": 100, "min": 1 },
        "maturity_years": { "type": "number", "default": 1.0, "min": 0.1, "max": 5.0 },
        "n_simulations": { "type": "integer", "default": 50000, "min": 1000 },
        "rebalancing_frequency": { "type": "select", "options": ["daily", "weekly", "monthly"], "default": "weekly" }
      }
    }
  ]
}
```

#### `POST /api/backtest`

Run a backtest for a given strategy and configuration.

**Request:**
```json
{
  "strategy_id": "min_variance",
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"],
  "start_date": "2020-01-01",
  "end_date": "2024-12-31",
  "initial_value": 100000,
  "params": {
    "lookback_window": 60,
    "rebalancing_frequency": "monthly"
  },
  "data_source": "yahoo"
}
```

**Response:**
```json
{
  "portfolio_values": { "2020-01-02": 100000, "2020-01-03": 100234, ... },
  "benchmark_values": { "2020-01-02": 100000, "2020-01-03": 100150, ... },
  "weights_history": {
    "2020-01-02": { "AAPL": 0.25, "MSFT": 0.20, ... },
    ...
  },
  "risk_metrics": {
    "total_return": 0.87,
    "annualized_return": 0.134,
    "annualized_volatility": 0.18,
    "sharpe_ratio": 0.72,
    "sortino_ratio": 1.05,
    "max_drawdown": -0.33,
    "calmar_ratio": 0.41,
    "var_95": -0.023,
    "cvar_95": -0.035,
    "win_rate": 0.53,
    "turnover": 0.12
  },
  "trades_log": [...],
  "computation_time_ms": 1243
}
```

#### `POST /api/backtest/compare`

Run multiple strategies on the same data for comparison.

**Request:**
```json
{
  "strategies": [
    { "strategy_id": "equal_weight", "params": {} },
    { "strategy_id": "min_variance", "params": { "lookback_window": 60 } },
    { "strategy_id": "momentum", "params": { "lookback_period": 126, "top_k": 3 } }
  ],
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"],
  "start_date": "2020-01-01",
  "end_date": "2024-12-31",
  "initial_value": 100000
}
```

**Response:** Array of `BacktestResult` objects (same schema as single backtest).

#### `POST /api/hedging/simulate`

Run a hedging forward-test or backtest (the core of the original Ensimag project).

**Request:**
```json
{
  "option_type": "basket_call",
  "weights": [0.3, 0.3, 0.4],
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "strike": 100,
  "maturity_years": 1.0,
  "risk_free_rate": 0.05,
  "volatilities": [0.2, 0.25, 0.3],
  "correlation_matrix": [[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]],
  "initial_spots": [150, 280, 140],
  "n_simulations": 50000,
  "rebalancing_frequency": "weekly",
  "data_source": "simulated",
  "n_paths": 5
}
```

**Response:**
```json
{
  "paths": [
    {
      "portfolio_values": [...],
      "option_prices": [...],
      "tracking_errors": [...],
      "deltas_history": [...]
    }
  ],
  "average_tracking_error": 0.023,
  "initial_option_price": 12.45,
  "initial_option_price_ci": [12.10, 12.80]
}
```

#### `POST /api/risk/analyze`

Compute detailed risk metrics on a given return series.

#### `GET /api/data/assets`

Return list of available assets with metadata.

#### `GET /api/data/prices?symbols=AAPL,MSFT&start=2020-01-01&end=2024-12-31`

Fetch historical prices.

#### `POST /api/pricing/option`

Price a single option (BS or MC).

---

## 4. Frontend Specifications

### 4.1 Design System

**Visual identity:**
- Dark theme primary (professional quant aesthetic), with light theme toggle
- Color palette: deep navy (`#0f172a`) background, electric blue (`#3b82f6`) accents, emerald green (`#10b981`) for gains, red (`#ef4444`) for losses
- Font: Inter (headings) + JetBrains Mono (numbers, code)
- Charts: Recharts with consistent color scheme

**Layout:**
- Fixed left sidebar navigation (collapsible)
- Top bar with project name + GitHub link
- Main content area with responsive grid

### 4.2 Pages

#### Page 1: Home (`HomePage.tsx`)

**Purpose:** Landing page that explains the project and invites exploration.

**Content:**
- Hero section: project title "QuantForge", one-line description, CTA buttons ("Explore Strategies", "Run a Backtest")
- 3 feature cards: "8 Strategies", "Real Market Data", "Risk Analytics"
- Architecture diagram (embedded SVG or image)
- Quick stats: "Run a backtest in under 3 seconds", etc.

#### Page 2: Strategy Explorer (`StrategyExplorerPage.tsx`)

**Purpose:** Browse all strategies, learn what each does, configure parameters.

**Content:**
- Left: list of strategy cards grouped by family (Hedging / Allocation / Signal)
- Right: when a strategy is selected, show:
  - Description, mathematical formulation (rendered LaTeX via KaTeX)
  - Parameter configuration form (dynamically generated from `get_param_schema()`)
  - Asset universe selector (multi-select with search)
  - Date range picker
  - "Run Backtest" button

#### Page 3: Backtest Results (`BacktestResultsPage.tsx`)

**Purpose:** Display comprehensive results after running a backtest.

**Content (in order):**
1. **Summary bar**: total return, annualized return, Sharpe ratio, max drawdown — in 4 metric cards
2. **Equity curve chart**: portfolio value over time vs. benchmark (equal-weight SPY), with log scale toggle
3. **Drawdown chart**: underwater chart showing drawdown % over time
4. **Rolling metrics chart**: rolling 30d Sharpe, rolling 30d volatility
5. **Portfolio composition over time**: stacked area chart showing weight of each asset
6. **Full metrics table**: all risk metrics in a clean table
7. **Trades log**: paginated table of all trades with date, symbol, quantity, price, cost

#### Page 4: Strategy Comparison (`StrategyComparisonPage.tsx`)

**Purpose:** Run 2-5 strategies side-by-side on the same data and compare.

**Content:**
1. **Strategy selector**: pick 2-5 strategies with their configs
2. **Shared settings**: same symbols, date range, initial capital
3. **"Compare" button**
4. **Results panel:**
   - Overlaid equity curves (all strategies on one chart, different colors)
   - Comparison table: rows = metrics, columns = strategies
   - Drawdown comparison chart
   - Bar chart of key metrics (Sharpe, max drawdown, return) side by side

#### Page 5: Risk Analytics (`RiskAnalyticsPage.tsx`)

**Purpose:** Deep-dive into risk metrics for a given backtest result.

**Content:**
1. **VaR/CVaR visualization**: histogram of daily returns with VaR and CVaR lines
2. **Rolling VaR chart**: 252-day rolling VaR over time
3. **Correlation heatmap**: correlation matrix of selected assets
4. **Return distribution**: histogram + fitted normal distribution overlay
5. **Stress test scenarios**: predefined scenarios (2008 crisis, COVID crash, 2022 rate hikes) — show how the strategy would have performed

#### Page 6: Hedging Simulator (`HedgingSimulatorPage.tsx`)

**Purpose:** This is the "original project, elevated." Interactive forward-test/backtest of delta hedging on basket options.

**Content:**
1. **Configuration panel:**
   - Basket option definition: select assets, set weights ω_i, strike K, maturity T
   - Market parameters: volatilities, correlations, risk-free rate
   - Hedging parameters: rebalancing frequency, number of MC simulations
   - Data source toggle: "Simulated" (forward-test) vs "Historical" (backtest)
   - Number of simulation paths (for forward-test)

2. **Simulation results:**
   - **Main chart**: portfolio value vs. theoretical option price over time (multiple paths if forward-test)
   - **Tracking error chart**: difference between portfolio and option price
   - **Delta evolution**: chart showing delta of each asset over time
   - **P&L decomposition**: breakdown of P&L from delta, gamma, theta, transaction costs
   - **Summary statistics**: initial option price (with CI), average tracking error, final P&L

3. **Greeks visualization:**
   - Delta surface: 3D chart of delta as function of spot price and time-to-maturity
   - Gamma, Vega, Theta surfaces (toggle between them)

---

## 5. Testing Requirements

### 5.1 Backend Tests (`backend/tests/`)

**Minimum test coverage: 80%.**

#### Pricing Tests (`test_black_scholes.py`)
- BS call/put prices match known analytical values (e.g., S=100, K=100, T=1, r=0.05, σ=0.2 → call ≈ 10.45)
- Put-call parity holds: `C - P = S - K*exp(-rT)`
- Greeks match finite-difference approximations
- Implied vol round-trips: `price → implied_vol → price` matches

#### Monte Carlo Tests (`test_monte_carlo.py`)
- MC price converges to BS price for single-asset European call (within 2 standard errors)
- Antithetic variates reduce variance vs. plain MC
- Basket option price is positive and reasonable
- Deltas are between 0 and 1 for call options

#### Strategy Tests (`test_strategies.py`)
- Equal-weight strategy produces equal weights
- Min-variance weights sum to 1 and are non-negative
- Momentum strategy selects top-k performers
- No strategy uses future data (test by running with truncated data)

#### Backtester Tests (`test_backtester.py`)
- Portfolio is self-financing: value before rebalancing = value after rebalancing (within floating point tolerance)
- Transaction costs reduce portfolio value
- Backtest result dates match input date range

#### Risk Tests (`test_risk_metrics.py`)
- Sharpe ratio of a constant return series is well-defined
- Max drawdown of a monotonically increasing series is 0
- VaR at 99% > VaR at 95%

### 5.2 Frontend Tests

- Component rendering tests for all pages
- Form validation tests for strategy configurator
- Chart rendering with mock data

---

## 6. Deployment

### 6.1 Docker

**`backend/Dockerfile`:**
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`frontend/Dockerfile`:**
```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json .
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
```

**`docker-compose.yml`:**
```yaml
version: "3.8"
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - PYTHONDONTWRITEBYTECODE=1
  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
```

### 6.2 Live Deployment

- **Frontend**: Deploy to Vercel (connect GitHub repo, auto-deploy on push)
- **Backend**: Deploy to Railway or Render (connect GitHub repo, auto-deploy on push)
- **Environment variables**: `API_URL` in frontend pointing to backend URL
- **CORS**: Configure backend to allow frontend origin

### 6.3 CI/CD (`.github/workflows/`)

**`backend-ci.yml`:**
- Trigger on push to `main` and PRs
- Steps: install Python 3.12, install deps, run `pytest --cov=core --cov-fail-under=80`, lint with `ruff`

**`frontend-ci.yml`:**
- Trigger on push to `main` and PRs
- Steps: install Node 20, `npm ci`, `npm run lint`, `npm run build`, `npm test`

---

## 7. Implementation Order (Recommended Phases)

### Phase 1: Core Engine Foundation (Week 1)
- [ ] Set up repository structure, pyproject.toml, basic configs
- [ ] Implement data models (`core/models/`)
- [ ] Implement `IDataProvider` + `SimulatedDataProvider`
- [ ] Implement `BlackScholesModel` (pricing + all Greeks)
- [ ] Implement `MonteCarloPricer` (with antithetic variates)
- [ ] Write pricing tests (`test_black_scholes.py`, `test_monte_carlo.py`)

### Phase 2: Strategy Framework (Week 2)
- [ ] Implement `IStrategy` interface and `StrategyRegistry`
- [ ] Implement `EqualWeightStrategy`
- [ ] Implement `MinVarianceStrategy`
- [ ] Implement `MaxSharpeStrategy`
- [ ] Implement `RiskParityStrategy`
- [ ] Implement `MomentumStrategy`
- [ ] Implement `MeanReversionStrategy`
- [ ] Implement `DeltaHedgeStrategy`
- [ ] Implement `DeltaGammaHedgeStrategy`
- [ ] Write strategy tests

### Phase 3: Backtesting Engine (Week 2-3)
- [ ] Implement `BacktestEngine` core loop
- [ ] Implement `RebalancingOracle` variants
- [ ] Implement `TransactionCostModel`
- [ ] Implement `YahooDataProvider`
- [ ] Write backtester tests
- [ ] Validate self-financing constraint in tests

### Phase 4: Risk Analytics (Week 3)
- [ ] Implement all performance metrics
- [ ] Implement VaR/CVaR calculators
- [ ] Implement Greeks surface calculator
- [ ] Write risk tests

### Phase 5: FastAPI Backend (Week 3)
- [ ] Set up FastAPI app with CORS
- [ ] Implement all API endpoints
- [ ] Implement Pydantic schemas
- [ ] Write API tests
- [ ] Generate OpenAPI documentation

### Phase 6: React Frontend (Week 3-4)
- [ ] Set up React + TypeScript + Vite + Tailwind
- [ ] Build layout components (Navbar, Sidebar)
- [ ] Build reusable chart components
- [ ] Build HomePage
- [ ] Build StrategyExplorerPage
- [ ] Build BacktestResultsPage
- [ ] Build StrategyComparisonPage
- [ ] Build RiskAnalyticsPage
- [ ] Build HedgingSimulatorPage

### Phase 7: Polish & Deploy (Week 4)
- [ ] Docker setup + docker-compose
- [ ] Deploy backend to Railway/Render
- [ ] Deploy frontend to Vercel
- [ ] CI/CD pipelines
- [ ] README with screenshots, architecture diagram, live demo link
- [ ] Final testing and bug fixes

---

## 8. Key Design Principles

1. **Data source independence**: The backtester and strategies must NEVER depend on a specific data provider. Always code against `IDataProvider`.

2. **No look-ahead bias**: This is critical. At time `t`, a strategy can only access data from times `≤ t`. The backtester must enforce this by passing only historical data to `compute_weights()`.

3. **Self-financing constraint**: When the portfolio is rebalanced, the total value before and after must be equal. The difference is traded (buying/selling shares), and transaction costs are deducted.

4. **Strategy as a plugin**: Adding a new strategy should require only creating a new file in the appropriate directory and decorating the class with `@StrategyRegistry.register`. No changes to the backtester or API needed.

5. **Separation of concerns**: `core/` has zero web framework dependencies. It's a pure Python library that could be used in a Jupyter notebook, CLI, or any other context. The `api/` layer is just a thin wrapper.

6. **Performance**: Backtests with 5 years of daily data and 5 assets should complete in under 5 seconds. Use NumPy vectorization, avoid Python loops on time series.

---

## 9. README Template

The README.md should include:

1. **Project title + one-line description**
2. **Live demo link** (the deployed URL)
3. **Screenshot** of the dashboard (Strategy Comparison page)
4. **Features** section (brief)
5. **Architecture diagram**
6. **Tech stack** badges
7. **Quick start** (docker-compose up, or manual setup)
8. **Strategies** section (table with all 8 strategies)
9. **API documentation** link (FastAPI /docs)
10. **Testing** instructions
11. **Acknowledgments** (mention Ensimag, Professor Mnacho Echenim's course)

---

*End of PRD. This document contains all specifications needed to implement the QuantForge platform from scratch.*