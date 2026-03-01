# Phase 1 — Core Engine: Everything You Need to Know

---

## What Phase 1 builds

Phase 1 is the pure quant engine — zero web dependencies, works standalone in a notebook or CLI.
It is divided into 3 modules: **Models**, **Data**, **Pricing**.

```
backend/core/
├── models/
│   ├── market_data.py    → DataFeed, OHLCV
│   ├── options.py        → VanillaOption, BasketOption
│   ├── portfolio.py      → Position, Portfolio
│   └── results.py        → PricingResult, BacktestResult
├── data/
│   ├── base.py           → IDataProvider (abstract interface)
│   └── simulated.py      → SimulatedDataProvider (GBM price paths)
└── pricing/
    ├── black_scholes.py  → BlackScholesModel (5 Greeks + IV)
    ├── monte_carlo.py    → MonteCarloPricer (basket option + deltas)
    └── utils.py          → cholesky_decompose, generate_correlated_normals
```

**Test coverage**: 46 tests, all passing.

---

## Module 1 — Models (data containers)

Models are pure data structures. They hold values, do no computation.

### DataFeed and OHLCV (`market_data.py`)

```python
@dataclass
class DataFeed:
    symbol: str   # ticker, e.g. "AAPL"
    date: date    # observation date
    price: float  # adjusted close price
```

`OHLCV` is the full candlestick bar (Open, High, Low, Close, Volume). Used when you need more than just closing prices.

### VanillaOption and BasketOption (`options.py`)

```python
@dataclass
class VanillaOption:
    underlying: str             # single stock ticker
    strike: float               # K — strike price
    maturity: float             # T in years
    option_type: "call"|"put"

@dataclass
class BasketOption:
    underlyings: list[str]      # e.g. ["AAPL", "MSFT", "GOOG"]
    weights: list[float]        # ωᵢ, must sum to 1
    strike: float               # K
    maturity: float             # T
    option_type: "call"|"put"
```

**Basket payoff**: `max(Σ ωᵢ · Sᵢᵀ − K, 0)` — Black-Scholes has NO closed-form for this, hence we need Monte Carlo.

### Portfolio (`portfolio.py`)

```python
@dataclass
class Position:
    symbol: str
    quantity: float   # number of shares held
    price: float      # last known price

@dataclass
class Portfolio:
    positions: dict[str, Position]   # symbol → Position
    cash: float = 0.0
    date: date = today

    def total_value(self, prices: dict[str, float]) -> float:
        equity = Σ(quantity × current_price for each position)
        return equity + cash
```

`total_value()` implements **mark-to-market** valuation — the portfolio is worth whatever you'd get if you sold everything right now.

### PricingResult and BacktestResult (`results.py`)

`PricingResult` is the output of the Monte Carlo pricer:
```python
@dataclass
class PricingResult:
    price: float                        # fair value estimate
    std_error: float                    # how uncertain the MC estimate is
    confidence_interval: tuple[float, float]  # 95% CI
    deltas: np.ndarray                  # ∂V/∂Sᵢ per asset
```

`BacktestResult` is the output of a full backtest (filled in Phase 3):
```python
@dataclass
class BacktestResult:
    portfolio_values: dict[str, float]      # NAV over time
    benchmark_values: dict[str, float]      # buy-and-hold comparison
    weights_history: dict[str, dict]        # weights at each rebalancing
    trades_log: list[dict]                  # every trade made
    risk_metrics: dict                      # Sharpe, max drawdown, etc.
    computation_time_ms: float
```

---

## Module 2 — Data (price sources)

### IDataProvider — the abstract interface (`data/base.py`)

```python
class IDataProvider(ABC):
    @abstractmethod
    def get_prices(self, symbols, start_date, end_date) -> pd.DataFrame:
        """Returns: DatetimeIndex rows, one column per symbol, values = adjusted close."""

    @abstractmethod
    def get_risk_free_rate(self, date) -> float:
        """Returns annualised rate, e.g. 0.05 for 5%."""
```

**Key design decision**: strategies and the backtester ALWAYS code against `IDataProvider`, never against a concrete class. This means you can swap the data source (simulated → Yahoo → Bloomberg) without touching a single strategy.

### SimulatedDataProvider — GBM price simulation (`data/simulated.py`)

Generates synthetic correlated price paths using Geometric Brownian Motion.

**Constructor parameters**:
- `spots`: initial price per symbol
- `volatilities`: annual volatility per symbol
- `correlation`: correlation matrix (must be positive definite)
- `drift`: annual drift μ (default 0.0)
- `risk_free_rate`: fixed rate returned by `get_risk_free_rate()`
- `seed`: for reproducibility

**What it does internally**:
```
1. Cholesky(correlation) → L  (lower-triangular matrix)
2. For each step t:
   a. Draw ε ~ N(0, I)       (n_assets independent normals)
   b. Correlate: Z = ε @ L.T
   c. log_return = (μ - σ²/2)*dt + σ*√dt * Z
3. prices = S₀ * exp(cumsum(log_returns))  (GBM paths)
```

**Why GBM?** It's the standard model. Key properties:
- Prices are always positive (exponential of log-returns)
- Returns are normally distributed (log-normal prices)
- Future uncertainty grows with √T (square-root of time)

---

## Module 3 — Pricing

### BlackScholesModel (`pricing/black_scholes.py`)

**All methods are static** — you never instantiate the class:
```python
BlackScholesModel.call_price(S=100, K=100, T=1, r=0.05, sigma=0.2)  # → ~10.45
```

The core formulas:
```
d1 = [ ln(S/K) + (r + σ²/2)·T ] / (σ·√T)
d2 = d1 − σ·√T

Call = S·N(d1) − K·e^(−rT)·N(d2)
Put  = Call − S + K·e^(−rT)          (put-call parity)
```

**The 5 Greeks implemented**:

| Greek | Formula | Meaning |
|-------|---------|---------|
| **Δ (Delta)** | `N(d1)` call, `N(d1)-1` put | $1 move in S → Δ$ move in option. Shares to hedge. |
| **Γ (Gamma)** | `n(d1) / (S·σ·√T)` | How fast delta changes. Always positive. High near expiry ATM. |
| **ν (Vega)** | `S·n(d1)·√T` | Sensitivity to vol. Always positive. Options are bets on volatility. |
| **Θ (Theta)** | (formula in code) / 365 | Daily time decay. Almost always negative. |
| **ρ (Rho)** | `K·T·e^(−rT)·N(d2)` call | Sensitivity to interest rates. Positive for calls. |

**Implied Volatility — Newton-Raphson**:
```
σ_new = σ_old − (BS(σ_old) − market_price) / Vega(σ_old)
```
Repeat until |error| < 1e-6. Vega is the derivative because it measures how BS price moves with σ.

### MonteCarloPricer (`pricing/monte_carlo.py`)

Used when there's no closed-form formula (basket options).

**Algorithm**:
```
1. Draw N correlated paths of terminal prices:
   S_i^T = S_i · exp((r - σ_i²/2)·T  +  σ_i·√T · Z_i)
   where Z = ε @ L.T  (Cholesky-correlated)

2. Compute basket payoff per path:
   payoff = max(Σ ωᵢ · S_i^T − K, 0)

3. Discount and average:
   price = mean(payoffs) · e^(−rT)

4. Standard error:
   std_error = std(payoffs) / √N
```

**Antithetic Variates** (variance reduction):
```
For each draw Z, also compute payoff with −Z.
Return average of the pair: (payoff(+Z) + payoff(−Z)) / 2
Effect: halves the variance for the same N → more accurate price at same cost.
```

**95% Confidence Interval**:
```
CI = (price − 1.96·std_error,  price + 1.96·std_error)
```

**Finite-Difference Deltas**:
```
Δᵢ ≈ [V(Sᵢ + bump) − V(Sᵢ − bump)] / (2·bump)
where bump = 0.01 · Sᵢ   (1% of spot price)
```
The same random seed is used for both bumps → noise cancels in the difference (control variate technique).

### Utils (`pricing/utils.py`)

Two standalone functions:
- `cholesky_decompose(correlation)` → validates and computes L
- `generate_correlated_normals(n_assets, n_samples, chol, seed)` → draws Z = ε @ L.T

---

## Key Design Decisions

### 1. Pure module (no web dependencies)
`core/` imports only numpy, scipy, pandas. No FastAPI, no HTTP. This means:
- Works in a Jupyter notebook
- Can be imported by tests without starting a server
- Can be used by the backtester, API, and CLI equally

### 2. Abstract interface for data
`IDataProvider` decouples strategies from data sources. Consequence:
- Tests use `SimulatedDataProvider` (fast, no internet)
- Production uses `YahooDataProvider` (Phase 3)
- Both are interchangeable — zero strategy code changes needed

### 3. Dataclasses for all models
`@dataclass` gives free `__init__`, `__repr__`, `__eq__`. No boilerplate.
`field(default_factory=dict)` for mutable defaults (avoids shared-state bugs).

### 4. Static methods in BlackScholesModel
No instance state needed. Static methods make the API cleaner and emphasise
that Black-Scholes is a pure mathematical function, not an object with state.

### 5. Central finite difference for deltas
Using central difference `[V(S+ε) − V(S−ε)] / 2ε` instead of forward difference
`[V(S+ε) − V(S)] / ε` because:
- Same random seed → noise cancels → much lower variance
- Order of error is O(ε²) vs O(ε) for forward difference → more accurate

---

## Math Summary

### GBM (Geometric Brownian Motion)
```
S_t = S_{t-1} · exp( (μ − σ²/2)·dt  +  σ·√dt · Z_t )
```
- `μ − σ²/2`: Itô correction — without it E[S_t] ≠ S_0·e^(μt)
- `σ·√dt·Z_t`: random shock, scales with √time (not time)

### Black-Scholes Intuition
The option price = expected value of the payoff in the risk-neutral world,
discounted back to today. Key insight: replace actual drift μ with risk-free rate r
(risk-neutral measure). Then price = `e^(-rT) · E^Q[payoff]`.

### Cholesky Decomposition
Turns a correlation matrix C into L such that `L @ L.T = C`.
Then `Z = ε @ L.T` where `ε ~ N(0,I)` gives `Cov(Z) = C`. This is how you make
independent random numbers move together with the right correlations.

### Monte Carlo Standard Error
```
std_error = σ_payoff / √N
```
To halve the error, you need 4× more simulations. Antithetic variates
give you roughly the same improvement at no extra computation cost.

---

## Interview Q&A

**Q: Why do you use Monte Carlo instead of Black-Scholes for the basket option?**
A: Black-Scholes has a closed form only for single-asset European options. A basket option's payoff depends on the weighted sum of multiple correlated assets — no closed form exists because the sum of log-normals is not log-normal. Monte Carlo can price any payoff by simulation.

**Q: What is the Itô correction term and why is it there?**
A: In GBM we model log-prices as Brownian motion. When you convert from log-space to price-space using the exponential, Jensen's inequality introduces a bias. The `−σ²/2` term corrects for this so that `E[S_t] = S_0 · e^(μt)` as expected.

**Q: How does Cholesky give you correlated random numbers?**
A: If `ε ~ N(0,I)` (independent), then `Z = L·ε` has `Cov(Z) = L·I·L.T = L·L.T = C` (your correlation matrix). So you first draw independent normals, then linearly transform them with L.

**Q: What is the 95% confidence interval telling you in a Monte Carlo result?**
A: If you ran the simulation many times, 95% of the resulting price estimates would fall within that interval. It's a measure of precision — wider CI means you need more simulations.

**Q: Why does DeltaHedge use the same seed for up and down bumps?**
A: To cancel out Monte Carlo noise. If you use different random seeds, the difference `V_up − V_dn` picks up random fluctuations that have nothing to do with the delta. Same seed → same noise in both → noise cancels in the difference → much more accurate delta estimate.
