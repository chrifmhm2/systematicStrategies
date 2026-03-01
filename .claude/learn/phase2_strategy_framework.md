# Phase 2 — Strategy Framework: Everything You Need to Know

---

## What Phase 2 builds

Phase 2 adds a **plugin-based strategy layer** on top of the Phase 1 engine.
Any strategy can be added by creating a single file — the system discovers it automatically.

```
backend/core/strategies/
├── base.py          → StrategyConfig, PortfolioWeights, IStrategy (abstract contract)
├── registry.py      → StrategyRegistry (plugin catalog)
├── __init__.py      → triggers all @register decorators at import time
├── allocation/
│   ├── equal_weight.py   → EqualWeightStrategy
│   ├── min_variance.py   → MinVarianceStrategy
│   ├── max_sharpe.py     → MaxSharpeStrategy
│   └── risk_parity.py    → RiskParityStrategy
├── signal/
│   ├── momentum.py       → MomentumStrategy
│   └── mean_reversion.py → MeanReversionStrategy
└── hedging/
    ├── delta_hedge.py        → DeltaHedgeStrategy
    └── delta_gamma_hedge.py  → DeltaGammaHedgeStrategy
```

**Test coverage**: 29 tests (P2-15a through P2-15i), all passing. Total: 75/75.

---

## The 3 foundation types (`base.py`)

Everything in Phase 2 is built on these three types.

### StrategyConfig — the settings object

```python
@dataclass
class StrategyConfig:
    name: str
    description: str
    rebalancing_frequency: str = "weekly"    # "daily" | "weekly" | "monthly"
    transaction_cost_bps: float = 10.0       # 10 bps = 0.10% per trade
```

Passed to every strategy at construction. Carries common settings that the backtester (Phase 3) reads.

### PortfolioWeights — the output of every strategy

```python
@dataclass
class PortfolioWeights:
    weights: dict[str, float]   # symbol → weight, e.g. {"AAPL": 0.4, "MSFT": 0.6}
    cash_weight: float          # weight not deployed in assets
    timestamp: date             # the date these weights are valid for

    def total_weight(self) -> float:
        return sum(self.weights.values()) + self.cash_weight
```

**Invariant**: `total_weight()` should always be ≤ 1.0 (no leverage) and ≥ 0.0 (no short beyond hedge).

### IStrategy — the abstract contract

```python
class IStrategy(ABC):
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    @abstractmethod
    def compute_weights(
        self,
        current_date: date,
        price_history: pd.DataFrame,
        current_portfolio=None,
    ) -> PortfolioWeights:
        """The only method that matters. Called by the backtester at every date."""

    @property
    @abstractmethod
    def required_history_days(self) -> int:
        """Backtester won't call compute_weights until this many rows of data are available."""

    @classmethod
    @abstractmethod
    def get_param_schema(cls) -> dict:
        """JSON-schema-like dict. Used by frontend to render a form for the strategy's params."""
```

**No look-ahead bias guarantee**: `price_history` is always sliced to `prices.loc[:current_date]` by the backtester before being passed in. The strategy cannot see the future.

---

## StrategyRegistry — the plugin system (`registry.py`)

### How it works

```python
class StrategyRegistry:
    _strategies: dict[str, type[IStrategy]] = {}   # class-level shared dict

    @classmethod
    def register(cls, strategy_class):
        cls._strategies[strategy_class.__name__] = strategy_class
        return strategy_class   # unchanged — decorator is transparent
```

When you decorate a class:
```python
@StrategyRegistry.register
class MomentumStrategy(IStrategy):
    ...
```

Python internally executes:
```python
MomentumStrategy = StrategyRegistry.register(MomentumStrategy)
```

This fires at import time — the moment the file is loaded.

### Three public methods

| Method | What it does |
|---|---|
| `register(klass)` | Decorator — adds to `_strategies` |
| `list_strategies()` | Returns list of dicts: name, description, family, param_schema |
| `create(name, config)` | Builds `StrategyConfig`, instantiates and returns the strategy |

### How registration is triggered (`__init__.py`)

```python
# core/strategies/__init__.py
from core.strategies.base import IStrategy, PortfolioWeights, StrategyConfig
from core.strategies.registry import StrategyRegistry

import core.strategies.hedging      # noqa: F401  ← triggers @register for delta_hedge, delta_gamma_hedge
import core.strategies.allocation   # noqa: F401  ← triggers @register for equal_weight, min_variance, max_sharpe, risk_parity
import core.strategies.signal       # noqa: F401  ← triggers @register for momentum, mean_reversion
```

`noqa: F401` silences the linter warning "imported but unused" — the imports are for their side effect (decorator execution), not for a value.

### Create by name (how the API uses it)

```python
strat = StrategyRegistry.create("MomentumStrategy", {
    "lookback_period": 60,
    "top_k": 2,
    "long_only": True,
})
pw = strat.compute_weights(today, price_df)
```

The API receives a strategy name as a string and a config dict → registry instantiates the right class.

---

## The 4 Allocation Strategies

All three use `scipy.optimize.minimize(method="SLSQP")` — a constrained optimizer that works iteratively until convergence (typically 50–200 iterations).

### 1. EqualWeightStrategy

**Math**: `wᵢ = 1/N` for all N assets. No optimization.

**When to use**: When you distrust your return/covariance estimates. DeMiguel et al. (2009) showed 1/N beats most optimized portfolios out-of-sample because it has zero estimation error.

```python
weights = {sym: 1.0 / len(symbols) for sym in symbols}
cash_weight = 0.0
```

`required_history_days = 1` — needs just one price row.

---

### 2. MinVarianceStrategy

**Goal**: Minimize portfolio volatility.

**Optimization**:
```
min   ωᵀ Σ ω
 ω

s.t.  Σωᵢ = 1   (fully invested)
      ωᵢ ≥ 0   (long only)
```

Where `Σ` is the sample covariance of daily returns over `lookback_window` days.

**Key insight**: Sits at the leftmost point of the Markowitz efficient frontier — minimum risk regardless of return. Ignores expected returns entirely (returns forecasts are notoriously unreliable).

```python
cov = returns.tail(lookback).pct_change().dropna().cov().values

def objective(w):
    return float(w @ cov @ w)    # portfolio variance

result = minimize(objective, w0, method="SLSQP",
                  bounds=[(0,1)]*n,
                  constraints=[{"type": "eq", "fun": lambda w: w.sum()-1}])
```

`required_history_days = lookback_window + 1`.

---

### 3. MaxSharpeStrategy

**Goal**: Maximize return per unit of risk (the Sharpe ratio).

**Sharpe Ratio**:
```
SR = (μₚ − rf) / σₚ  =  (ωᵀμ − rf) / √(ωᵀΣω)
```

**Optimization** (minimize negative SR):
```
min   −(ωᵀμ − rf) / √(ωᵀΣω)
 ω

s.t.  Σωᵢ = 1,  ωᵢ ≥ 0
```

**Annualisation** (daily → annual):
```
μ_annual = μ_daily × 252
Σ_annual = Σ_daily × 252
```

**Key insight**: The MaxSharpe portfolio is the **tangency portfolio** — the point where the Capital Market Line (drawn from the risk-free rate) is tangent to the efficient frontier. It is the best risky portfolio for any risk-averse investor.

```
Expected Return
↑
|         * MaxSharpe (tangency)
|       /
|     /  ← Capital Market Line (from rf)
|   /
rf /
|  * MinVariance
|    ← Efficient Frontier
|___________________→ Volatility
```

---

### 4. RiskParityStrategy

**Goal**: Each asset contributes equally to total portfolio variance.

**Risk Contribution of asset i**:
```
RC_i = ωᵢ × (Σω)ᵢ / (ωᵀΣω)
```
Where `(Σω)ᵢ` is the i-th element of the vector `Σω` (the marginal risk contribution).

**Optimization** (minimize dispersion of risk contributions):
```
min   Σᵢ (RC_i − 1/N)²
 ω

s.t.  Σωᵢ = 1
      ωᵢ > 0   (lower bound: 1e-6 to avoid division by zero)
```

**Key insight**: A classic 60/40 portfolio has ~90% of its risk in equities — bonds are almost irrelevant for risk. Risk parity gives bonds more weight so each asset contributes equally. Used by Bridgewater's All Weather fund.

**Implementation detail**: `bounds = [(1e-6, 1.0)]` instead of `[(0.0, 1.0)]` because the RC formula divides by `port_var` and needs `wᵢ > 0`.

---

## The 2 Signal Strategies

Signal strategies make bets based on statistical patterns in prices.

### 5. MomentumStrategy

**Theory**: Jegadeesh & Titman (1993) — one of the most replicated anomalies in finance. Assets that outperformed over the past 3–12 months tend to continue outperforming.

**Signal**:
```
returnᵢ = Sᵢ,t / Sᵢ,t−L − 1    (trailing return over L days)
```

**Long-only mode**: Rank all N assets by trailing return, go long the top k:
```python
ranked = trailing_returns.sort_values(ascending=False)
top_k_symbols = list(ranked.index[:k])
weights = {sym: 1.0/k for sym in top_k_symbols}   # equal weight among top-k
```

**Long-short mode**: Long top-k at `+1/(2k)`, short bottom-k at `−1/(2k)`. Net exposure = 0 (market-neutral).

**Why does momentum work?** Investor underreaction to news, herding behavior, slow diffusion of information across investors.

`required_history_days = lookback_period + 1`.

---

### 6. MeanReversionStrategy

**Theory**: Prices tend to revert toward their rolling mean. A sharp drop is a buying opportunity (contrarian).

**Z-score** (the signal):
```
zᵢ = (Sᵢ,t − MAᵢ) / σᵢ
```
Where `MAᵢ` and `σᵢ` are the mean and std over `lookback_window` days.

**Interpretation**:
- `z = 0` → price is exactly at its rolling mean
- `z = −3` → price is 3 standard deviations below the mean → buy signal
- `z = +3` → price is 3 standard deviations above the mean → potential sell

**Signal rule**: buy when `z < −z_threshold` (default: threshold = 2.0)

**Weighting** (inversely proportional to |z|):
```python
inv_z = {sym: 1.0 / abs(z[sym]) for sym in triggered_symbols}
total = sum(inv_z.values())
weights = {sym: v / total for sym, v in inv_z.items()}
```
Why inverse? The more extreme the z-score (farther from mean), the more uncertain the reversion timing → take a smaller, more cautious position.

**If no signal**: `cash_weight = 1.0` (all cash, sit out the market).

`required_history_days = lookback_window + 1`.

---

## The 2 Hedging Strategies

Hedging strategies don't try to make money — they try to neutralize risk from an existing option position.

### 7. DeltaHedgeStrategy

**Context**: You hold a basket call option. To hedge it, replicate its payoff dynamically by holding the underlying stocks.

**Delta** = sensitivity of option price to a $1 move in the stock price:
```
Single asset: Δ = ∂C/∂S = N(d₁)   (Black-Scholes closed form)
Basket:       Δᵢ ≈ [V(Sᵢ+ε) − V(Sᵢ−ε)] / 2ε   (finite difference via Monte Carlo)
```

**Converting deltas to portfolio weights**:
```python
option_value = dot(deltas, spots)   # V ≈ Σ(Δᵢ · Sᵢ)
weights[i] = (deltas[i] * spots[i]) / option_value
cash_weight = max(0, 1 - sum(weights))
```

**Why does this hedge?** Black-Scholes proves that if you continuously rebalance to hold `Δᵢ` shares of each stock, the randomness in your stock position exactly cancels the randomness in your option position → riskless portfolio earning the risk-free rate.

**What this strategy does step by step**:
1. Get current prices from `price_history.iloc[-1]`
2. Run `MonteCarloPricer.compute_deltas()` (Phase 1) with 1% bumps
3. Convert Δᵢ → weight `wᵢ = Δᵢ·Sᵢ/V`
4. Remainder goes to cash

`required_history_days = 1`.

---

### 8. DeltaGammaHedgeStrategy

**Theory** (not yet implemented): Delta hedging leaves **gamma risk** — the error from discrete (not continuous) rebalancing.

Gamma = rate of change of delta:
```
Γ = ∂²C/∂S² = ∂Δ/∂S
```

To neutralize gamma, add a second option with gamma `Γ_h`:
```
ω_hedge = −Γ_portfolio / Γ_h
```
Then re-delta-hedge the combined portfolio.

**Current state**: This is a **stub**. It composes a `DeltaHedgeStrategy` internally and fully delegates:
```python
class DeltaGammaHedgeStrategy(IStrategy):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self._delta_hedge = DeltaHedgeStrategy(config, **kwargs)   # composition

    def compute_weights(self, ...):
        return self._delta_hedge.compute_weights(...)   # delegation
```

This is the **Composition Pattern**: `DeltaGammaHedge` HAS a `DeltaHedge`. It IS a `IStrategy` (inheritance).

---

## Key Design Decisions

### 1. Plugin architecture — zero central list
Adding a new strategy = create a file + add `@StrategyRegistry.register`. The registry, API, and frontend automatically discover it. No file to maintain, no list to update.

### 2. `required_history_days` — backtester safety
The backtester reads this property before calling `compute_weights()`. If there aren't enough rows, the backtester waits until there are. No strategy can accidentally be called with insufficient data.

### 3. `get_param_schema()` — frontend form generation
The React frontend calls `GET /api/strategies` → gets the registry list → reads `param_schema` for each strategy → renders a form with the right fields, types, and defaults. Strategies are self-describing.

### 4. `DESCRIPTION` and `FAMILY` as class attributes
`StrategyRegistry.list_strategies()` reads these directly from the class (not from an instance) using `getattr(klass, "DESCRIPTION", "")`. The registry never instantiates a strategy just to discover metadata.

### 5. SLSQP with equal-weight starting point
All optimizers start from `w0 = ones(n)/n` (equal weight). This is a reasonable starting point that satisfies the equality constraint, making SLSQP converge faster and more reliably than a random start.

### 6. Numerical stability in RiskParity
`bounds = [(1e-6, 1.0)]` instead of `(0.0, 1.0)` because the RC formula is:
```
RC_i = ωᵢ · (Σω)ᵢ / (ωᵀΣω)
```
If any `ωᵢ = 0`, the numerator is 0/something = 0, but the optimizer's gradient becomes ill-defined. The small lower bound prevents this.

---

## Data Flow: How Phase 1 and Phase 2 Connect

```
Phase 1                          Phase 2
───────                          ───────
MonteCarloPricer                 DeltaHedgeStrategy
  .compute_deltas()   ◄────────    calls it inside compute_weights()

BlackScholesModel                (used in Phase 4 for Greeks surface)

SimulatedDataProvider            (used in tests and demos as price_history)
  .get_prices()       ─────────►  passed as price_history to compute_weights()
```

`DeltaHedgeStrategy` is the only strategy that depends on Phase 1. The 7 other strategies only need a pandas DataFrame of prices.

---

## Test Structure (29 tests)

| Test class | Strategy tested | Key assertion |
|---|---|---|
| `TestEqualWeight` (3) | EqualWeightStrategy | `wᵢ = 1/N`, sum = 1, cash = 0 |
| `TestMinVariance` (3) | MinVarianceStrategy | weights ≥ 0, sum = 1, cash = 0 |
| `TestMaxSharpe` (2) | MaxSharpeStrategy | weights ≥ 0, sum = 1 |
| `TestRiskParity` (2) | RiskParityStrategy | risk contributions ≈ 1/N (within 5%), sum = 1 |
| `TestMomentum` (3) | MomentumStrategy | exactly top_k nonzero, equal weights, sum = 1 |
| `TestMeanReversion` (3) | MeanReversionStrategy | only triggered symbols nonzero, all-cash if no signal |
| `TestNoLookAheadBias` (6) | All 6 non-hedging strats | weights identical whether or not future rows exist |
| `TestRegistry` (5) | StrategyRegistry | ≥6 strategies, all keys present, create() works |
| `TestDeltaHedge` (2) | DeltaHedgeStrategy | total_weight ≤ 1, all weights ≥ 0 |

---

## Interview Q&A

**Q: What is the plugin pattern and why did you use it?**
A: A plugin pattern lets components self-register without a central coordinator knowing about them. In our case, each strategy file uses `@StrategyRegistry.register` which fires at import time and adds the class to a shared dict. The registry, API, and frontend automatically discover new strategies the moment a file is added. No central list to update, no risk of forgetting.

**Q: What is the difference between MinVariance and MaxSharpe?**
A: MinVariance only minimizes risk (`min ωᵀΣω`) and completely ignores expected returns. MaxSharpe maximizes the risk-return tradeoff (`max (ωᵀμ−rf)/σₚ`) and uses both mean returns and covariance. MinVariance is more robust because return estimates are very noisy; MaxSharpe gives better expected performance but is more sensitive to estimation errors.

**Q: What is risk parity and why is it better than 60/40?**
A: In a 60% equity / 40% bond portfolio, equities are ~3× more volatile than bonds. So despite being 40% of capital, bonds contribute only ~10% of total portfolio risk — the portfolio is actually 90% driven by equity risk. Risk parity fixes this by weighting inversely to volatility: higher-vol assets get less capital so every asset contributes equally to risk. Bridgewater's All Weather fund uses this principle.

**Q: What does `required_history_days` do and why is it important?**
A: It tells the backtester the minimum number of historical price rows needed before the strategy can produce meaningful weights. For example, MinVariance with `lookback_window=60` needs at least 61 rows to compute a covariance matrix. Without this guard, the optimizer would receive near-empty data and produce garbage weights (or crash).

**Q: What is no look-ahead bias and how do you prevent it?**
A: Look-ahead bias is when a strategy uses future data that wouldn't have been available on the trading date. For example, if you compute weights for Jan 15 using prices from Jan 16–20, your backtest is cheating — those prices didn't exist yet on Jan 15. We prevent it structurally: the backtester always passes `prices.loc[:current_date]` — the dataframe is sliced to include only data up to and including the current date before being passed to `compute_weights()`. The strategy cannot access future rows even if it wanted to.

**Q: Why does MeanReversion weight inversely to |z|?**
A: The z-score measures how far from the mean. A very high |z| (e.g. z = −5) means the price has moved extremely far — the signal is strong but the timing of reversion is more uncertain. A moderate |z| (e.g. z = −2.5) is near the threshold — the signal is weaker but more reliable as a short-term mean reversion. So `w ∝ 1/|z|` gives more weight to the "safer" signals.

**Q: What is the composition pattern and where did you use it?**
A: Composition means an object owns another object and delegates work to it, rather than inheriting. `DeltaGammaHedgeStrategy` holds a `DeltaHedgeStrategy` instance (`self._delta_hedge`) and calls `self._delta_hedge.compute_weights()`. Both inherit from `IStrategy` (they both ARE a strategy), but DeltaGamma also HAS a DeltaHedge (composition). This is cleaner than multiple inheritance and makes the stub easy to replace later.
