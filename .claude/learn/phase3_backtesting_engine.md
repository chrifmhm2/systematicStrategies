# Phase 3 — Backtesting Engine: Everything You Need to Know

---

## What Phase 3 builds

Phase 3 adds the **simulation engine** that drives any `IStrategy` through historical prices
and produces a full performance record: NAV series, benchmark, weights history, and trades log.

```
backend/core/backtester/
├── __init__.py      → exports all public classes
├── engine.py        → BacktestConfig (dataclass), BacktestEngine
├── costs.py         → TransactionCostModel
└── rebalancing.py   → RebalancingOracle (ABC), PeriodicRebalancing, ThresholdRebalancing

backend/core/data/
└── yahoo.py         → YahooDataProvider (IDataProvider implementation)

backend/core/models/
└── results.py       → BacktestResult (updated), PricingResult (unchanged)

backend/tests/
└── test_backtester.py   → 22 tests labelled [P3-15a] … [P3-15g]
```

---

## Dependency graph

```
                 ┌──────────────────────────────────────────────┐
                 │             BacktestEngine                    │
                 │  (orchestrates the entire simulation loop)    │
                 └───────────┬───────────────────────┬──────────┘
                             │ uses                  │ uses
              ┌──────────────▼────────┐   ┌──────────▼──────────────┐
              │    BacktestConfig     │   │    TransactionCostModel  │
              │  (dataclass: params)  │   │  compute_cost(value)→$  │
              └───────────────────────┘   └─────────────────────────┘
                             │
                             │ creates one of
              ┌──────────────▼─────────────────────────────┐
              │          RebalancingOracle  (ABC)            │
              │  should_rebalance(date, portfolio) → bool    │
              └──────┬───────────────────────────┬──────────┘
                     │ implements                │ implements
        ┌────────────▼────────┐     ┌────────────▼──────────────┐
        │  PeriodicRebalancing│     │  ThresholdRebalancing     │
        │  daily/weekly/      │     │  fires when any weight    │
        │  monthly calendar   │     │  drifts > threshold       │
        └─────────────────────┘     └───────────────────────────┘

                 BacktestEngine also uses:
              ┌────────────────────┐    ┌───────────────────┐
              │   IDataProvider    │    │    IStrategy       │
              │ get_prices()       │    │ compute_weights()  │
              │ get_risk_free()    │    │ required_history   │
              └──────┬─────────────┘    └───────────────────┘
                     │ implemented by
          ┌──────────▼─────────────┐   ┌──────────────────────┐
          │  SimulatedDataProvider │   │  YahooDataProvider   │
          │  (tests / offline)     │   │  (yfinance / live)   │
          └────────────────────────┘   └──────────────────────┘

              BacktestEngine produces:
              ┌─────────────────────────────────────────────────┐
              │              BacktestResult                     │
              │  portfolio_values  : pd.Series (DatetimeIndex)  │
              │  benchmark_values  : pd.Series                  │
              │  weights_history   : pd.DataFrame               │
              │  trades_log        : list[dict]                 │
              │  risk_metrics      : dict  (Phase 4 fills this) │
              │  config            : BacktestConfig | None      │
              │  strategy_name     : str                        │
              │  computation_time_ms: float                     │
              └─────────────────────────────────────────────────┘
```

### Circular import resolution

```
results.py  needs  BacktestConfig  (for type annotation)
engine.py   needs  BacktestResult  (for return type)
→ solved with TYPE_CHECKING guard in results.py
```

---

## The 8 classes — what each one does

### 1. `BacktestConfig` — dataclass

The single "bag of parameters" passed to `BacktestEngine`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `initial_value` | float | 100 000 | Starting NAV in $ |
| `start_date` | date \| None | None | First backtest date |
| `end_date` | date \| None | None | Last backtest date |
| `symbols` | list[str] | [] | Universe of tickers |
| `rebalancing_frequency` | str | "weekly" | "daily", "weekly", "monthly", "threshold" |
| `transaction_cost_bps` | float | 10.0 | Broker commission (bps) |
| `slippage_bps` | float | 5.0 | Bid-ask / market impact (bps) |
| `threshold` | float | 0.05 | Drift limit for ThresholdRebalancing |

---

### 2. `TransactionCostModel`

**One formula:**
```
cost = |trade_value| × (commission_bps + slippage_bps) / 10 000
cost = max(cost, min_commission)
```

Example: 15 bps of $10 000 trade = $15.00.
Example: 15 bps of $1 trade = $0.0015 → floored to $1.00 (min_commission).

```python
model = TransactionCostModel(commission_bps=10, slippage_bps=5, min_commission=1.0)
model.compute_cost(10_000)  # → 15.0
model.compute_cost(1.0)     # → 1.0  (floored)
```

---

### 3. `RebalancingOracle` — ABC

Defines the contract: one abstract method.
```python
@abstractmethod
def should_rebalance(self, current_date: date, portfolio: object) -> bool: ...
```
Engine calls this every day in the loop. Returns True → trigger rebalancing.

---

### 4. `PeriodicRebalancing`

**State:** `_last_rebalance_date: date | None`

| Frequency | Logic |
|---|---|
| `"daily"` | Always returns `True` |
| `"weekly"` | Returns `True` if `current_date.weekday() == 0` (Monday) |
| `"monthly"` | Returns `True` if `current_date.month != _last_rebalance_date.month` |

Updates `_last_rebalance_date` when it fires. Any other string → raises `ValueError`.

---

### 5. `ThresholdRebalancing`

**State:** `_last_target_weights: dict[str, float]`

Algorithm per day:
```
1. If _last_target_weights is empty → return True  (first call, never traded yet)
2. If portfolio is not a Portfolio or has no positions → return True
3. total = sum(qty × price for each position)
4. For each symbol in target_weights:
     current_w = (qty × price) / total
     if |current_w - target_w| > threshold → return True
5. return False
```

After each rebalancing, engine calls `update_target_weights(pw.weights)` to store the new targets.

---

### 6. `BacktestEngine`

Main orchestrator. Contains two public methods:

#### `__init__(config: BacktestConfig)`
Stores config. No computation.

#### `run(strategy, data_provider) → BacktestResult`

The full loop:

```
Step 1 — Fetch prices
  prices = data_provider.get_prices(symbols, start_date, end_date)
  prices = prices.dropna(how="all")    # remove empty rows

Step 2 — Build oracle and cost model
  oracle = _build_oracle(cfg)          # PeriodicRebalancing or ThresholdRebalancing
  cost_model = TransactionCostModel(commission_bps, slippage_bps)

Step 3 — Initialise portfolio
  portfolio = Portfolio(positions={}, cash=initial_value)

Step 4 — Main loop for t in prices.index:
  a) Update position prices:
       for sym, pos in portfolio.positions:
           pos.price = current_prices[sym]

  b) Cash accrues interest:
       portfolio.cash *= (1 + risk_free_rate × 1/252)

  c) Mark-to-market NAV:
       V_t = portfolio.total_value(valid_prices)
       portfolio_values[t] = V_t

  d) Guard: if len(prices[:t]) < strategy.required_history_days: continue

  e) Oracle check: if not oracle.should_rebalance(t_date, portfolio): continue

  f) Rebalance:
     pw = strategy.compute_weights(t_date, prices.loc[:t], portfolio)
     for each symbol:
         new_qty = pw.weights[sym] × V_t / price
         delta_qty = new_qty - old_qty
         cost = cost_model.compute_cost(delta_qty × price)
     equity_deployed = Σ (new_qty × price)
     new_cash = V_t - equity_deployed - total_costs

  g) Assert self-financing:
     |equity_deployed + new_cash - (V_t - total_costs)| < 1e-6

  h) Update portfolio positions and cash
  i) If ThresholdRebalancing: oracle.update_target_weights(pw.weights)

Step 5 — Build pd.Series for portfolio_values
Step 6 — Compute equal-weight buy-and-hold benchmark
Step 7 — Build pd.DataFrame for weights_history
Step 8 — Return BacktestResult
```

---

### 7. `BacktestResult` — dataclass

The single return object from `BacktestEngine.run()`.

```python
@dataclass
class BacktestResult:
    portfolio_values:   pd.Series      # DatetimeIndex → NAV each day
    benchmark_values:   pd.Series      # DatetimeIndex → benchmark NAV
    weights_history:    pd.DataFrame   # rebalancing dates × symbols
    trades_log:         list[dict]     # [{date, symbol, quantity, price, cost}, ...]
    risk_metrics:       dict           # {} until Phase 4 fills it
    config:             BacktestConfig | None
    strategy_name:      str
    computation_time_ms: float
```

Each trade dict has exactly 5 keys: `date`, `symbol`, `quantity`, `price`, `cost`.

---

### 8. `YahooDataProvider`

Implements `IDataProvider` using `yfinance`.

```python
class YahooDataProvider(IDataProvider):
    def __init__(self, cache: bool = True):
        self._cache: dict[tuple, pd.DataFrame] = {}

    def get_prices(self, symbols, start_date, end_date) -> pd.DataFrame:
        # 1. Check cache key = (tuple(sorted(symbols)), start_date, end_date)
        # 2. yf.download(symbols, auto_adjust=True, progress=False)
        # 3. Handle multi-level columns: raw["Close"] for multiple tickers
        # 4. Ensure all symbols present (NaN if missing)
        # 5. .ffill()   ← fills weekends, holidays
        # 6. Store in cache
        # 7. Return df

    def get_risk_free_rate(self, as_of_date) -> float:
        # Fetch ^IRX (13-week T-bill), divide by 100
        # Fall back to 0.05 on any Exception
```

---

## Math & Finance — the concepts you must know

### 1. Self-financing constraint

The most important invariant in backtesting:

> **No money appears from nowhere or disappears into thin air.**
> When you rebalance, the only "leak" is transaction costs.

Formula:
```
V_after = V_before − TC
```
Where:
- `V_before = V_t` = NAV right before rebalancing (equity + cash, marked to market)
- `TC` = total transaction costs for this rebalancing step
- `V_after = equity_deployed + new_cash`

In the engine:
```python
new_cash = V_t - equity_deployed - total_costs
# Then we assert:
assert |equity_deployed + new_cash - (V_t - total_costs)| < 1e-6
```
This assertion fires immediately if there's a bug — no silent corruption.

---

### 2. Mark-to-market NAV

At every time step `t`, the portfolio value is:
```
V_t = Σᵢ (qᵢ × Pᵢ(t)) + cash(t)
```
- `qᵢ` = quantity of asset `i` (fixed until next rebalancing)
- `Pᵢ(t)` = current market price of asset `i`
- `cash(t)` = cash balance (grows with risk-free interest)

This is **marked-to-market**: we price the portfolio at today's prices every day.

---

### 3. Transaction costs — basis points

**1 basis point (1 bps) = 0.01% = 0.0001**

```
cost = |trade_value| × (commission_bps + slippage_bps) / 10 000
```

Two components:
- **Commission**: fee charged by broker (e.g. 10 bps = 0.10% of trade value)
- **Slippage**: you never get the exact quoted price — bid-ask spread and market impact push price against you (e.g. 5 bps)

Combined: 15 bps = 0.15%.
On $1 M trade: $1 500 in costs.
On $100 trade: $0.15 → floored to $1.00 (min commission).

---

### 4. Cash interest accrual

Between rebalancings, uninvested cash earns the risk-free rate:
```
cash(t) = cash(t-1) × (1 + r_f × Δt)
```
- `r_f` = annualized risk-free rate (e.g. 0.05 for 5%)
- `Δt = 1/252` = one trading day in years (252 trading days/year convention)

This is a discrete approximation of continuous compounding `e^(r_f × Δt)`.

---

### 5. Equal-weight buy-and-hold benchmark

The simplest possible benchmark — buy equal amounts of each asset on day 1, never touch it:
```
qᵢ = (initial_value / N) / Sᵢ(t₀)
benchmark(t) = Σᵢ qᵢ × Sᵢ(t)
```
- `N` = number of assets
- `Sᵢ(t₀)` = price of asset `i` on first day
- No rebalancing, no transaction costs

Used to benchmark whether the strategy adds value over the simplest possible passive portfolio.

---

### 6. Weight drift and threshold rebalancing

After rebalancing to target weights `w*`, asset prices move and weights drift:
```
w_i(t) = (q_i × P_i(t)) / V(t)
drift_i = |w_i(t) - w*_i|
```
If `max_i(drift_i) > threshold` → rebalance.

**Trade-off:**
- Low threshold → many rebalances → high costs → lower net return
- High threshold → few rebalances → cheaper but portfolio strays from target

In our tests we verify: tight threshold (0.1%) → more trades than loose (99%).

---

### 7. Required history guard

Strategies like Momentum and MinVariance need a **lookback window** of past prices before they can compute sensible weights (e.g. 60 days for covariance estimation).

The engine enforces this:
```python
if len(prices.loc[:t]) < strategy.required_history_days:
    continue   # skip rebalancing, just record NAV
```
This prevents the strategy from operating on too-short time series where estimates are unreliable.

---

### 8. No look-ahead bias

The most dangerous bug in backtesting — accidentally using future information.

The engine guarantees this by construction:
```python
history = prices.loc[:t]                   # includes ONLY dates ≤ t
pw = strategy.compute_weights(t_date, history, portfolio)
```
`prices.loc[:t]` with a DatetimeIndex is an **inclusive** slice up to `t`, never beyond.

Test [P3-15c] verifies this with a spy:
```python
def spy_compute(current_date, price_history, current_portfolio=None):
    for ts in price_history.index:
        assert ts.date() <= current_date   # no future dates in history
    return original_compute(current_date, price_history, current_portfolio)
```

---

## Python / Coding patterns specific to Phase 3

### 1. `@dataclass` with `field(default_factory=...)`

Mutable defaults (list, dict, pd.Series) cannot be set as plain defaults in dataclasses.
You must use `field(default_factory=...)`:

```python
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class BacktestResult:
    portfolio_values: pd.Series = field(default_factory=pd.Series)
    trades_log: list[dict] = field(default_factory=list)
    risk_metrics: dict    = field(default_factory=dict)
```
Why: if you wrote `trades_log: list = []`, all instances would share the SAME list object.

---

### 2. `TYPE_CHECKING` guard — avoiding circular imports

```python
# results.py
from __future__ import annotations  # ← makes all annotations strings (lazy eval)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.backtester.engine import BacktestConfig  # only imported during mypy/pylance

@dataclass
class BacktestResult:
    config: BacktestConfig | None = None   # annotation only, never evaluated at runtime
```

Execution flow:
- At **runtime**: `TYPE_CHECKING = False` → import is skipped → no circular import
- At **type-check time**: `TYPE_CHECKING = True` → import runs → type checker sees types

Key: `from __future__ import annotations` makes all annotations lazy (PEP 563), so `BacktestConfig` is just a string `"BacktestConfig"` at runtime, never evaluated.

---

### 3. Local import to avoid circular dependency

```python
# ThresholdRebalancing.should_rebalance()
def should_rebalance(self, current_date: date, portfolio: object) -> bool:
    from core.models.portfolio import Portfolio  # local import inside method
    if not isinstance(portfolio, Portfolio):
        return True
```
Importing inside a function means the import only runs when the function is called,
not when the module is loaded — avoids the circular dependency entirely.

---

### 4. Abstract Base Class pattern (same as Phase 2, applied here)

```python
from abc import ABC, abstractmethod

class RebalancingOracle(ABC):
    @abstractmethod
    def should_rebalance(self, current_date: date, portfolio: object) -> bool: ...
```
Cannot instantiate `RebalancingOracle` directly.
Subclasses that forget to implement `should_rebalance` raise `TypeError` at instantiation.

---

### 5. `dict` accumulator → `pd.Series` / `pd.DataFrame`

Pattern: build results in a plain dict (fast), convert to pandas at the end (clean API):
```python
portfolio_values: dict = {}   # fast accumulation
# ... loop:
portfolio_values[t] = V_t

# After loop:
pv_series = pd.Series(portfolio_values)
pv_series.index = pd.DatetimeIndex(pv_series.index)
```
Same for weights history:
```python
weights_records: dict = {}    # {timestamp: {sym: weight, ...}}
weights_df = pd.DataFrame(weights_records).T   # rows=timestamps, cols=symbols
```

---

### 6. `time.perf_counter()` for timing

```python
import time
start_time = time.perf_counter()
# ... do work ...
elapsed_ms = (time.perf_counter() - start_time) * 1000
```
`perf_counter()` is the highest-resolution timer available. Returns seconds (float). We multiply by 1000 for milliseconds.

---

### 7. `prices.loc[:t]` — inclusive DatetimeIndex slice

In pandas, `.loc[a:b]` on an index is **inclusive** on both ends:
```python
history = prices.loc[:t]     # all rows with index ≤ t
```
This is different from list/array slicing where the upper bound is exclusive.
Critical for no look-ahead bias: only rows up to and including `t` are passed.

---

### 8. yfinance multi-level column handling

When downloading multiple tickers, yfinance returns a `pd.MultiIndex` of columns:
```
             Close           High          ...
             AAPL  MSFT      AAPL  MSFT
2023-01-02   130   260       131   261
```
We extract only the "Close" level:
```python
if isinstance(raw.columns, pd.MultiIndex):
    df = raw["Close"]     # selects one level → flat columns: AAPL, MSFT
else:
    df = raw[["Close"]]   # single ticker: flat ["Close"] → rename to [symbol]
    df.columns = symbols
```

---

### 9. Spy pattern in tests (monkey-patching)

Testing a side effect (what data a function received) without changing the production code:
```python
original_compute = real_strategy.compute_weights

def spy_compute(current_date, price_history, current_portfolio=None):
    received_histories.append((current_date, price_history.index.copy()))
    return original_compute(current_date, price_history, current_portfolio)  # delegate

real_strategy.compute_weights = spy_compute   # replace with spy
```
The spy records what it saw, then calls the real implementation. The test can then assert on the recorded data. No mocking library needed.

---

## The 22 tests — what they do and how to present them

| Label | Class | What it tests |
|---|---|---|
| [P3-15a] | `TestSelfFinancing` | Self-financing invariant: engine runs without assertion error |
| [P3-15a] | `TestSelfFinancing` | Portfolio with costs < zero-cost portfolio (same seed) |
| [P3-15b] | `TestTransactionCosts` | Zero-cost > nonzero-cost (same prices, same strategy) |
| [P3-15b] | `TestTransactionCosts` | Cost model: 15 bps of $10k = $15.00 |
| [P3-15b] | `TestTransactionCosts` | Cost model: min_commission floor when trade too small |
| [P3-15c] | `TestNoLookAheadBias` | Spy on compute_weights: all history dates ≤ current_date |
| [P3-15d] | `TestDateRange` | portfolio_values index within [start_date, end_date] |
| [P3-15d] | `TestDateRange` | portfolio_values is a pd.Series |
| [P3-15d] | `TestDateRange` | benchmark_values is pd.Series, same length as portfolio_values |
| [P3-15e] | `TestTradesLog` | Each trade has {date, symbol, quantity, price, cost} keys |
| [P3-15e] | `TestTradesLog` | Trade cost ≥ 0 |
| [P3-15e] | `TestTradesLog` | Trade price > 0 |
| [P3-15f] | `TestPeriodicRebalancing` | Weekly fires on Monday |
| [P3-15f] | `TestPeriodicRebalancing` | Weekly does not fire Tue–Fri |
| [P3-15f] | `TestPeriodicRebalancing` | Daily fires every day |
| [P3-15f] | `TestPeriodicRebalancing` | Monthly fires once per month |
| [P3-15f] | `TestPeriodicRebalancing` | Invalid frequency raises ValueError |
| [P3-15f] | `TestPeriodicRebalancing` | Integrated backtest: all rebalancing dates are Mondays |
| [P3-15g] | `TestThresholdRebalancing` | No rebalance when within threshold |
| [P3-15g] | `TestThresholdRebalancing` | Rebalances when drift > threshold |
| [P3-15g] | `TestThresholdRebalancing` | First call always rebalances (no target set) |
| [P3-15g] | `TestThresholdRebalancing` | Integrated: tight threshold → more trades than loose threshold |

### How to present them in an interview

> "Phase 3 has 22 tests organized in 7 groups.
>
> The most important are the **3 invariant tests**:
>
> 1. **Self-financing** [P3-15a]: the engine itself asserts `|V_after - (V_before - TC)| < 1e-6` at every rebalancing step. If it runs without crashing, the invariant holds.
>
> 2. **No look-ahead bias** [P3-15c]: we replace `strategy.compute_weights` with a spy wrapper that records the price history received on each call, then verify every date in that history is ≤ the current backtest date. This is a direct audit trail.
>
> 3. **Transaction costs reduce returns** [P3-15b]: we run the same strategy with the same simulated data twice — once with 15 bps costs, once with 0 bps. The zero-cost run must end higher. This validates the economic intuition that costs always hurt.
>
> The **rebalancing oracle tests** are unit tests of the scheduling logic: weekly fires only on Mondays, monthly fires once per calendar month, threshold fires when drift exceeds the configured percentage. These are important because a bug in the oracle (e.g. rebalancing every day instead of every week) would inflate turnover and destroy the cost model's accuracy."

---

## Design decisions and why

### Why an ABC for RebalancingOracle?

Because we have two fundamentally different rebalancing philosophies — calendar-based and drift-based — and more can be added (e.g. volatility-triggered, news-triggered). The ABC ensures all implementations satisfy the same contract (`should_rebalance` returns bool) without duplicating any code.

### Why assert self-financing instead of silently fixing it?

A silent fix would mask bugs. If the constraint is violated, we want to know immediately — an `AssertionError` with context is better than silently corrupted NAV. Bugs in backtesting are often invisible because the code runs without error but produces wrong numbers.

### Why `TYPE_CHECKING` guard instead of restructuring the code?

The alternative would be to move `BacktestConfig` into `models/` to break the cycle. But `BacktestConfig` is tightly coupled to `BacktestEngine` — it describes engine parameters, not domain model objects. The TYPE_CHECKING guard keeps the logical structure correct while satisfying Python's runtime import system.

### Why a dict accumulator pattern?

Building a `pd.Series` by appending in a loop is O(n²) due to repeated copying. Building a plain dict (O(1) per insert) and converting once at the end is O(n). For long backtests (20+ years of daily data = ~5000 rows) this is a real difference.

### Why yfinance's `auto_adjust=True`?

Adjusted prices account for dividends and stock splits. Without adjustment, a stock that paid a large dividend would show a sharp price drop that looks like a crash but is actually just the ex-dividend date. Strategies built on raw prices would behave completely differently than on adjusted prices.

---

## Test structure (from test_backtester.py)

```python
# Shared fixtures
SYMBOLS = ["AAPL", "MSFT"]
START = date(2023, 1, 2)
END   = date(2023, 6, 30)

def _make_provider(seed):  # SimulatedDataProvider
def _make_strategy(name):  # EqualWeightStrategy
def _make_config(**overrides):  # BacktestConfig with defaults
```

Each test class is labeled with the specification label it tests.
Same seed → same simulated prices → deterministic comparison between runs.

---

## Quick-fire interview Q&A

**Q: What is self-financing and why does it matter?**
A: Self-financing means portfolio value after rebalancing = value before − transaction costs. No money is created or destroyed. Without this constraint, a backtest can artificially inflate returns by implicitly adding capital. We assert it explicitly with a tolerance of 1e-6.

**Q: How do you prevent look-ahead bias in the backtest?**
A: The engine only passes `prices.loc[:t]` to `compute_weights()` at time step `t`. This is a strict slice that includes only dates up to and including `t`. We have a dedicated test that spies on every call and verifies no future dates appear in the history.

**Q: What's the difference between PeriodicRebalancing and ThresholdRebalancing?**
A: Periodic fires on a calendar schedule (every Monday, first of month, etc.) regardless of what the market did. Threshold fires only when a position has drifted enough from its target weight. Threshold rebalancing is more cost-efficient in calm markets (few rebalances) but can trigger frequently during volatile periods.

**Q: What are basis points and why use them for costs?**
A: 1 basis point = 0.01%. Using basis points makes costs scale-independent — 10 bps costs the same fraction of trade value whether the trade is $100 or $1,000,000. Typical retail broker: 10–25 bps. Institutional: 1–5 bps.

**Q: Why does the engine accrue risk-free interest on cash?**
A: Uninvested cash is not idle — it's typically held in T-bills or a money market fund. Not accruing interest would understate the portfolio's return and bias the comparison against strategies that hold more cash.

**Q: What is the benchmark and why equal-weight buy-and-hold?**
A: The benchmark is the simplest possible passive portfolio: buy equal amounts of each asset on day 1 and never trade again. It has zero turnover, zero transaction costs, and requires no forecasting. Any strategy that underperforms this benchmark is adding no value while consuming costs.
