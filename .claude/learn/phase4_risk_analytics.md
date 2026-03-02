# Phase 4 — Risk Analytics: Complete Reference

> Everything you need to know about Phase 4: architecture, math, code patterns, API, tests, demos.

---

## 1. What Was Built

Phase 4 adds a `core/risk/` package with three static-only classes, wires them into `BacktestEngine`, and adds 10 tests.

```
backend/core/risk/
├── __init__.py       — exports all 3 classes
├── metrics.py        — PerformanceMetrics (13 static methods)
├── var.py            — VaRCalculator (4 static methods)
└── greeks.py         — GreeksCalculator (2 static methods)

backend/core/backtester/engine.py   — wired: compute_all called at end of run()
backend/tests/test_risk_metrics.py  — 10 tests (P4-25a through P4-25h)
backend/demos/phase3_demo.py        — demo for phase 3 (created alongside phase 4)
backend/demos/phase4_demo.py        — demo for phase 4
```

**Test count after Phase 4:** 107/107 (46 P1 + 29 P2 + 22 P3 + 10 P4)

---

## 2. Architecture Overview

```
BacktestEngine.run()
    └─ at end of loop, calls:
        PerformanceMetrics.compute_all(
            values=portfolio_nav,
            benchmark=benchmark_nav,
            weights_history=weights_df,
            risk_free_rate=rfr
        )
        │
        ├─ calls all PerformanceMetrics.* methods
        └─ local-imports VaRCalculator → historical_var / historical_cvar

BacktestResult.risk_metrics  ← dict of all computed metrics, never empty


FastAPI (Phase 5)
    POST /backtest  ──────────────────→ returns BacktestResult.risk_metrics directly
    POST /risk/analyze ───────────────→ calls compute_all() on provided NAV series
    POST /pricing/option ─────────────→ calls GreeksCalculator (for Greeks surface)
```

---

## 3. Full API Reference

### PerformanceMetrics (`core/risk/metrics.py`)

All methods take a `pd.Series` of **NAV values** (not returns). All return `float`. `NaN` on edge cases, never crash.

```python
from core.risk.metrics import PerformanceMetrics

PerformanceMetrics.total_return(values)                    # → float
PerformanceMetrics.annualized_return(values, trading_days=252)
PerformanceMetrics.annualized_volatility(values, trading_days=252)
PerformanceMetrics.sharpe_ratio(values, risk_free_rate=0.05)   # NaN if vol==0
PerformanceMetrics.sortino_ratio(values, risk_free_rate=0.05)
PerformanceMetrics.max_drawdown(values)                    # negative float
PerformanceMetrics.calmar_ratio(values)                    # NaN if MDD==0
PerformanceMetrics.win_rate(values)                        # ∈ [0,1]
PerformanceMetrics.profit_factor(values)
PerformanceMetrics.tracking_error(portfolio, benchmark)    # needs benchmark
PerformanceMetrics.information_ratio(portfolio, benchmark) # needs benchmark
PerformanceMetrics.turnover(weights_history)               # pd.DataFrame input
PerformanceMetrics.compute_all(values, benchmark=None, weights_history=None, risk_free_rate=0.05)
# → dict with all keys below
```

**`compute_all` output keys:**

| Key | Type | Notes |
|-----|------|-------|
| `total_return` | float | e.g. 0.35 = 35% |
| `annualized_return` | float | CAGR |
| `annualized_volatility` | float | annual std |
| `sharpe_ratio` | float | NaN if vol=0 |
| `sortino_ratio` | float | NaN if no negative days |
| `max_drawdown` | float | negative, e.g. -0.15 |
| `calmar_ratio` | float | NaN if MDD=0 |
| `win_rate` | float | ∈ [0, 1] |
| `profit_factor` | float | >1 is profitable |
| `var_95` | float | 5th percentile of returns, negative |
| `cvar_95` | float | mean of worst 5%, negative, ≤ var_95 |
| `tracking_error` | float | only if benchmark provided |
| `information_ratio` | float | only if benchmark provided |
| `turnover` | float | NaN if weights_history is None |

---

### VaRCalculator (`core/risk/var.py`)

Takes a `pd.Series` of **daily returns** (not NAV values). `returns = values.pct_change().dropna()`

```python
from core.risk.var import VaRCalculator

VaRCalculator.historical_var(returns, confidence=0.95)    # → float (negative)
VaRCalculator.parametric_var(returns, confidence=0.95)    # → float (negative, assumes normal)
VaRCalculator.historical_cvar(returns, confidence=0.95)   # → float (≤ var, negative)
VaRCalculator.rolling_var(returns, window=252, confidence=0.95)  # → pd.Series
# rolling_var: NaN for first (window-1) entries, then valid values
```

**Key invariants (always true):**
```
VaR_99 ≤ VaR_95 ≤ 0       higher confidence → more extreme (more negative)
CVaR_95 ≤ VaR_95           CVaR averages the losses beyond the VaR threshold
```

---

### GreeksCalculator (`core/risk/greeks.py`)

Wraps `BlackScholesModel` to produce surfaces and time series.

```python
from core.risk.greeks import GreeksCalculator
import numpy as np

# Surface: sweep spot × vol → 2D matrix per Greek
result = GreeksCalculator.compute_greeks_surface(
    spot_range     = np.linspace(70, 130, 40),
    vol_range      = np.linspace(0.05, 0.60, 40),
    strike         = 100.0,
    maturity       = 1.0,       # years
    risk_free_rate = 0.05,
)
# result keys: "spots", "vols", "delta", "gamma", "vega", "theta"
# each matrix: list of lists, shape (n_spots, n_vols)

# Time series: shrinking maturity along a real price path
df = GreeksCalculator.compute_greeks_over_time(
    price_history  = pd.Series([...], index=date_index),
    strike         = 100.0,
    risk_free_rate = 0.05,
    sigma          = 0.20,
)
# df.columns: ["delta", "gamma", "vega", "theta", "rho"]
# df.index: same as price_history.index
```

---

## 4. Mathematics

### Metrics Formulas

| Metric | Formula | Input |
|--------|---------|-------|
| Total Return | `(V_T - V_0) / V_0` | NAV series |
| Annualized Return | `(1 + TR)^(252/N) - 1` | NAV series, N=len |
| Annualized Vol | `std(pct_change) × √252` | NAV series |
| Sharpe | `(ann_ret - r_f) / ann_vol` | NAV + r_f |
| Sortino | `(ann_ret - r_f) / downside_vol` | downside_vol = std(neg returns only) × √252 |
| Max Drawdown | `min((V_t - cummax_t) / cummax_t)` | NAV series → always ≤ 0 |
| Calmar | `ann_ret / |MDD|` | NAV series |
| Win Rate | `count(r_t > 0) / count(r_t)` | returns |
| Profit Factor | `sum(r>0) / |sum(r<0)|` | returns |
| Tracking Error | `std(r_port - r_bench) × √252` | aligned returns |
| Information Ratio | `(ann_port - ann_bench) / TE` | two NAV series |
| Turnover | `mean(Σᵢ |Δwᵢ|)` over rebalancings | weights DataFrame |

### VaR Formulas

**Historical VaR** — empirical, no assumption:
```
VaR_α = quantile(returns, 1 - α)
VaR_95 = quantile(returns, 0.05)   ← 5th percentile of return distribution
```

**Parametric VaR** — assumes normal returns:
```
z = norm.ppf(α)              ← inverse normal CDF, e.g. 1.645 for 95%
VaR_α = μ - z × σ
```
Underestimates tail risk (real distributions have fatter tails than normal).

**CVaR (Expected Shortfall):**
```
CVaR_α = E[returns | returns ≤ VaR_α]
       = mean of all returns below VaR threshold
```

**Rolling VaR:**
```
rolling_VaR_α(t) = quantile(returns[t-window : t], 1 - α)
```
NaN for first `window-1` entries.

### Greeks Surface

Sweeps two axes to build a 2D heatmap per Greek:
- x-axis: implied volatility (sigma)
- y-axis: spot price (S)
- Fixed: K (strike), T (maturity), r (risk-free rate)

At each (S, sigma) cell: call `BlackScholesModel.delta/gamma/vega/theta`.

**Surface patterns to know:**

| Greek | ATM | Deep ITM (S >> K) | Deep OTM (S << K) | High vol |
|-------|-----|-------------------|-------------------|----------|
| Delta | ≈ 0.5 | → 1 (certain to exercise) | → 0 (worthless) | smooths transition |
| Gamma | **peaks** | → 0 | → 0 | decreases (wider dist) |
| Vega | largest | smaller | smaller | always positive |
| Theta | most negative | less negative | less negative | varies |

**Gamma trap:** near expiry ATM, gamma spikes — tiny price moves cause large delta swings → expensive frequent re-hedging.

### Greeks Over Time

As the option approaches expiry, maturity shrinks:
```
At step i of n total days:
  T_remaining = max((n - i) / 252, 1/252)   ← floor at 1 trading day
  Greeks = BlackScholesModel.*(S_i, K, T_remaining, r, sigma)
```

Delta increases toward 1 (ITM) or 0 (OTM) as time runs out. Gamma and vega shrink (less time for reversals). Theta accelerates (time value erodes faster near expiry).

---

## 5. Key Design Decisions

1. **Static-only classes** — all 3 classes have only `@staticmethod` methods. No `__init__`, no instance state. Pure functions of their inputs → easy to call from anywhere (backtester, API, notebook).

2. **NaN policy** — every metric returns `float("nan")` on invalid input (zero vol, empty data, zero denominator). Never raises, never returns 0 (which would be misleading). Callers use `math.isnan()` to check.

3. **VaR sign convention** — VaR is returned as a **negative number** (e.g. -0.02 = 2% loss). This is the industry standard. `VaR_95 = quantile(returns, 0.05)` is naturally negative for a portfolio that sometimes loses money.

4. **CVaR includes VaR observation** — `returns[returns <= var]` (≤ not <) includes the VaR boundary point. Ensures `cvar ≤ var` by construction.

5. **Local import in `compute_all`** — `from core.risk.var import VaRCalculator` sits inside the method body to avoid circular imports at module load time. Python caches the import so it's only slow on the first call.

6. **`risk_free_rate` from data provider** — the same rate used for cash accrual in the backtest loop is passed to `compute_all` for consistency. No magic constants.

7. **Greeks surface uses nested loops** — not vectorized numpy meshgrid. Slower but cleaner: each cell independently wrapped in `try/except` → bad cell gets NaN, not crash.

8. **`.tolist()` on numpy arrays** — the surface dicts use `.tolist()` to convert numpy arrays to pure Python lists, making them JSON-serializable by default (numpy arrays are not).

---

## 6. Python Patterns Used

### Static-only class
```python
class PerformanceMetrics:
    @staticmethod
    def sharpe_ratio(values: pd.Series, risk_free_rate: float = 0.05) -> float:
        ...
# Call: PerformanceMetrics.sharpe_ratio(values) — no instance needed
```

### `math.isnan()` vs alternatives
```python
math.isnan(x)   # for a Python float — fast, explicit
pd.isna(x)      # for float, None, pd.NaT, pd.NA — most flexible
np.isnan(x)     # for float and numpy arrays
# math.isnan crashes on pd.Series — don't use it on a Series
```

### `cummax()` for drawdown
```python
rolling_max = values.cummax()                  # high-water mark series
drawdown = (values - rolling_max) / rolling_max  # ≤ 0 always
mdd = float(drawdown.min())                    # worst trough
# Example: values=[100,110,120,105,115] → cummax=[100,110,120,120,120]
```

### `quantile()` for VaR
```python
returns.quantile(1 - confidence)
# confidence=0.95 → quantile(0.05) = 5th percentile = worst 5%
# Rolling version:
returns.rolling(window=60).quantile(0.05)
# NaN for first 59 entries, valid after that
```

### `norm.ppf()` — inverse normal CDF
```python
from scipy.stats import norm
norm.ppf(0.95)  # → 1.6449  (z-score where 95% of normal dist is below)
norm.ppf(0.05)  # → -1.6449 (5th percentile z-score)
# Parametric VaR: mu - norm.ppf(confidence) * sigma
```

### `diff().abs().sum(axis=1)` for turnover
```python
changes = weights_history.diff().dropna()    # row[t] - row[t-1]
per_date = changes.abs().sum(axis=1)         # sum across assets (axis=1=columns)
avg = float(per_date.mean())
# axis=1 → reduces columns → one value per row (per date)
# axis=0 → reduces rows → one value per column (per asset) — wrong
```

### `align(join="inner")` for synchronized arithmetic
```python
port_ret, bench_ret = port_ret.align(bench_ret, join="inner")
# Keeps only dates present in BOTH series
# Without this: port_ret - bench_ret produces NaN for unshared dates
```

### Nested loop + `np.empty()` + `try/except → NaN`
```python
n_s, n_v = len(spots), len(vols)
delta_arr = np.empty((n_s, n_v))    # pre-allocate, no init (faster than zeros)
for i, S in enumerate(spots):
    for j, sigma in enumerate(vols):
        try:
            delta_arr[i, j] = BlackScholesModel.delta(S, strike, T, r, sigma)
        except (ValueError, ZeroDivisionError):
            delta_arr[i, j] = np.nan
return {"delta": delta_arr.tolist()}  # .tolist() → JSON-serializable
```

### Local import to avoid circular deps
```python
@staticmethod
def compute_all(values, ...):
    from core.risk.var import VaRCalculator   # ← deferred to call time
    result["var_95"] = VaRCalculator.historical_var(returns, 0.95)
# Module-level import would create circular dep if var.py ever imports metrics.py
# Python caches modules — not re-imported on subsequent calls
```

### List-of-dicts → DataFrame
```python
records = []
for i, (date, S) in enumerate(price_history.items()):  # .items() on Series
    T = max((n - i) / 252, 1 / 252)
    records.append({"delta": ..., "gamma": ..., "rho": ...})
df = pd.DataFrame(records, index=price_history.index)
# Each dict → one row; keys → column names; index= assigns the date index
```

---

## 7. Tests (P4-25a through P4-25h)

File: `backend/tests/test_risk_metrics.py`

| ID | Test | What it checks |
|----|------|----------------|
| P4-25a | `test_total_return_doubles` | 100→200 series gives `total_return == 1.0` |
| P4-25b | `test_max_drawdown_no_drawdown` | monotone series → `max_drawdown == 0.0` |
| P4-25b | `test_max_drawdown_fifty_percent` | [100,200,100] → `max_drawdown == -0.5` |
| P4-25c | `test_sharpe_ratio_zero_vol_returns_nan` | constant NAV → Sharpe is NaN, not crash |
| P4-25d | `test_var_99_more_negative_than_var_95` | `VaR_99 < VaR_95` always |
| P4-25e | `test_cvar_more_negative_than_or_equal_to_var` | `CVaR_95 ≤ VaR_95` always |
| P4-25f | `test_rolling_var_same_length` | rolling_var output length = input length |
| P4-25f | `test_rolling_var_nan_prefix` | first (window-1) entries are NaN, rest are not |
| P4-25g | `test_compute_all_has_required_keys` | compute_all dict has all 11 required keys |
| P4-25h | `test_greeks_surface_keys_and_delta_range` | surface has right keys; delta ∈ [0,1] |

### Fixtures
```python
@pytest.fixture
def rising_series() -> pd.Series:
    return pd.Series(range(100, 201), dtype=float)   # 100→200, 101 pts, monotone

@pytest.fixture
def random_returns() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.001, 0.02, 500))   # 500 daily returns, seed=42
```

### Run commands
```bash
cd backend
.venv/bin/pytest tests/test_risk_metrics.py -v           # Phase 4 only
.venv/bin/pytest tests/ -v                               # all 107 tests
.venv/bin/pytest tests/ --cov=core --cov-report=term-missing
```

---

## 8. Demo Script (`backend/demos/phase4_demo.py`)

Run: `.venv/bin/python demos/phase4_demo.py`

**What it does:**
1. Runs 3 strategies (EqualWeight, Momentum, MinVariance) on 5 simulated assets (2022–2023, seed=7)
2. Prints a 14-metric table for all 3 strategies
3. Prints VaR/CVaR at 90/95/99% confidence for all 3
4. Saves 3 charts to `demos/`

**Charts produced:**

| File | Content |
|------|---------|
| `phase4_metrics_comparison.png` | 4-panel bar chart: Return/Risk, Risk-Adjusted, Trade Quality, VaR |
| `phase4_rolling_var.png` | Rolling 60-day historical VaR (95%) for each strategy |
| `phase4_greeks_surface.png` | 2×2 heatmaps: Delta, Gamma, Vega, Theta over (spot, vol) grid |

**Same data as phase3_demo** (seed=7, same 5 symbols) — results are directly comparable.

---

## 9. BacktestEngine Integration

After Phase 4, `BacktestEngine.run()` automatically computes risk metrics:

```python
# engine.py (simplified)
from core.risk.metrics import PerformanceMetrics

class BacktestEngine:
    def run(self, strategy, provider) -> BacktestResult:
        # ... backtest loop ...

        risk_metrics = PerformanceMetrics.compute_all(
            values          = pv_series,          # pd.Series of daily NAV
            benchmark       = benchmark_series,   # pd.Series or None
            weights_history = weights_df,         # pd.DataFrame of weights over time
            risk_free_rate  = risk_free_rate,     # from data_provider.get_risk_free_rate()
        )
        return BacktestResult(..., risk_metrics=risk_metrics)
```

`BacktestResult.risk_metrics` is always a populated dict after a run. Never empty.

---

## 10. Common Mistakes to Avoid

| Mistake | Correct approach |
|---------|-----------------|
| Pass NAV series to VaRCalculator | VaRCalculator takes **returns** (`values.pct_change().dropna()`) |
| Pass returns to PerformanceMetrics | PerformanceMetrics takes **NAV values** |
| Use `math.isnan()` on a pd.Series | Use `pd.isna(series)` for element-wise check |
| `axis=0` in turnover sum | Use `axis=1` to sum across assets (columns), giving one value per date |
| Forget `.align()` when subtracting two series | `port_ret - bench_ret` gives NaN for unshared dates without `.align()` |
| Read VaR_95 = -0.02 as a 2% gain | VaR is always a **loss**; -0.02 means 2% loss in the worst 5% of days |
| Interpret CVaR > VaR as a bug | CVaR ≤ VaR numerically (both negative) means CVaR is more negative |
