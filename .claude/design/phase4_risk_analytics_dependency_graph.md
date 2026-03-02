# Phase 4 — Risk Analytics Dependency Graph

> `backend/core/risk/` · `backend/core/backtester/engine.py` (updated) · Python 3.12
> Generated after session 4 (107/107 tests passing)

---

## Legend

```
──────►   uses / imports / depends on
══════►   returns / produces
   ▲      inherits from (is-a / subclass)
───┤      composes (has-a, stored as field)
TYPE      import only at type-check time (TYPE_CHECKING guard — not at runtime)
[ ]       concrete class or dataclass
( )       abstract class / interface (ABC)
{ }       free function / module-level function
⬡         external library (outside core/)
⚠         potential issue — resolved by local import or guard
```

---

## High-Level Overview — One New Package, Wired Into the Engine

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║   core/risk/  (new in Phase 4)          core/backtester/  (updated)             ║
║   ────────────────────────────          ─────────────────────────────            ║
║   Pure math — no web deps               Engine wires metrics in at the end      ║
║                                                                                  ║
║   [PerformanceMetrics]                  BacktestEngine.run()                    ║
║        13 static methods                    …                                    ║
║        compute_all() ◄──────────────── called here at step 9                   ║
║             │                               ║                                   ║
║             │ local import                  ▼                                   ║
║             ▼                           BacktestResult.risk_metrics             ║
║   [VaRCalculator]                           (was {} in Phase 3,                 ║
║        4 static methods                      now filled by Phase 4)             ║
║                                                                                  ║
║   [GreeksCalculator]                    core/pricing/  (Phase 1)                ║
║        2 static methods ──────────────► BlackScholesModel                       ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## Module 1 — `core/risk/metrics.py`  [P4-02 to P4-15]

```
  External
  ────────
  stdlib: math          pandas ⬡         Phase 4 internal
      │                    │                    │
      ▼                    ▼                    ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  metrics.py                                                                   │
  │                                                                               │
  │  ┌────────────────────────────────────────────────────────────────────────┐  │
  │  │  [PerformanceMetrics]   (all methods are @staticmethod — no state)     │  │
  │  │────────────────────────────────────────────────────────────────────────│  │
  │  │                                                                         │  │
  │  │  Inputs always come from BacktestResult fields:                         │  │
  │  │    values          : pd.Series    ← portfolio_values (DatetimeIndex)   │  │
  │  │    benchmark       : pd.Series    ← benchmark_values                   │  │
  │  │    weights_history : pd.DataFrame ← rebalancing dates × symbols        │  │
  │  │                                                                         │  │
  │  │  ── Return metrics ────────────────────────────────────────────────    │  │
  │  │                                                                         │  │
  │  │  total_return(values) → float                       [P4-03]            │  │
  │  │      (V_T − V_0) / V_0                                                 │  │
  │  │      len < 2 → NaN                                                     │  │
  │  │                                                                         │  │
  │  │  annualized_return(values, trading_days=252) → float  [P4-04]          │  │
  │  │      (1 + total_return)^(252/N) − 1                                    │  │
  │  │      N = len(values)                                                    │  │
  │  │                                                                         │  │
  │  │  annualized_volatility(values, trading_days=252) → float  [P4-05]      │  │
  │  │      std(daily_returns) × √252                                          │  │
  │  │      daily_returns = values.pct_change().dropna()                       │  │
  │  │      len < 2 → NaN                                                     │  │
  │  │                                                                         │  │
  │  │  sharpe_ratio(values, risk_free_rate=0.05) → float   [P4-06]           │  │
  │  │      (ann_return − r_f) / ann_vol                                       │  │
  │  │      ann_vol == 0 or NaN → NaN  (never crashes)                        │  │
  │  │                                                                         │  │
  │  │  sortino_ratio(values, risk_free_rate=0.05) → float  [P4-07]           │  │
  │  │      downside = returns[returns < 0]                                    │  │
  │  │      downside_vol = std(downside) × √252                                │  │
  │  │      (ann_return − r_f) / downside_vol                                  │  │
  │  │      fewer than 2 negative returns → NaN                                │  │
  │  │                                                                         │  │
  │  │  max_drawdown(values) → float                         [P4-08]           │  │
  │  │      rolling_max = values.cummax()                                      │  │
  │  │      dd = (values − rolling_max) / rolling_max                          │  │
  │  │      return dd.min()    ← always ≤ 0  (e.g. −0.33)                    │  │
  │  │      monotonically increasing series → 0.0                              │  │
  │  │                                                                         │  │
  │  │  calmar_ratio(values) → float                         [P4-09]           │  │
  │  │      ann_return / abs(max_drawdown)                                     │  │
  │  │      max_drawdown == 0 → NaN                                            │  │
  │  │                                                                         │  │
  │  │  win_rate(values) → float                             [P4-10]           │  │
  │  │      mean(returns > 0)   ∈ [0, 1]                                      │  │
  │  │                                                                         │  │
  │  │  profit_factor(values) → float                        [P4-11]           │  │
  │  │      sum(positive_returns) / abs(sum(negative_returns))                 │  │
  │  │      no negative returns → NaN                                          │  │
  │  │                                                                         │  │
  │  │  tracking_error(portfolio, benchmark) → float         [P4-12]           │  │
  │  │      active_returns = port_returns − bench_returns  (inner join dates)  │  │
  │  │      std(active_returns) × √252                                         │  │
  │  │                                                                         │  │
  │  │  information_ratio(portfolio, benchmark) → float      [P4-13]           │  │
  │  │      (ann_port_return − ann_bench_return) / tracking_error              │  │
  │  │      tracking_error == 0 → NaN                                          │  │
  │  │                                                                         │  │
  │  │  turnover(weights_history) → float                    [P4-14]           │  │
  │  │      changes = weights_history.diff().dropna()                          │  │
  │  │      mean( Σ_i |w_i,t − w_i,t−1| )  across rebalancing dates           │  │
  │  │      fewer than 2 rows → NaN                                            │  │
  │  │                                                                         │  │
  │  │  ── Aggregator ────────────────────────────────────────────────────    │  │
  │  │                                                                         │  │
  │  │  compute_all(                                          [P4-15]           │  │
  │  │      values          : pd.Series,                                       │  │
  │  │      benchmark       : pd.Series | None = None,                         │  │
  │  │      weights_history : pd.DataFrame | None = None,                      │  │
  │  │      risk_free_rate  : float = 0.05,                                    │  │
  │  │  ) → dict                                                               │  │
  │  │                                                                         │  │
  │  │  Always in output dict:                                                  │  │
  │  │      total_return, annualized_return, annualized_volatility,            │  │
  │  │      sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,           │  │
  │  │      win_rate, profit_factor, var_95, cvar_95,                          │  │
  │  │      turnover (NaN if no weights_history)                               │  │
  │  │                                                                         │  │
  │  │  Only if benchmark is not None:                                          │  │
  │  │      tracking_error, information_ratio                                   │  │
  │  │                                                                         │  │
  │  │  ⚠ var_95 / cvar_95 computed via local import:                         │  │
  │  │      from core.risk.var import VaRCalculator                            │  │
  │  │      (deferred to avoid module-level circular dependency)               │  │
  │  │                                                                         │  │
  │  └────────────────────────────────────────────────────────────────────────┘  │
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 2 — `core/risk/var.py`  [P4-17 to P4-21]

```
  External
  ────────
  scipy.stats.norm ⬡     pandas ⬡
         │                  │
         ▼                  ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  var.py                                                                       │
  │                                                                               │
  │  ┌────────────────────────────────────────────────────────────────────────┐  │
  │  │  [VaRCalculator]   (all methods are @staticmethod — no state)          │  │
  │  │────────────────────────────────────────────────────────────────────────│  │
  │  │                                                                         │  │
  │  │  Input: returns = pd.Series of daily returns  (pct_change().dropna())  │  │
  │  │         NOT portfolio NAV values — always pass returns                  │  │
  │  │                                                                         │  │
  │  │  historical_var(returns, confidence=0.95) → float      [P4-18]          │  │
  │  │      returns.quantile(1 − confidence)                                   │  │
  │  │      confidence=0.95 → 5th percentile → worst 5% of days               │  │
  │  │      result is negative  (e.g. −0.021 = lose 2.1% on bad days)         │  │
  │  │      empty returns → NaN                                                │  │
  │  │                                                                         │  │
  │  │  parametric_var(returns, confidence=0.95) → float      [P4-19]          │  │
  │  │      assumes returns ~ N(μ, σ)                                          │  │
  │  │      μ − z × σ   where z = norm.ppf(confidence)                        │  │
  │  │      less conservative than historical (misses fat tails)               │  │
  │  │      len < 2 → NaN                                                     │  │
  │  │                                                                         │  │
  │  │  historical_cvar(returns, confidence=0.95) → float     [P4-20]          │  │
  │  │      var = historical_var(returns, confidence)                          │  │
  │  │      tail = returns[returns ≤ var]                                      │  │
  │  │      return mean(tail)                                                  │  │
  │  │      always ≤ var  (deeper into the tail)                               │  │
  │  │      empty tail → return var itself                                     │  │
  │  │                                                                         │  │
  │  │  rolling_var(returns, window=252, confidence=0.95) → pd.Series [P4-21]  │  │
  │  │      returns.rolling(window=window).quantile(1 − confidence)            │  │
  │  │      same length as input — NaN for first window−1 entries              │  │
  │  │      useful for: risk regime detection, time-varying VaR charts         │  │
  │  │                                                                         │  │
  │  └────────────────────────────────────────────────────────────────────────┘  │
  └──────────────────────────────────────────────────────────────────────────────┘

  VaR Ordering Invariants (enforced by tests):
  ─────────────────────────────────────────────
  historical_var(r, 0.99) < historical_var(r, 0.95)   [P4-25d]
    (99% VaR captures a more extreme tail — more negative)

  historical_cvar(r, 0.95) ≤ historical_var(r, 0.95)  [P4-25e]
    (CVaR averages the tail losses — always at least as bad as VaR)
```

---

## Module 3 — `core/risk/greeks.py`  [P4-22 to P4-24]

```
  Phase 1
  ───────
  BlackScholesModel ⬡-like (internal)
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  greeks.py                                                                    │
  │                                                                               │
  │  ┌────────────────────────────────────────────────────────────────────────┐  │
  │  │  [GreeksCalculator]   (all methods are @staticmethod — no state)       │  │
  │  │────────────────────────────────────────────────────────────────────────│  │
  │  │                                                                         │  │
  │  │  compute_greeks_surface(                              [P4-23]           │  │
  │  │      spot_range   : list[float] | np.ndarray,                           │  │
  │  │      vol_range    : list[float] | np.ndarray,                           │  │
  │  │      strike       : float,                                              │  │
  │  │      maturity     : float,          ← years                            │  │
  │  │      risk_free_rate: float,                                             │  │
  │  │  ) → dict                                                               │  │
  │  │                                                                         │  │
  │  │  Implementation:                                                         │  │
  │  │      spots = np.asarray(spot_range)   shape: (n_s,)                    │  │
  │  │      vols  = np.asarray(vol_range)    shape: (n_v,)                    │  │
  │  │                                                                         │  │
  │  │      delta[i,j] = BS.delta(spots[i], K, T, r, vols[j])                │  │
  │  │      gamma[i,j] = BS.gamma(spots[i], K, T, r, vols[j])                │  │
  │  │      vega[i,j]  = BS.vega (spots[i], K, T, r, vols[j])                │  │
  │  │      theta[i,j] = BS.theta(spots[i], K, T, r, vols[j])                │  │
  │  │                                                                         │  │
  │  │      exception → NaN at that cell (no crash)                           │  │
  │  │                                                                         │  │
  │  │  Returns:                                                                │  │
  │  │      {                                                                  │  │
  │  │        "spots": list[float],          shape: (n_s,)                    │  │
  │  │        "vols":  list[float],          shape: (n_v,)                    │  │
  │  │        "delta": list[list[float]],    shape: (n_s, n_v)               │  │
  │  │        "gamma": list[list[float]],    shape: (n_s, n_v)               │  │
  │  │        "vega":  list[list[float]],    shape: (n_s, n_v)               │  │
  │  │        "theta": list[list[float]],    shape: (n_s, n_v)               │  │
  │  │      }                                                                  │  │
  │  │                                                                         │  │
  │  │  call delta ∈ [0, 1] for all (spot, vol) pairs  ← verified in tests   │  │
  │  │                                                                         │  │
  │  ├────────────────────────────────────────────────────────────────────────┤  │
  │  │                                                                         │  │
  │  │  compute_greeks_over_time(                        [P4-24]               │  │
  │  │      price_history : pd.Series,     ← spot at each date                │  │
  │  │      strike        : float,                                             │  │
  │  │      risk_free_rate: float,                                             │  │
  │  │      sigma         : float,         ← constant vol over time           │  │
  │  │  ) → pd.DataFrame                                                       │  │
  │  │                                                                         │  │
  │  │  Maturity model:                                                         │  │
  │  │      n = len(price_history)           total trading days                │  │
  │  │      T_remaining(i) = max((n − i) / 252, 1/252)                        │  │
  │  │      ← option starts with T = n/252 years, shrinks to 1-day floor      │  │
  │  │                                                                         │  │
  │  │  At each date i, spot S = price_history.iloc[i]:                        │  │
  │  │      delta = BS.delta(S, K, T_remaining, r, sigma)                     │  │
  │  │      gamma = BS.gamma(S, K, T_remaining, r, sigma)                     │  │
  │  │      vega  = BS.vega (S, K, T_remaining, r, sigma)                     │  │
  │  │      theta = BS.theta(S, K, T_remaining, r, sigma)                     │  │
  │  │      rho   = BS.rho  (S, K, T_remaining, r, sigma)                     │  │
  │  │      exception → row of NaN                                             │  │
  │  │                                                                         │  │
  │  │  Returns DataFrame: index=price_history.index, columns: [P4-24]        │  │
  │  │      delta, gamma, vega, theta, rho                                     │  │
  │  │                                                                         │  │
  │  └────────────────────────────────────────────────────────────────────────┘  │
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 4 — `core/risk/__init__.py`  [P4-01]

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  __init__.py                                                                  │
  │                                                                               │
  │  from core.risk.greeks  import GreeksCalculator                               │
  │  from core.risk.metrics import PerformanceMetrics                             │
  │  from core.risk.var     import VaRCalculator                                  │
  │                                                                               │
  │  __all__ = ["PerformanceMetrics", "VaRCalculator", "GreeksCalculator"]        │
  │                                                                               │
  │  Consumers can import as:                                                     │
  │      from core.risk import PerformanceMetrics, VaRCalculator                 │
  │      from core.risk.metrics import PerformanceMetrics    ← also works        │
  │                                                                               │
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 5 — `core/backtester/engine.py`  (updated)  [P4-16]

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  engine.py  (one addition only — everything else from Phase 3 unchanged)     │
  │                                                                               │
  │  NEW at top-level import:                                                     │
  │      from core.risk.metrics import PerformanceMetrics                         │
  │                                                                               │
  │  NEW in BacktestEngine.run() — step 9 (added after the loop):                │
  │                                                                               │
  │  # ── 9. Compute risk metrics [P4-16] ──────────────────────────────────    │
  │  risk_metrics = PerformanceMetrics.compute_all(                               │
  │      values          = pv_series,          ← pd.Series (DatetimeIndex)       │
  │      benchmark       = benchmark_series,   ← pd.Series or None               │
  │      weights_history = weights_df,         ← pd.DataFrame (rebalancing×sym)  │
  │      risk_free_rate  = risk_free_rate,     ← float from data_provider        │
  │  )                                                                            │
  │                                                                               │
  │  return BacktestResult(                                                        │
  │      …,                                                                       │
  │      risk_metrics = risk_metrics,   ← was {} in Phase 3, now fully populated │
  │      …,                                                                       │
  │  )                                                                            │
  │                                                                               │
  │  No other changes — self-financing loop, oracle, cost model all unchanged.   │
  │                                                                               │
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

## Inheritance Tree

```
  All risk classes are concrete, stateless, and static-only — no ABC needed.

  [PerformanceMetrics]     ← metrics.py  — 13 static methods + compute_all
  [VaRCalculator]          ← var.py      —  4 static methods
  [GreeksCalculator]       ← greeks.py   —  2 static methods

  No inheritance, no instantiation required.
  Usage pattern: PerformanceMetrics.sharpe_ratio(values)  ← class.method()
```

---

## Internal Import Graph (Phase 4 only)

```
  core/risk/__init__.py
       │
       ├──────────────────────► core/risk/metrics.py
       │                              │
       │                              │ (local import — deferred to call time)
       │                              └─────────────────► core/risk/var.py
       │
       ├──────────────────────► core/risk/var.py
       │                              │
       │                              └──── scipy.stats.norm ⬡
       │
       └──────────────────────► core/risk/greeks.py
                                      │
                                      └──── core/pricing/black_scholes.py  (Phase 1)

  ⚠ Why local import in compute_all()?
    metrics.py imports var.py at call time (inside the function body).
    This avoids any potential circular import if var.py were ever to import
    from metrics.py in the future. It is also self-documenting: the reader
    knows VaRCalculator is only needed for var_95/cvar_95.
```

---

## Full Cross-Phase Dependency Map

```
  Phase 1                    Phase 2        Phase 3                Phase 4
  ───────                    ───────        ───────                ───────

  core/pricing/
  ┌──────────────────────┐
  │  [BlackScholesModel] │◄──────────────────────────────── GreeksCalculator
  │  .delta()            │   called for each (spot, vol) cell
  │  .gamma()            │   and each date in price_history
  │  .vega()             │
  │  .theta()            │
  │  .rho()              │
  └──────────────────────┘

  core/models/results.py
  ┌──────────────────────┐
  │  [BacktestResult]    │
  │  .portfolio_values   │──────────────────────────────────► PerformanceMetrics
  │   (pd.Series)        │   .total_return()
  │  .benchmark_values   │──────────────────────────────────► PerformanceMetrics
  │   (pd.Series|None)   │   .tracking_error(), .information_ratio()
  │  .weights_history    │──────────────────────────────────► PerformanceMetrics
  │   (pd.DataFrame)     │   .turnover()
  │  .risk_metrics       │◄══════════════════════════════════ compute_all() output
  │   (dict, was {})     │   Phase 4 fills what Phase 3 left empty
  └──────────────────────┘

  core/backtester/engine.py
  ┌──────────────────────┐
  │  [BacktestEngine]    │──────────────────────────────────► PerformanceMetrics
  │  .run()              │   calls compute_all() after loop
  │  risk_free_rate      │──────────────────────────────────► compute_all(rfr=…)
  │   (from provider)    │   same rfr used for cash accrual
  └──────────────────────┘

  Phase 4 → Phase 1, 3: YES (reads BlackScholesModel, BacktestResult)
  Phase 1, 2, 3 → Phase 4: NONE (core phases have zero knowledge of risk/)
```

---

## Data Flow — From BacktestResult to risk_metrics

```
  BacktestEngine.run() completes loop
         │
         ├─ pv_series        = pd.Series(portfolio_values_dict)
         │                     DatetimeIndex → float NAV
         │
         ├─ benchmark_series = _compute_benchmark(prices, symbols, initial_value)
         │                     equal-weight buy-and-hold
         │
         ├─ weights_df       = pd.DataFrame(weights_records).T
         │                     rows = rebalancing dates, cols = symbols
         │
         └─ risk_free_rate   = data_provider.get_risk_free_rate(start_date)
                               scalar float (e.g. 0.05 from ^IRX or fallback)

         ↓
  PerformanceMetrics.compute_all(pv_series, benchmark_series, weights_df, rfr)
         │
         ├─ returns = pv_series.pct_change().dropna()   ← used by VaR
         │
         ├─ total_return          = (V_T − V_0) / V_0
         ├─ annualized_return     = (1 + tr)^(252/N) − 1
         ├─ annualized_volatility = std(returns) × √252
         ├─ sharpe_ratio          = (ann_ret − rfr) / ann_vol
         ├─ sortino_ratio         = (ann_ret − rfr) / downside_vol
         ├─ max_drawdown          = min((values − cummax) / cummax)
         ├─ calmar_ratio          = ann_ret / |max_drawdown|
         ├─ win_rate              = mean(returns > 0)
         ├─ profit_factor         = Σ(pos) / |Σ(neg)|
         ├─ var_95                = VaRCalculator.historical_var(returns, 0.95)
         ├─ cvar_95               = VaRCalculator.historical_cvar(returns, 0.95)
         ├─ tracking_error        = std(port_ret − bench_ret) × √252
         ├─ information_ratio     = (ann_port − ann_bench) / te
         └─ turnover              = mean(Σ|w_t − w_{t−1}|) over rebalancing dates

         ↓
  dict → stored in BacktestResult.risk_metrics
```

---

## NaN Policy

```
  Every metric function returns float("nan") instead of crashing when:
  ────────────────────────────────────────────────────────────────────
  Metric                Condition that returns NaN
  ─────────────────     ──────────────────────────
  total_return          len(values) < 2
  annualized_return     len(values) < 2
  annualized_volatility len(returns) < 2
  sharpe_ratio          ann_vol == 0 or NaN           ← zero-vol edge case [P4-25c]
  sortino_ratio         fewer than 2 negative returns
  calmar_ratio          max_drawdown == 0
  profit_factor         no negative returns
  tracking_error        fewer than 2 aligned dates
  information_ratio     tracking_error == 0 or NaN
  turnover              weights_history has < 2 rows

  All NaN checks use math.isnan(v) — works for float("nan"), not for numpy NaN
  in general (but inputs are always Python floats here).

  Consumers (Phase 5 API) must sanitize NaN → None before JSON serialization.
```

---

## tests/test_risk_metrics.py  [P4-25]

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  test_risk_metrics.py   (10 tests — all pass)                                 │
  │                                                                               │
  │  Fixtures:                                                                    │
  │      rising_series   pd.Series(range(100, 201)) — 101 pts, 100→200           │
  │      random_returns  pd.Series(normal(0.001, 0.02, 500), seed=42)            │
  │                                                                               │
  │  [P4-25a] total_return_doubles                                                │
  │      rising_series: 100 → 200  →  total_return == 1.0                        │
  │                                                                               │
  │  [P4-25b] max_drawdown_no_drawdown                                            │
  │      rising_series (monotonically up) → max_drawdown == 0.0                  │
  │                                                                               │
  │  [P4-25b] max_drawdown_fifty_percent                                          │
  │      [100, 200, 100]: peak=200, trough=100 → max_drawdown ≈ −0.5             │
  │                                                                               │
  │  [P4-25c] sharpe_ratio_zero_vol_returns_nan                                  │
  │      constant series [100.0 × 100] → std=0 → Sharpe = NaN, no crash         │
  │                                                                               │
  │  [P4-25d] var_99_more_negative_than_var_95                                   │
  │      VaR(r, 0.99) < VaR(r, 0.95)  (deeper percentile = more negative)       │
  │                                                                               │
  │  [P4-25e] cvar_more_negative_than_or_equal_to_var                            │
  │      CVaR(r, 0.95) ≤ VaR(r, 0.95)  (average of tail ≤ threshold)           │
  │                                                                               │
  │  [P4-25f] rolling_var_same_length                                             │
  │      len(rolling_var(r, window=100)) == len(r)                               │
  │                                                                               │
  │  [P4-25f] rolling_var_nan_prefix                                              │
  │      first window−1 entries are NaN; entries from window onward are not NaN  │
  │                                                                               │
  │  [P4-25g] compute_all_has_required_keys                                       │
  │      required = {total_return, annualized_return, annualized_volatility,     │
  │                  sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,    │
  │                  var_95, cvar_95, win_rate, turnover}                        │
  │      all ⊆ compute_all(values).keys()                                        │
  │                                                                               │
  │  [P4-25h] greeks_surface_keys_and_delta_range                                │
  │      result.keys() == {"spots","vols","delta","gamma","vega","theta"}        │
  │      delta.shape == (len(spot_range), len(vol_range))                        │
  │      all delta ∈ [0, 1]  (call option invariant)                             │
  │                                                                               │
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

## Dependency Count Summary

| Class | Depends On (internal) | Depended On By |
|---|---|---|
| `PerformanceMetrics` | stdlib `math`, `pandas`, `VaRCalculator` (local) | `BacktestEngine.run()`, tests, API (Phase 5) |
| `VaRCalculator` | `pandas`, `scipy.stats.norm` | `PerformanceMetrics.compute_all()`, tests, API (Phase 5) |
| `GreeksCalculator` | `numpy`, `pandas`, `BlackScholesModel` (Phase 1) | tests, API (Phase 5) |

---

## What This Means for Phase 5

Phase 5 adds `backend/api/`. It will:

1. **Call** `PerformanceMetrics.compute_all()` indirectly — already in `BacktestResult.risk_metrics`
2. **Call** `PerformanceMetrics.compute_all()` directly — for `POST /risk/analyze` endpoint
3. **Call** `VaRCalculator` directly — for per-confidence-level breakdown in API response
4. **Call** `GreeksCalculator.compute_greeks_surface()` — for the Greeks surface endpoint
5. **Sanitize NaN → None** before returning risk_metrics in JSON responses

```
  Phase 4                             Phase 5 (api/)
  ───────                             ──────────────
  PerformanceMetrics ────────────────► routes/risk.py (POST /risk/analyze)
  VaRCalculator      ────────────────► routes/risk.py (VaR breakdown)
  GreeksCalculator   ────────────────► routes/pricing.py (Greeks surface)
  BacktestResult.risk_metrics ───────► routes/backtest.py (already populated)
```

No changes needed to any Phase 1, 2, 3, or 4 code.
