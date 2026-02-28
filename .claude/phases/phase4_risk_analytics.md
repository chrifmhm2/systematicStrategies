# Phase 4 — Risk Analytics

> **Goal**: Implement all performance metrics, VaR/CVaR calculators, and the Greeks surface calculator. Wire the risk metrics into `BacktestResult` so every backtest automatically computes them.

**Prerequisites**: Phase 3 complete (`BacktestResult` is being produced).

---

## Performance Metrics (`backend/core/risk/metrics.py`)

- [ ] **[P4-01]** Create `core/risk/__init__.py`
- [ ] **[P4-02]** Create `core/risk/metrics.py` with a `PerformanceMetrics` class (or module-level functions)
- [ ] **[P4-03]** Implement `total_return(values: pd.Series) -> float` — `(V_T - V_0) / V_0`
- [ ] **[P4-04]** Implement `annualized_return(values: pd.Series, trading_days: int = 252) -> float` — `(1 + total_return)^(252/N) - 1`
- [ ] **[P4-05]** Implement `annualized_volatility(values: pd.Series, trading_days: int = 252) -> float` — `std(daily_returns) * sqrt(252)`
- [ ] **[P4-06]** Implement `sharpe_ratio(values: pd.Series, risk_free_rate: float = 0.05) -> float` — `(ann_return - r_f) / ann_vol`; return `NaN` if `ann_vol == 0`
- [ ] **[P4-07]** Implement `sortino_ratio(values: pd.Series, risk_free_rate: float = 0.05) -> float` — like Sharpe but use downside volatility (std of negative returns only × √252)
- [ ] **[P4-08]** Implement `max_drawdown(values: pd.Series) -> float` — `max((peak - trough) / peak)` over the series; return as a negative number (e.g. `-0.33`)
- [ ] **[P4-09]** Implement `calmar_ratio(values: pd.Series) -> float` — `annualized_return / abs(max_drawdown)`; return `NaN` if `max_drawdown == 0`
- [ ] **[P4-10]** Implement `win_rate(values: pd.Series) -> float` — fraction of positive daily returns
- [ ] **[P4-11]** Implement `profit_factor(values: pd.Series) -> float` — `sum(positive_returns) / abs(sum(negative_returns))`
- [ ] **[P4-12]** Implement `tracking_error(portfolio: pd.Series, benchmark: pd.Series) -> float` — `std(portfolio_returns - benchmark_returns) * sqrt(252)`
- [ ] **[P4-13]** Implement `information_ratio(portfolio: pd.Series, benchmark: pd.Series) -> float` — `(ann_portfolio_return - ann_bench_return) / tracking_error`
- [ ] **[P4-14]** Implement `turnover(weights_history: pd.DataFrame) -> float` — mean of the total absolute weight change at each rebalancing: `mean(Σ_i |w_i,t - w_i,t-1|)` across rebalancing dates
- [ ] **[P4-15]** Implement `compute_all(values: pd.Series, benchmark: pd.Series | None, weights_history: pd.DataFrame | None, risk_free_rate: float = 0.05) -> dict` — calls all of the above and returns a single dict with keys matching the API response schema (`total_return`, `annualized_return`, `annualized_volatility`, `sharpe_ratio`, etc.)

---

## Wire Metrics into BacktestEngine

- [ ] **[P4-16]** At the end of `BacktestEngine.run()` (in `core/backtester/engine.py`), call `compute_all()` and store the result in `BacktestResult.risk_metrics`

---

## Value at Risk (`core/risk/var.py`)

- [ ] **[P4-17]** Create `core/risk/var.py` with `VaRCalculator` class
- [ ] **[P4-18]** Implement `historical_var(returns: pd.Series, confidence: float = 0.95) -> float` — sort returns, take the `(1 - confidence)` quantile; return as a negative number
- [ ] **[P4-19]** Implement `parametric_var(returns: pd.Series, confidence: float = 0.95) -> float` — assume normal distribution; `μ - z * σ` where `z = norm.ppf(confidence)`
- [ ] **[P4-20]** Implement `historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float` — mean of returns below the VaR threshold (expected shortfall)
- [ ] **[P4-21]** Implement `rolling_var(returns: pd.Series, window: int = 252, confidence: float = 0.95) -> pd.Series` — compute historical VaR at each date using a rolling window

---

## Greeks Surface Calculator (`core/risk/greeks.py`)

- [ ] **[P4-22]** Create `core/risk/greeks.py` with `GreeksCalculator` class
- [ ] **[P4-23]** Implement `compute_greeks_surface(spot_range, vol_range, strike, maturity, risk_free_rate) -> dict`
  - Create a meshgrid of `(spot, vol)` pairs
  - For each pair, compute delta, gamma, vega, theta using `BlackScholesModel`
  - Return a dict: `{"spots": [...], "vols": [...], "delta": [[...]], "gamma": [[...]], "vega": [[...]], "theta": [[...]]}`
  - Shape of each surface matrix: `(len(spot_range), len(vol_range))`
- [ ] **[P4-24]** Implement `compute_greeks_over_time(price_history: pd.Series, strike, risk_free_rate, sigma) -> pd.DataFrame`
  - At each date compute BS Greeks for the option with the remaining maturity
  - Return a DataFrame with columns `delta`, `gamma`, `vega`, `theta`, `rho`

---

## Tests (`backend/tests/`)

- [ ] **[P4-25]** Create `tests/test_risk_metrics.py`
  - **[P4-25a]** `total_return`: a series going from 100 to 200 gives `total_return = 1.0`
  - **[P4-25b]** `max_drawdown`: a monotonically increasing series gives `0.0`; a series that drops 50% from peak gives `≈ -0.5`
  - **[P4-25c]** `sharpe_ratio`: a constant-return series (zero volatility) returns `NaN` without crashing
  - **[P4-25d]** VaR ordering: `historical_var(returns, 0.99) < historical_var(returns, 0.95)` (99% VaR is more negative)
  - **[P4-25e]** CVaR is always ≤ VaR (CVaR is the mean of tail losses, so it's more negative)
  - **[P4-25f]** `rolling_var` returns a Series of the same length as input (with `NaN` for the first `window-1` values)
  - **[P4-25g]** `compute_all` returns a dict containing all required keys: `total_return`, `annualized_return`, `annualized_volatility`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `calmar_ratio`, `var_95`, `cvar_95`, `win_rate`, `turnover`
  - **[P4-25h]** Greeks surface: returned dict has keys `spots`, `vols`, `delta`, `gamma`, `vega`, `theta`; delta surface values are all in `[0, 1]` for a call

---

## How to Run Tests

```bash
cd backend
pytest tests/test_risk_metrics.py -v
pytest tests/ --cov=core --cov-report=term-missing
```

---

## Definition of Done

- All tests in `[P4-25]` pass
- After running a backtest with `BacktestEngine.run()`, `result.risk_metrics` is a non-empty dict containing all keys from `[P4-25g]`
- Coverage on `core/risk/` ≥ 80%
