# Phase 2 — Strategy Framework

> **Goal**: Implement the pluggable strategy system and all 8 strategies. By the end of this phase you can call any strategy's `compute_weights()` on historical prices and get back a portfolio allocation.

**Prerequisites**: Phase 1 complete (pricing engine and data models available).

---

## Abstract Interface (`backend/core/strategies/`)

- [ ] **[P2-01]** Create `core/strategies/__init__.py`
- [ ] **[P2-02]** Create `core/strategies/base.py` with these three items:
  - `StrategyConfig` dataclass: `name: str`, `description: str`, `rebalancing_frequency: str = "weekly"`, `transaction_cost_bps: float = 10.0`
  - `PortfolioWeights` dataclass: `weights: dict[str, float]`, `cash_weight: float`, `timestamp: date`
  - `IStrategy(ABC)` abstract class with:
    - `__init__(self, config: StrategyConfig)`
    - Abstract method `compute_weights(current_date, price_history, current_portfolio) -> PortfolioWeights`
    - Abstract property `required_history_days -> int`
    - Abstract classmethod `get_param_schema() -> dict` (returns a JSON-schema-like dict for the frontend to render a form)

---

## Strategy Registry (`core/strategies/registry.py`)

- [ ] **[P2-03]** Create `core/strategies/registry.py` with `StrategyRegistry` class:
  - Class-level `_strategies: dict[str, type[IStrategy]] = {}`
  - `register(strategy_class)` classmethod — decorator that adds the class to `_strategies` by its `__name__`
  - `list_strategies() -> list[dict]` — return name, description, family, and param schema for each registered strategy
  - `create(name: str, config: dict) -> IStrategy` — look up by name and instantiate with the given config dict

---

## Hedging Strategies (`core/strategies/hedging/`)

- [ ] **[P2-04]** Create `core/strategies/hedging/__init__.py`
- [ ] **[P2-05]** Create `core/strategies/hedging/delta_hedge.py` — `@StrategyRegistry.register class DeltaHedgeStrategy(IStrategy)`
  - Config params: `option_weights: list[float]`, `strike: float`, `maturity_years: float`, `n_simulations: int = 50_000`, `rebalancing_frequency: str = "weekly"`, `risk_free_rate: float = 0.05`, `volatilities: list[float]`, `correlation: list[list[float]]`
  - `required_history_days` → `1`
  - `compute_weights`: call `MonteCarloPricer.compute_deltas()` using current prices and remaining maturity; return `weights = {symbol: delta_i * S_i / V}` (delta-shares expressed as portfolio weights), `cash_weight = 1 - sum(weights)`
  - `get_param_schema`: return JSON schema dict for the config fields above
- [ ] **[P2-06]** Create `core/strategies/hedging/delta_gamma_hedge.py` — `@StrategyRegistry.register class DeltaGammaHedgeStrategy(IStrategy)` (stub: same as delta hedge but document that gamma hedging with a second instrument is planned; compute_weights falls back to delta hedge for now)

---

## Allocation Strategies (`core/strategies/allocation/`)

- [ ] **[P2-07]** Create `core/strategies/allocation/__init__.py`
- [ ] **[P2-08]** Create `core/strategies/allocation/equal_weight.py` — `@StrategyRegistry.register class EqualWeightStrategy(IStrategy)`
  - `required_history_days` → `1`
  - `compute_weights`: assign `1/n` to each symbol, `cash_weight = 0`
  - `get_param_schema`: only `rebalancing_frequency`
- [ ] **[P2-09]** Create `core/strategies/allocation/min_variance.py` — `@StrategyRegistry.register class MinVarianceStrategy(IStrategy)`
  - Config params: `lookback_window: int = 60`, `rebalancing_frequency: str = "monthly"`
  - `required_history_days` → `lookback_window + 1`
  - `compute_weights`:
    - Compute daily returns from `price_history.tail(lookback_window)`
    - Estimate covariance matrix `Σ` from returns
    - Solve `min ω^T Σ ω` s.t. `Σω_i = 1`, `ω_i >= 0` using `scipy.optimize.minimize` with SLSQP
    - Return weights dict and `cash_weight = 0`
- [ ] **[P2-10]** Create `core/strategies/allocation/max_sharpe.py` — `@StrategyRegistry.register class MaxSharpeStrategy(IStrategy)`
  - Config params: `lookback_window: int = 60`, `risk_free_rate_override: float | None = None`, `rebalancing_frequency: str = "monthly"`
  - `required_history_days` → `lookback_window + 1`
  - `compute_weights`:
    - Estimate mean returns `μ` and covariance `Σ` from rolling window
    - Use `r_f` from `risk_free_rate_override` or `data_provider.get_risk_free_rate(current_date)`
    - Maximize `(ω^T μ - r_f) / sqrt(ω^T Σ ω)` (equivalently minimize the negative Sharpe) s.t. `Σω_i = 1`, `ω_i >= 0`
- [ ] **[P2-11]** Create `core/strategies/allocation/risk_parity.py` — `@StrategyRegistry.register class RiskParityStrategy(IStrategy)`
  - Config params: `lookback_window: int = 60`, `rebalancing_frequency: str = "monthly"`
  - `required_history_days` → `lookback_window + 1`
  - `compute_weights`:
    - Estimate covariance `Σ`
    - Minimize `Σ (RC_i - 1/n)^2` where risk contribution `RC_i = ω_i * (Σω)_i / (ω^T Σ ω)` using SLSQP
    - Normalize final weights so they sum to 1

---

## Signal Strategies (`core/strategies/signal/`)

- [ ] **[P2-12]** Create `core/strategies/signal/__init__.py`
- [ ] **[P2-13]** Create `core/strategies/signal/momentum.py` — `@StrategyRegistry.register class MomentumStrategy(IStrategy)`
  - Config params: `lookback_period: int = 252`, `top_k: int = 3`, `long_only: bool = True`, `rebalancing_frequency: str = "monthly"`
  - `required_history_days` → `lookback_period + 1`
  - `compute_weights`:
    - Compute trailing return for each symbol over `lookback_period` days
    - Rank all symbols by trailing return (descending)
    - Long-only: assign equal weight `1/top_k` to the top `k` symbols, `0` to the rest
    - Long-short: assign `+1/(2*k)` to top k, `-1/(2*k)` to bottom k (weights can be negative)
- [ ] **[P2-14]** Create `core/strategies/signal/mean_reversion.py` — `@StrategyRegistry.register class MeanReversionStrategy(IStrategy)`
  - Config params: `lookback_window: int = 20`, `z_threshold: float = 2.0`, `rebalancing_frequency: str = "weekly"`
  - `required_history_days` → `lookback_window + 1`
  - `compute_weights`:
    - Compute z-score per symbol: `z_i = (S_i - MA_i) / std_i` using `lookback_window`
    - Buy symbols with `z < -z_threshold` (mean-reversion long signal)
    - Avoid/short symbols with `z > +z_threshold`
    - Weights inversely proportional to `|z_i|` among triggered symbols; normalize so they sum to 1

---

## Tests (`backend/tests/`)

- [ ] **[P2-15]** Create `tests/test_strategies.py`
  - **[P2-15a]** Equal weight: for any n assets, all weights equal `1/n` and sum to 1
  - **[P2-15b]** Min-variance: weights are non-negative and sum to 1
  - **[P2-15c]** Max-Sharpe: weights are non-negative and sum to 1
  - **[P2-15d]** Risk parity: risk contributions are (approximately) equal across assets
  - **[P2-15e]** Momentum: returned weights contain exactly `top_k` non-zero entries (long-only mode)
  - **[P2-15f]** Mean reversion: only symbols with `|z| > threshold` get non-zero weights
  - **[P2-15g]** No look-ahead bias: run each strategy on a DataFrame truncated at date `t`, then extend by one day — weights at `t` must not change
  - **[P2-15h]** Registry: `StrategyRegistry.list_strategies()` returns at least 6 strategies; `StrategyRegistry.create("EqualWeightStrategy", {...})` returns an `IStrategy` instance
  - **[P2-15i]** Delta hedge: `compute_weights()` returns weights that sum to ≤ 1 (delta-hedging can hold cash)

---

## How to Run Tests

```bash
cd backend
pytest tests/test_strategies.py -v
pytest tests/ --cov=core --cov-report=term-missing
```

---

## Definition of Done

- All tests in `[P2-15]` pass
- `StrategyRegistry.list_strategies()` returns all 7 concrete strategies (including stub DeltaGamma)
- Running any strategy's `compute_weights()` on a 252-row price DataFrame returns a valid `PortfolioWeights` with weights summing to ≤ 1 and no future data accessed
