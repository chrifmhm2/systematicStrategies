# Phase 3 — Backtesting Engine

> **Goal**: Build the core backtest loop, rebalancing oracles, transaction cost model, and the Yahoo Finance data provider. By the end you can run a full backtest of any Phase 2 strategy on real market data and get back a complete result object.

**Prerequisites**: Phase 1 (pricing + data models) and Phase 2 (strategy framework) complete.

---

## Complete `BacktestResult` Model

- [ ] **[P3-01]** Update `core/models/results.py` — fill in `BacktestResult` dataclass:
  - `portfolio_values: pd.Series` (DatetimeIndex → portfolio value)
  - `benchmark_values: pd.Series | None`
  - `weights_history: pd.DataFrame` (DatetimeIndex × symbol → weight at each rebalancing date)
  - `trades_log: list[dict]` (each dict: `date`, `symbol`, `quantity`, `price`, `cost`)
  - `risk_metrics: dict` (filled in Phase 4; empty dict for now)
  - `config: BacktestConfig`
  - `strategy_name: str`
  - `computation_time_ms: float`

---

## Backtesting Configs and Engine (`backend/core/backtester/`)

- [ ] **[P3-02]** Create `core/backtester/__init__.py`
- [ ] **[P3-03]** Create `core/backtester/engine.py`
  - `BacktestConfig` dataclass: `initial_value: float = 100_000.0`, `start_date: date = None`, `end_date: date = None`, `data_provider: str = "yahoo"`, `symbols: list[str] = None`, `transaction_cost_bps: float = 10.0`, `slippage_bps: float = 5.0`
  - `BacktestEngine` class:
    - `__init__(self, config: BacktestConfig)`
    - `run(self, strategy: IStrategy, data_provider: IDataProvider) -> BacktestResult`

- [ ] **[P3-04]** Implement the core loop inside `BacktestEngine.run()`:
  1. Fetch all prices from `data_provider.get_prices(symbols, start_date, end_date)`
  2. Initialize portfolio: hold cash = `initial_value`, positions empty
  3. For each date `t` in the price series (chronological order):
     - a. Mark portfolio to market: `V_t = Σ q_i * S_i^t + cash * (1 + r * dt)` where `dt = 1/252`
     - b. If `rebalancing_oracle.should_rebalance(t, portfolio)`:
       - Pass only `prices.loc[:t]` to `strategy.compute_weights()` (no future data)
       - Compute trade quantities: `q_i_new = weights_i * V_t / S_i^t`
       - For each changed position, compute trade cost via `TransactionCostModel`
       - Deduct costs from cash
       - Update positions and cash so portfolio remains **self-financing** (value before = value after minus costs)
       - Log the trade
  4. Record portfolio value and weights at each step
  5. Return `BacktestResult`

- [ ] **[P3-05]** Enforce the self-financing invariant: before and after rebalancing (excluding costs), portfolio value is identical. Add an assertion during backtesting (or a tolerance check) so this is caught early if violated.

---

## Rebalancing Oracles (`core/backtester/rebalancing.py`)

- [ ] **[P3-06]** Create `core/backtester/rebalancing.py`
- [ ] **[P3-07]** Implement abstract `RebalancingOracle(ABC)` with `should_rebalance(current_date: date, portfolio) -> bool`
- [ ] **[P3-08]** Implement `PeriodicRebalancing(RebalancingOracle)`:
  - Constructor: `frequency: str` — accepts `"daily"`, `"weekly"` (Monday), `"monthly"` (first trading day of month)
  - `should_rebalance`: check weekday (Monday=0) or month boundary
- [ ] **[P3-09]** Implement `ThresholdRebalancing(RebalancingOracle)`:
  - Constructor: `threshold: float = 0.05` (5% drift)
  - `should_rebalance`: return `True` if any asset weight has drifted more than `threshold` from its last target weight
- [ ] **[P3-10]** Connect `BacktestConfig.rebalancing_frequency` string to the correct `RebalancingOracle` instance inside `BacktestEngine.run()`

---

## Transaction Cost Model (`core/backtester/costs.py`)

- [ ] **[P3-11]** Create `core/backtester/costs.py`
- [ ] **[P3-12]** Implement `TransactionCostModel`:
  - Constructor: `commission_bps: float = 10.0`, `slippage_bps: float = 5.0`, `min_commission: float = 1.0`
  - `compute_cost(trade_value: float) -> float`:
    - `cost = abs(trade_value) * (commission_bps + slippage_bps) / 10_000`
    - Return `max(cost, min_commission)`

---

## Yahoo Finance Data Provider (`core/data/yahoo.py`)

- [ ] **[P3-13]** Create `core/data/yahoo.py` — `YahooDataProvider(IDataProvider)`
  - Constructor: `cache: bool = True`
  - `get_prices(symbols, start_date, end_date)`:
    - Use `yfinance.download(symbols, start=start_date, end=end_date, auto_adjust=True)`
    - Return DataFrame with DatetimeIndex, one column per symbol
    - If `cache=True`, store result in an in-memory dict keyed by `(tuple(symbols), start, end)` to avoid duplicate API calls
  - `get_risk_free_rate(date)`:
    - Attempt to fetch `^IRX` (13-week T-bill) from yfinance; return as decimal (divide by 100)
    - Fall back to `0.05` if the fetch fails
- [ ] **[P3-14]** Add a default asset universe constant in `yahoo.py`:
  ```python
  DEFAULT_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ",
                      "WMT", "PG", "MA", "HD", "UNH", "DIS", "BAC", "XOM", "PFE", "BRK-B"]
  ```

---

## Tests (`backend/tests/`)

- [ ] **[P3-15]** Create `tests/test_backtester.py`
  - **[P3-15a]** Self-financing constraint: run a backtest with `EqualWeightStrategy`; assert that portfolio value immediately before and immediately after every rebalancing differs by at most the transaction costs (within floating-point tolerance)
  - **[P3-15b]** Transaction costs: run two identical backtests, one with `commission_bps=0` and one with `commission_bps=10`; assert the zero-cost portfolio has strictly higher final value
  - **[P3-15c]** No look-ahead bias: mock `strategy.compute_weights` to record the `price_history` it receives; assert it never contains dates after the current backtest date
  - **[P3-15d]** Date range: `BacktestResult.portfolio_values` index starts on or after `start_date` and ends on or before `end_date`
  - **[P3-15e]** Trades log: each trade dict contains keys `date`, `symbol`, `quantity`, `price`, `cost`
  - **[P3-15f]** Periodic rebalancing — weekly: `should_rebalance` returns `True` only on Mondays (or next trading day)
  - **[P3-15g]** Threshold rebalancing: no rebalancing fires when all weights are within threshold; one fires when a weight drifts beyond threshold

---

## How to Run Tests

```bash
cd backend
pytest tests/test_backtester.py -v
pytest tests/ --cov=core --cov-report=term-missing
```

---

## Definition of Done

- All tests in `[P3-15]` pass
- The following end-to-end snippet runs without errors:

```python
from datetime import date
from core.data.simulated import SimulatedDataProvider
from core.strategies.allocation.equal_weight import EqualWeightStrategy
from core.strategies.base import StrategyConfig
from core.backtester.engine import BacktestEngine, BacktestConfig
import numpy as np

provider = SimulatedDataProvider(
    spots={"AAPL": 150.0, "MSFT": 280.0},
    volatilities={"AAPL": 0.2, "MSFT": 0.25},
    correlation=np.array([[1.0, 0.5], [0.5, 1.0]]),
    seed=42,
)
strategy = EqualWeightStrategy(StrategyConfig(name="EW", description=""))
config = BacktestConfig(
    initial_value=100_000,
    start_date=date(2023, 1, 1),
    end_date=date(2023, 12, 31),
    symbols=["AAPL", "MSFT"],
)
engine = BacktestEngine(config)
result = engine.run(strategy, provider)
print(result.portfolio_values.tail())
print(f"Trades executed: {len(result.trades_log)}")
```
