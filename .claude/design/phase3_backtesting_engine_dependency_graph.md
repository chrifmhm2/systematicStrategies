# Phase 3 — Backtesting Engine Dependency Graph

> `backend/core/backtester/` · `backend/core/data/yahoo.py` · Python 3.12
> Generated after session 3 (97/97 tests passing)

---

## Legend

```
──────►   uses / imports / depends on
══════►   returns / produces
   ▲      inherits from (is-a / subclass)
───┤      composes (has-a, stores as field)
TYPE      import only at type-check time (TYPE_CHECKING guard — not at runtime)
[ ]       concrete class or dataclass
( )       abstract class / interface (ABC)
{ }       free function
⬡         external library (outside core/)
⚠         circular import — resolved by TYPE_CHECKING guard or local import
```

---

## High-Level Overview — Three Packages, One Simulation Loop

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║   core/backtester/              core/data/ (extended)    core/models/ (updated) ║
║   ────────────────              ──────────────────────   ──────────────────────  ║
║   Simulation engine             Real market data         Result container        ║
║                                                                                  ║
║   [BacktestConfig]              [YahooDataProvider]      [BacktestResult]        ║
║   [BacktestEngine]                    ▲                  (pd.Series + df)        ║
║   [TransactionCostModel]              │ inherits                                 ║
║   (RebalancingOracle)           (IDataProvider)  ← Phase 1                      ║
║        ▲         ▲                                                               ║
║        │         │                                                               ║
║   [Periodic-]  [Threshold-]                                                      ║
║   [Rebalancing] [Rebalancing]                                                    ║
║                                                                                  ║
║   Cross-phase dependencies (what Phase 3 consumes from Phase 1 & 2):            ║
║     BacktestEngine ──────► IStrategy.compute_weights()        (Phase 2)          ║
║     BacktestEngine ──────► IDataProvider.get_prices()         (Phase 1)          ║
║     BacktestEngine ──────► Portfolio.total_value()            (Phase 1)          ║
║     BacktestEngine ══════► BacktestResult                     (Phase 1, updated) ║
║     ThresholdRebalancing ─► Portfolio  (local import)         (Phase 1)          ║
║     YahooDataProvider  ───► IDataProvider                     (Phase 1)          ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## Module 1 — `core/backtester/engine.py`

```
  External
  ────────
  stdlib: date, time          pandas         Phase 1          Phase 2
       │                        │               │                │
       ▼                        ▼               ▼                ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │  engine.py                                                                 │
  │                                                                            │
  │  ┌──────────────────────────────────────────────────────────────────┐    │
  │  │  [BacktestConfig]   (dataclass)                                   │    │
  │  │──────────────────────────────────────────────────────────────────│    │
  │  │ initial_value          : float = 100_000.0                        │    │
  │  │ start_date             : date | None = None                       │    │
  │  │ end_date               : date | None = None                       │    │
  │  │ symbols                : list[str] = []                           │    │
  │  │ rebalancing_frequency  : str = "weekly"                           │    │
  │  │ transaction_cost_bps   : float = 10.0                             │    │
  │  │ slippage_bps           : float = 5.0                              │    │
  │  │ threshold              : float = 0.05                             │    │
  │  │                                                                   │    │
  │  │  Consumed by: BacktestEngine.__init__()                           │    │
  │  │  Referenced by: BacktestResult.config (TYPE_CHECKING)            │    │
  │  └──────────────────────────────────────────────────────────────────┘    │
  │                       │ consumed by                                       │
  │                       ▼                                                   │
  │  ┌──────────────────────────────────────────────────────────────────┐    │
  │  │  [BacktestEngine]                                                 │    │
  │  │──────────────────────────────────────────────────────────────────│    │
  │  │                                                                   │    │
  │  │  __init__(config: BacktestConfig)                                 │    │
  │  │      self.config = config                                         │    │
  │  │                                                                   │    │
  │  │  run(strategy: IStrategy,                ← Phase 2 contract      │    │
  │  │      data_provider: IDataProvider)       ← Phase 1 contract      │    │
  │  │   │                                                               │    │
  │  │   ├─ 1. data_provider.get_prices()    ──────────────────────────►│    │
  │  │   ├─ 2. data_provider.get_risk_free_rate()  ──────────────────── │    │
  │  │   ├─ 3. _build_oracle(cfg)  ══════════► RebalancingOracle        │    │
  │  │   ├─ 4. TransactionCostModel(commission, slippage)  ─────────── ►│    │
  │  │   ├─ 5. Portfolio(positions={}, cash=initial_value) ─────────── ►│ P1 │
  │  │   │                                                               │    │
  │  │   └─ 6. Loop for t in prices.index:                              │    │
  │  │         ├─ pos.price = current_prices[sym]  (mark-to-market)     │    │
  │  │         ├─ cash *= (1 + rfr × 1/252)        (interest accrual)   │    │
  │  │         ├─ V_t = portfolio.total_value()  ─────────────────────► │ P1 │
  │  │         ├─ Guard: len(history) < required_history_days → skip    │    │
  │  │         ├─ oracle.should_rebalance(t, portfolio) → bool          │    │
  │  │         ├─ pw = strategy.compute_weights(t, prices.loc[:t], …)  ►│ P2 │
  │  │         ├─ For each symbol:                                       │    │
  │  │         │    new_qty = pw.weights[sym] × V_t / price             │    │
  │  │         │    cost = cost_model.compute_cost(delta × price)       │    │
  │  │         ├─ new_cash = V_t − equity_deployed − total_costs        │    │
  │  │         ├─ assert |reconstructed − (V_t−TC)| < 1e-6  (invariant) │    │
  │  │         └─ oracle.update_target_weights(pw.weights) [threshold]  │    │
  │  │                                                                   │    │
  │  │  ══════════════════════════════════════════════════► BacktestResult    │
  │  │                                                                   │    │
  │  │  @staticmethod                                                    │    │
  │  │  _build_oracle(cfg) → RebalancingOracle                          │    │
  │  │      "threshold" → ThresholdRebalancing(threshold)               │    │
  │  │      else        → PeriodicRebalancing(frequency)                │    │
  │  │                                                                   │    │
  │  │  @staticmethod                                                    │    │
  │  │  _compute_benchmark(prices, symbols, initial_value) → pd.Series  │    │
  │  │      qty_i = (initial_value / N) / P_i(t₀)                      │    │
  │  │      benchmark = Σᵢ qty_i × P_i(t)  for each t                  │    │
  │  └──────────────────────────────────────────────────────────────────┘    │
  └───────────────────────────────────────────────────────────────────────────┘
```

---

## Module 2 — `core/backtester/costs.py`

```
  ┌───────────────────────────────────────────────────────────────────────────┐
  │  costs.py                                                                  │
  │                                                                            │
  │  ┌─────────────────────────────────────────────────────────────────────┐  │
  │  │  [TransactionCostModel]                                              │  │
  │  │─────────────────────────────────────────────────────────────────────│  │
  │  │                                                                      │  │
  │  │  commission_bps  : float = 10.0   broker fee in basis points        │  │
  │  │  slippage_bps    : float = 5.0    bid-ask + market impact           │  │
  │  │  min_commission  : float = 1.0    floor per trade in currency units  │  │
  │  │                                                                      │  │
  │  │  compute_cost(trade_value: float) → float                           │  │
  │  │    cost = |trade_value| × (commission_bps + slippage_bps) / 10_000  │  │
  │  │    return max(cost, min_commission)                                  │  │
  │  │                                                                      │  │
  │  │  Examples:                                                           │  │
  │  │    compute_cost(10_000)  → 15.0   (15 bps × $10k)                   │  │
  │  │    compute_cost(1.0)     → 1.0    (0.0015 floored to min_commission) │  │
  │  │                                                                      │  │
  │  │  Pure computation — no state, no external deps, no imports           │  │
  │  │  Used by: BacktestEngine (one instance per run)                      │  │
  │  └─────────────────────────────────────────────────────────────────────┘  │
  └───────────────────────────────────────────────────────────────────────────┘
```

---

## Module 3 — `core/backtester/rebalancing.py`

```
  ┌───────────────────────────────────────────────────────────────────────────┐
  │  rebalancing.py                                                            │
  │                                                                            │
  │  base.py                                                                   │
  │  ┌─────────────────────────────────────────────────────────────────────┐  │
  │  │  (RebalancingOracle)   — Abstract Base Class (ABC)                  │  │
  │  │─────────────────────────────────────────────────────────────────────│  │
  │  │  @abstractmethod                                                     │  │
  │  │  should_rebalance(current_date: date, portfolio: object) → bool      │  │
  │  │      called every trading day by BacktestEngine                     │  │
  │  │      returns True  → trigger strategy.compute_weights() + trades    │  │
  │  │      returns False → record NAV only, no trading                    │  │
  │  └──────────────────────────────────┬──────────────────────────────────┘  │
  │                                     │ inherits                             │
  │               ┌─────────────────────┴────────────────────┐                │
  │               ▼                                           ▼                │
  │  ┌────────────────────────────────┐  ┌───────────────────────────────┐    │
  │  │  [PeriodicRebalancing]         │  │  [ThresholdRebalancing]       │    │
  │  │────────────────────────────────│  │───────────────────────────────│    │
  │  │                                │  │                               │    │
  │  │  _frequency: str               │  │  _threshold: float            │    │
  │  │    "daily" | "weekly" |        │  │  _last_target_weights:        │    │
  │  │    "monthly"                   │  │    dict[str, float] = {}      │    │
  │  │  _last_rebalance_date:         │  │                               │    │
  │  │    date | None = None          │  │  should_rebalance():          │    │
  │  │                                │  │   1. {} → True  (first call)  │    │
  │  │  __init__(frequency: str):     │  │   2. no Portfolio → True      │    │
  │  │    raises ValueError if        │  │   3. prices = {sym: pos.price}│    │
  │  │    frequency not in            │  │   4. total = portfolio.       │    │
  │  │    {"daily","weekly","monthly"}│  │        total_value(prices)   │    │
  │  │                                │  │   5. for sym in targets:      │    │
  │  │  should_rebalance():           │  │     curr_w = qty×P / total   │    │
  │  │    daily   → always True       │  │     if |curr-target|>thresh  │    │
  │  │    weekly  → weekday() == 0    │  │       → True                 │    │
  │  │             (Monday)           │  │   6. else → False             │    │
  │  │    monthly → new calendar month│  │                               │    │
  │  │    updates _last_rebalance_date│  │  update_target_weights(       │    │
  │  │    when returning True         │  │    weights: dict[str,float])  │    │
  │  │                                │  │    stores copy of new weights  │    │
  │  │  Stateful: carries memory of   │  │    called by engine after     │    │
  │  │  last rebalancing across calls │  │    every rebalancing step     │    │
  │  └────────────────────────────────┘  └───────────────────────────────┘    │
  │                                                   │                        │
  │  ⚠ local import inside should_rebalance():       │                        │
  │    from core.models.portfolio import Portfolio    │                        │
  │    (avoids circular dep at module load time)  ────┘                        │
  └───────────────────────────────────────────────────────────────────────────┘
```

---

## Module 4 — `core/data/yahoo.py`

```
  ┌───────────────────────────────────────────────────────────────────────────┐
  │  core/data/yahoo.py                        External: yfinance ⬡           │
  │                                                          │                 │
  │  ┌─────────────────────────────────────────────────────────────────────┐  │
  │  │  [YahooDataProvider]   (IDataProvider)   ← inherits from Phase 1    │  │
  │  │─────────────────────────────────────────────────────────────────────│  │
  │  │                                                                      │  │
  │  │  __init__(cache: bool = True)                                        │  │
  │  │      _cache: dict[tuple, pd.DataFrame] = {}                          │  │
  │  │      _use_cache: bool                                                │  │
  │  │                                                                      │  │
  │  │  get_prices(symbols, start_date, end_date) → pd.DataFrame           │  │
  │  │   │                                                                  │  │
  │  │   ├─ 1. key = (tuple(sorted(symbols)), start_date, end_date)        │  │
  │  │   ├─ 2. cache hit → return cached result                             │  │
  │  │   ├─ 3. yf.download(symbols, auto_adjust=True, progress=False) ─── ►⬡│  │
  │  │   ├─ 4. if MultiIndex columns → raw["Close"]    (multi-ticker)       │  │
  │  │   │     else                  → raw[["Close"]]  (single ticker)      │  │
  │  │   ├─ 5. enforce column order: df = df[symbols]                       │  │
  │  │   ├─ 6. df.ffill()   ← forward-fill weekends / holidays             │  │
  │  │   └─ 7. store in cache, return df                                    │  │
  │  │                                                                      │  │
  │  │  get_risk_free_rate(as_of_date: date) → float                       │  │
  │  │   │                                                                  │  │
  │  │   ├─ yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1] / 100 │  │
  │  │   └─ except any Exception → return 0.05  (5% fallback)              │  │
  │  │                                                                      │  │
  │  │  Cache key design:                                                   │  │
  │  │    sorted(symbols) → same result regardless of input order          │  │
  │  │    (tuple(...), start_date, end_date) → hashable for dict key       │  │
  │  └─────────────────────────────────────────────────────────────────────┘  │
  │                                                                            │
  │  DEFAULT_UNIVERSE = 20 large-cap tickers (AAPL, MSFT, GOOGL, …)           │
  │    Convenience constant — not used by the class itself                     │
  └───────────────────────────────────────────────────────────────────────────┘
```

---

## Module 5 — `core/models/results.py` (updated)

```
  ┌───────────────────────────────────────────────────────────────────────────┐
  │  core/models/results.py                                                    │
  │                                                                            │
  │  ⚠ Circular import problem:                                                │
  │    results.py  needs  BacktestConfig  (type annotation)                   │
  │    engine.py   needs  BacktestResult  (return type)                       │
  │                                                                            │
  │  ✓ Resolution: TYPE_CHECKING guard + from __future__ import annotations   │
  │                                                                            │
  │  from __future__ import annotations   ← all annotations become strings    │
  │  if TYPE_CHECKING:                    ← False at runtime, True for mypy   │
  │      from core.backtester.engine import BacktestConfig                    │
  │                                                                            │
  │  ┌─────────────────────────────────────────────────────────────────────┐  │
  │  │  [BacktestResult]   (dataclass)                                      │  │
  │  │─────────────────────────────────────────────────────────────────────│  │
  │  │                                                                      │  │
  │  │  portfolio_values  : pd.Series    DatetimeIndex → float (NAV)        │  │
  │  │                      field(default_factory=pd.Series)                │  │
  │  │                                                                      │  │
  │  │  benchmark_values  : pd.Series | None = None                        │  │
  │  │                      DatetimeIndex → float (benchmark NAV)           │  │
  │  │                                                                      │  │
  │  │  weights_history   : pd.DataFrame                                    │  │
  │  │                      field(default_factory=pd.DataFrame)             │  │
  │  │                      rows = rebalancing dates, cols = symbols        │  │
  │  │                                                                      │  │
  │  │  trades_log        : list[dict]                                      │  │
  │  │                      field(default_factory=list)                     │  │
  │  │                      each dict: {date, symbol, quantity, price, cost}│  │
  │  │                                                                      │  │
  │  │  risk_metrics      : dict     = {}                                   │  │
  │  │                      populated by Phase 4 — empty in Phase 3        │  │
  │  │                                                                      │  │
  │  │  config            : BacktestConfig | None = None                    │  │
  │  │                      TYPE_CHECKING import — annotation only          │  │
  │  │                                                                      │  │
  │  │  strategy_name     : str = ""                                        │  │
  │  │  computation_time_ms: float = 0.0                                    │  │
  │  │                                                                      │  │
  │  │  Produced by: BacktestEngine.run()                                   │  │
  │  │  Consumed by: API layer (Phase 5), risk module (Phase 4)             │  │
  │  └─────────────────────────────────────────────────────────────────────┘  │
  └───────────────────────────────────────────────────────────────────────────┘
```

---

## Inheritance Tree

```
  (RebalancingOracle)   ← abstract, in rebalancing.py
       │
       ├── [PeriodicRebalancing]    — state: _last_rebalance_date
       └── [ThresholdRebalancing]   — state: _last_target_weights
                                      local import: Portfolio (P1)

  (IDataProvider)       ← abstract, in core/data/base.py  (Phase 1)
       │
       ├── [SimulatedDataProvider]  (Phase 1)
       ├── [CsvDataProvider]        (Phase 1)
       └── [YahooDataProvider]      (Phase 3 ← new)
```

---

## Circular Import Resolution Map

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                          │
  │   results.py ─────────────────────────────────────────────► engine.py  │
  │   (needs BacktestConfig type)          runtime: NOT imported            │
  │                                        type-check: imported via guard   │
  │                                                                          │
  │   engine.py  ─────────────────────────────────────────────► results.py │
  │   (needs BacktestResult)               always imported (return type)    │
  │                                                                          │
  │  Solution in results.py:                                                 │
  │                                                                          │
  │    from __future__ import annotations  ← (1) annotations = lazy strings │
  │    from typing import TYPE_CHECKING                                       │
  │                                                                          │
  │    if TYPE_CHECKING:                   ← (2) False at runtime           │
  │        from core.backtester.engine import BacktestConfig                │
  │                                                                          │
  │    @dataclass                                                            │
  │    class BacktestResult:                                                 │
  │        config: BacktestConfig | None = None  ← (3) just the string      │
  │                                                  "BacktestConfig|None"   │
  │                                                  at runtime             │
  │                                                                          │
  │  ─────────────────────────────────────────────────────────────────────  │
  │                                                                          │
  │   rebalancing.py ──────────────────────────────────────► portfolio.py  │
  │   (ThresholdRebalancing needs Portfolio)                                 │
  │                                                                          │
  │  Solution in rebalancing.py:                                             │
  │    def should_rebalance(self, current_date, portfolio):                 │
  │        from core.models.portfolio import Portfolio  ← local import      │
  │        if not isinstance(portfolio, Portfolio): ...                     │
  │        (import deferred to function call — no module-load cycle)        │
  │                                                                          │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## Full Cross-Phase Dependency Map

```
  Phase 1                           Phase 2                  Phase 3
  ───────                           ───────                  ───────

  core/models/portfolio.py
  ┌────────────────────┐
  │  [Portfolio]       │◄──────────────────────────── BacktestEngine (creates)
  │  total_value()     │◄──────────────────────────── BacktestEngine.run() loop
  │  positions: dict   │◄──────────────────────────── ThresholdRebalancing (local)
  └────────────────────┘

  core/models/results.py
  ┌────────────────────┐
  │  [BacktestResult]  │◄══════════════════════════════ BacktestEngine.run()
  │  (stub → filled)   │        returns this object
  └────────────────────┘
  ▲
  │ TYPE_CHECKING only (not at runtime)
  └─────────────────────────────────── BacktestConfig (engine.py)

  core/data/base.py
  ┌────────────────────┐
  │  (IDataProvider)   │◄──────────────────────────── BacktestEngine.run()
  │  get_prices()      │     injected as argument
  │  get_risk_free()   │◄─────────────────────────── YahooDataProvider (P3 new)
  └────────────────────┘

  core/strategies/base.py         (Phase 2)
  ┌────────────────────┐
  │  (IStrategy)       │◄──────────────────────────── BacktestEngine.run()
  │  compute_weights() │     injected as argument
  │  required_history  │
  └────────────────────┘
  ┌────────────────────┐
  │  [PortfolioWeights]│◄══════════════════════════ strategy.compute_weights()
  │  weights: dict     │      BacktestEngine reads pw.weights
  └────────────────────┘

  Phase 3 → Phase 1 and 2 (many):
    BacktestEngine  ──────────────►  IDataProvider    (P1)
    BacktestEngine  ──────────────►  IStrategy        (P2)
    BacktestEngine  ──────────────►  Portfolio        (P1)
    BacktestEngine  ══════════════►  BacktestResult   (P1 updated)
    ThresholdRebalancing  ─────────► Portfolio        (P1)
    YahooDataProvider  ────────────► IDataProvider    (P1)

  Phase 1 and 2 → Phase 3: NONE
  (Phase 1 and 2 code has zero knowledge of the backtester — correct layering)
```

---

## Data Flow Through a Full Backtest Run

```
  User / test
       │
       │  provider = SimulatedDataProvider(...)  or  YahooDataProvider()
       │  strategy = EqualWeightStrategy(config)   (any IStrategy)
       │  config   = BacktestConfig(initial_value=100_000, symbols=["AAPL","MSFT"], ...)
       │  engine   = BacktestEngine(config)
       │  result   = engine.run(strategy, provider)
       │
       ▼
  BacktestEngine.run()
       │
       ├─ prices = provider.get_prices(symbols, start, end)  ─────► pd.DataFrame
       │                                                      rows = trading days
       │                                                      cols = symbols
       │
       ├─ oracle = _build_oracle(cfg)   ─────────────────────► PeriodicRebalancing
       │                                                        or ThresholdRebalancing
       │
       ├─ cost_model = TransactionCostModel(10, 5)
       │
       ├─ portfolio = Portfolio(positions={}, cash=100_000)
       │
       └─ for t in prices.index:
               │
               ├─ [mark to market]  pos.price = current_prices[sym]
               ├─ [interest]        cash *= (1 + rfr/252)
               ├─ [NAV]             V_t = portfolio.total_value()
               │                   portfolio_values[t] = V_t
               │
               ├─ [guard]           len(history) < required_history → skip
               ├─ [oracle]          oracle.should_rebalance() → skip if False
               │
               ├─ [weights]         pw = strategy.compute_weights(t, prices.loc[:t])
               │                         │
               │                         └── returns PortfolioWeights.weights
               │
               ├─ [trade sizing]    new_qty = pw.weights[sym] × V_t / price
               ├─ [costs]          cost = cost_model.compute_cost(delta × price)
               ├─ [self-financing] new_cash = V_t − equity − total_costs
               ├─ [assert]         |equity + new_cash − (V_t − TC)| < 1e-6
               ├─ [update]         portfolio.positions = new_positions
               └─ [threshold]      oracle.update_target_weights(pw.weights)

       ══════════════════════════════════════════════════════════════► BacktestResult
               portfolio_values  = pd.Series(portfolio_values_dict)
               benchmark_values  = _compute_benchmark(prices, symbols, 100_000)
               weights_history   = pd.DataFrame(weights_records).T
               trades_log        = list of {date, symbol, qty, price, cost}
               risk_metrics      = {}  ← Phase 4 fills this
               computation_time_ms = elapsed
```

---

## Oracle State Machine

```
  PeriodicRebalancing("weekly"):

    state: _last_rebalance_date = None
    ┌──────────────────────────────────────────────────────────┐
    │  Input: current_date                                      │
    │                                                           │
    │  weekday()==0 ?                                           │
    │   YES → _last_rebalance_date = current_date              │
    │          return True  ────────────────────► rebalance    │
    │   NO  → return False  ────────────────────► skip         │
    └──────────────────────────────────────────────────────────┘

  ThresholdRebalancing(threshold=0.05):

    state: _last_target_weights = {}
    ┌──────────────────────────────────────────────────────────┐
    │  Input: current_date, portfolio                          │
    │                                                           │
    │  _last_target_weights empty?                             │
    │   YES → return True  ─────────────────────► rebalance   │
    │   NO  → compute current weights from positions          │
    │          any |current_w − target_w| > threshold?         │
    │           YES → return True  ────────────────► rebalance │
    │           NO  → return False ────────────────► skip      │
    │                                                           │
    │  After engine rebalances:                                 │
    │    oracle.update_target_weights(pw.weights) called       │
    │    _last_target_weights ← pw.weights.copy()              │
    └──────────────────────────────────────────────────────────┘
```

---

## Dependency Count Summary

| Class | Depends On (internal) | Depended On By |
|---|---|---|
| `TransactionCostModel` | — (pure math) | `BacktestEngine` |
| `RebalancingOracle` | stdlib `date` | `PeriodicRebalancing`, `ThresholdRebalancing`, `BacktestEngine` |
| `PeriodicRebalancing` | `RebalancingOracle` | `BacktestEngine._build_oracle()` |
| `ThresholdRebalancing` | `RebalancingOracle`, `Portfolio` (local import P1) | `BacktestEngine._build_oracle()` |
| `BacktestConfig` | stdlib `date` | `BacktestEngine`, `BacktestResult` (TYPE_CHECKING) |
| `BacktestEngine` | `BacktestConfig`, `TransactionCostModel`, `RebalancingOracle` subtypes, **`IDataProvider` (P1)**, **`IStrategy` (P2)**, **`Portfolio` (P1)**, **`BacktestResult` (P1/P3)**, **`PortfolioWeights` (P2)** | tests, API (Phase 5) |
| `BacktestResult` | `pd.Series`, `pd.DataFrame`, `BacktestConfig` (TYPE only) | `BacktestEngine` (produces), API (P5), risk module (P4) |
| `YahooDataProvider` | **`IDataProvider` (P1)**, yfinance ⬡ | tests, `BacktestEngine` (injected) |

---

## What This Means for Phase 4

Phase 4 adds `core/risk/`. It will:

1. **Read** from `BacktestResult.portfolio_values` (pd.Series) — already filled by Phase 3
2. **Compute** `PerformanceMetrics`, `VaRCalculator`, `GreeksCalculator`
3. **Write** results back into `BacktestResult.risk_metrics` (the dict left empty in Phase 3)
4. **Use** `BlackScholesModel` from Phase 1 for Greeks surface

```
  Phase 3                        Phase 4
  ───────                        ───────
  BacktestResult
  ├─ portfolio_values (filled) ──────────────►  PerformanceMetrics
  ├─ benchmark_values (filled) ──────────────►  PerformanceMetrics (vs benchmark)
  ├─ trades_log       (filled) ──────────────►  turnover stats
  └─ risk_metrics     = {}     ◄══════════════  VaRCalculator, GreeksCalculator
                                 Phase 4 fills this after computing

  Phase 1
  ───────
  BlackScholesModel ──────────────────────────► GreeksCalculator (Phase 4)
```

No changes needed to any Phase 1, 2, or 3 code.
