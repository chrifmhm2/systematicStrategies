# Phase 2 — Strategy Framework Dependency Graph

> `backend/core/strategies/` · Python 3.10
> Generated after session 2 (75/75 tests passing)

---

## Legend

```
──────►   uses / imports / depends on
══════►   returns / produces
   ▲      inherits from (is-a / subclass)
───┤      composes (has-a, stores as field)
@reg      registered via @StrategyRegistry.register (at import time)
[ ]       concrete class or dataclass
( )       abstract class / interface (ABC)
{ }       free function
⬡         external library (outside core/)
```

---

## High-Level Overview

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        core/strategies/  —  Three Layers                        ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  LAYER 1 — Base contracts  (base.py)                                             ║
║  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐  ║
║  │ [StrategyConfig] │  │[PortfolioWeights] │  │       (IStrategy)            │  ║
║  │  name            │  │  weights: dict   │  │  abstract contract for all   │  ║
║  │  description     │  │  cash_weight     │  │  8 concrete strategies       │  ║
║  │  rebal_frequency │  │  timestamp       │  └──────────────────────────────┘  ║
║  │  cost_bps        │  │  total_weight()  │                   ▲                 ║
║  └──────────────────┘  └──────────────────┘                   │ inherits        ║
║           │ used by IStrategy                  ╔══════════════╩════════════╗    ║
║           └────────────────────────────────────║  8 concrete strategies    ║    ║
║                                                ╚═══════════════════════════╝    ║
║  LAYER 2 — Plugin system  (registry.py)                                          ║
║  ┌──────────────────────────────────────────────────────────────────────────┐   ║
║  │  [StrategyRegistry]   global catalog, one shared dict for whole program  │   ║
║  │  _strategies: dict[str, type[IStrategy]]                                 │   ║
║  │  register() · list_strategies() · create()                               │   ║
║  └──────────────────────────────────────────────────────────────────────────┘   ║
║                           ▲  every concrete strategy self-registers via @reg     ║
║                                                                                  ║
║  LAYER 3 — Concrete strategies  (hedging / allocation / signal)                  ║
║  ┌─────────────────┐  ┌───────────────────────────────┐  ┌──────────────────┐  ║
║  │    hedging/     │  │        allocation/            │  │    signal/       │  ║
║  │  DeltaHedge     │  │  EqualWeight  MinVariance     │  │  Momentum        │  ║
║  │  DeltaGamma     │  │  MaxSharpe    RiskParity      │  │  MeanReversion   │  ║
║  └─────────────────┘  └───────────────────────────────┘  └──────────────────┘  ║
║         │ cross-phase dependency                                                  ║
║         └──────► MonteCarloPricer  (Phase 1 — core/pricing/)                    ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## Layer 1 — Base Contracts (`core/strategies/base.py`)

```
  External
  ────────
  stdlib: date          pandas: pd.DataFrame
      │                        │
      ▼                        ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │  base.py                                                                   │
  │                                                                            │
  │  ┌──────────────────────────────────┐                                     │
  │  │  [StrategyConfig]  (dataclass)   │                                     │
  │  │──────────────────────────────────│                                     │
  │  │ name                : str        │                                     │
  │  │ description         : str        │                                     │
  │  │ rebalancing_frequency: str="weekly"                                    │
  │  │ transaction_cost_bps : float=10.0│                                     │
  │  └──────────────────────────────────┘                                     │
  │                 │ consumed by IStrategy.__init__                           │
  │                 ▼                                                          │
  │  ┌──────────────────────────────────────────────────────────────────┐    │
  │  │  (IStrategy)   — Abstract Base Class                              │    │
  │  │──────────────────────────────────────────────────────────────────│    │
  │  │ __init__(config: StrategyConfig)                                  │    │
  │  │    └── self.config = config                                       │    │
  │  │                                                                   │    │
  │  │ @abstractmethod                                                   │    │
  │  │ compute_weights(current_date: date,                               │    │
  │  │                 price_history: pd.DataFrame,   ← no look-ahead   │    │
  │  │                 current_portfolio: object|None)                   │    │
  │  │    ══════════════════════════════════════════► PortfolioWeights   │    │
  │  │                                                                   │    │
  │  │ @property @abstractmethod                                         │    │
  │  │ required_history_days → int                                       │    │
  │  │    └── backtester reads this to know when strategy can trade      │    │
  │  │                                                                   │    │
  │  │ @classmethod @abstractmethod                                      │    │
  │  │ get_param_schema() → dict                                         │    │
  │  │    └── frontend reads this to render the config form              │    │
  │  └──────────────────────────────────────────────────────────────────┘    │
  │                                                    ║                      │
  │                                                    ║ returns              │
  │                                                    ▼                      │
  │  ┌───────────────────────────────────────────────────────────────────┐   │
  │  │  [PortfolioWeights]  (dataclass)                                   │   │
  │  │───────────────────────────────────────────────────────────────────│   │
  │  │ weights    : dict[str, float]   symbol → weight (neg = short)     │   │
  │  │ cash_weight: float              uninvested fraction                │   │
  │  │ timestamp  : date               when this allocation was computed  │   │
  │  │                                                                    │   │
  │  │ total_weight() → float                                             │   │
  │  │    return sum(weights.values()) + cash_weight                      │   │
  │  │    must be ≤ 1.0  (< 1 for hedging strategies)                     │   │
  │  └───────────────────────────────────────────────────────────────────┘   │
  └───────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 2 — Plugin System (`core/strategies/registry.py`)

```
  ┌───────────────────────────────────────────────────────────────────────────┐
  │  registry.py                                                               │
  │                                                                            │
  │  ┌─────────────────────────────────────────────────────────────────────┐  │
  │  │  [StrategyRegistry]   class-level state — one dict for the process  │  │
  │  │─────────────────────────────────────────────────────────────────────│  │
  │  │                                                                      │  │
  │  │  _strategies: dict[str, type[IStrategy]] = {}                        │  │
  │  │       key   = class name string  (e.g. "MomentumStrategy")           │  │
  │  │       value = the class itself   (not an instance)                   │  │
  │  │                                                                      │  │
  │  │  @classmethod                                                        │  │
  │  │  register(strategy_class) → strategy_class        ← decorator       │  │
  │  │     _strategies[class.__name__] = class                              │  │
  │  │     fires at import time, returns class unchanged                    │  │
  │  │                                                                      │  │
  │  │  @classmethod                                                        │  │
  │  │  list_strategies() → list[dict]               ← API / frontend      │  │
  │  │     reads DESCRIPTION, FAMILY, get_param_schema() from each class    │  │
  │  │     returns [{name, description, family, param_schema}, ...]         │  │
  │  │                                                                      │  │
  │  │  @classmethod                                                        │  │
  │  │  create(name: str, config: dict) → IStrategy  ← backtester / API    │  │
  │  │     1. look up class by name                                         │  │
  │  │     2. build StrategyConfig from config dict                         │  │
  │  │     3. instantiate class(strategy_config, **extra_kwargs)            │  │
  │  │     raises KeyError with available list if name not found            │  │
  │  └─────────────────────────────────────────────────────────────────────┘  │
  │                                                                            │
  │  Registration flow (happens automatically at import time):                 │
  │                                                                            │
  │  Python loads core/strategies/__init__.py                                  │
  │       └── import core.strategies.hedging                                   │
  │               └── import delta_hedge.py   → @register fires               │
  │               └── import delta_gamma_hedge.py → @register fires           │
  │       └── import core.strategies.allocation                                │
  │               └── import equal_weight.py  → @register fires               │
  │               └── ...                                                      │
  │       └── import core.strategies.signal                                    │
  │               └── import momentum.py      → @register fires               │
  │               └── ...                                                      │
  │                                                                            │
  │  Result: _strategies has all 8 strategies before any code runs             │
  └───────────────────────────────────────────────────────────────────────────┘

                    @register ◄──────────────────────────────────────────────────┐
                    @register ◄──────────────────────────────┐                   │
                    @register ◄───────────────┐              │                   │
                                              │              │                   │
                                         signal/        allocation/          hedging/
```

---

## Layer 3 — Concrete Strategies

### Hedging Family (`core/strategies/hedging/`)

```
  ┌───────────────────────────────────────────────────────────────────────────┐
  │  hedging/                                                                  │
  │                                                                            │
  │  delta_hedge.py                                                            │
  │  ┌─────────────────────────────────────────────────────────────────────┐  │
  │  │  @reg  [DeltaHedgeStrategy]  (IStrategy)                             │  │
  │  │─────────────────────────────────────────────────────────────────────│  │
  │  │  FAMILY = "hedging"                                                  │  │
  │  │                                                                      │  │
  │  │  Constructor params (beyond config):                                 │  │
  │  │    option_weights  : list[float] | None   basket weights ωᵢ          │  │
  │  │    strike          : float = 100.0        K                          │  │
  │  │    maturity_years  : float = 1.0          T                          │  │
  │  │    n_simulations   : int   = 50_000       Monte Carlo paths          │  │
  │  │    risk_free_rate  : float = 0.05         r                          │  │
  │  │    volatilities    : list[float] | None   σᵢ per asset               │  │
  │  │    correlation     : list[list] | None    ρ matrix                   │  │
  │  │                                                                      │  │
  │  │  required_history_days → 1   (only needs today's price)              │  │
  │  │                                                                      │  │
  │  │  compute_weights():                                                   │  │
  │  │    spots = price_history.iloc[-1]                                    │  │
  │  │    pricer = MonteCarloPricer(n_simulations, seed=42)                 │  │
  │  │    deltas = pricer.compute_deltas(spots, ...)  ─────────────────────►│  │
  │  │                                              (Phase 1 cross-dep)     │  │
  │  │    wᵢ = Δᵢ · Sᵢ / Σ(Δⱼ·Sⱼ)                                        │  │
  │  │    cash = max(0, 1 - Σwᵢ)                                            │  │
  │  │    ══════════════════════════════════════════► PortfolioWeights       │  │
  │  └─────────────────────────────────────────────────────────────────────┘  │
  │                │ composed by                                                │
  │                ▼                                                            │
  │  delta_gamma_hedge.py                                                      │
  │  ┌─────────────────────────────────────────────────────────────────────┐  │
  │  │  @reg  [DeltaGammaHedgeStrategy]  (IStrategy)  — stub               │  │
  │  │─────────────────────────────────────────────────────────────────────│  │
  │  │  FAMILY = "hedging"                                                  │  │
  │  │                                                                      │  │
  │  │  _delta_hedge : DeltaHedgeStrategy  ◄── composes (has-a)            │  │
  │  │                                                                      │  │
  │  │  compute_weights():                                                   │  │
  │  │    return self._delta_hedge.compute_weights(...)  ← delegates 100%   │  │
  │  │                                                                      │  │
  │  │  get_param_schema():                                                  │  │
  │  │    return DeltaHedgeStrategy.get_param_schema()  ← reuses schema     │  │
  │  │                                                                      │  │
  │  │  Planned: neutralise Γ using a second vanilla option instrument       │  │
  │  └─────────────────────────────────────────────────────────────────────┘  │
  └───────────────────────────────────────────────────────────────────────────┘

  Cross-phase dependency:
  DeltaHedgeStrategy  ──────►  MonteCarloPricer          (core/pricing/)
                      ══════►  PricingResult.deltas       (core/models/)
```

### Allocation Family (`core/strategies/allocation/`)

```
  ┌───────────────────────────────────────────────────────────────────────────┐
  │  allocation/          External: scipy.optimize.minimize (SLSQP)            │
  │                                                ⬡                           │
  │  equal_weight.py                               │                           │
  │  ┌─────────────────────────────────────┐       │                           │
  │  │  @reg  [EqualWeightStrategy]         │       │                           │
  │  │─────────────────────────────────────│       │                           │
  │  │  required_history_days → 1           │       │                           │
  │  │  compute_weights():                  │       │                           │
  │  │    wᵢ = 1/n  for all i               │       │                           │
  │  │    cash = 0                          │       │                           │
  │  │  No scipy — pure arithmetic          │       │                           │
  │  └─────────────────────────────────────┘       │                           │
  │                                                │                           │
  │  min_variance.py                               │                           │
  │  ┌─────────────────────────────────────┐       │                           │
  │  │  @reg  [MinVarianceStrategy]         │       │                           │
  │  │─────────────────────────────────────│       │                           │
  │  │  lookback_window: int = 60           │       │                           │
  │  │  required_history_days → lookback+1  │       │                           │
  │  │  compute_weights():                  │       │                           │
  │  │    returns = prices.pct_change()     │       │                           │
  │  │    Σ = returns.cov()                 │       │                           │
  │  │    min ωᵀΣω  s.t. Σω=1, ω≥0  ───────┼───────┘                           │
  │  │    (SLSQP via scipy.optimize)        │                                   │
  │  └─────────────────────────────────────┘                                   │
  │                                                                            │
  │  max_sharpe.py                                                             │
  │  ┌─────────────────────────────────────┐                                   │
  │  │  @reg  [MaxSharpeStrategy]           │                                   │
  │  │─────────────────────────────────────│                                   │
  │  │  lookback_window: int = 60           │                                   │
  │  │  risk_free_rate_override: float|None │                                   │
  │  │  required_history_days → lookback+1  │                                   │
  │  │  compute_weights():                  │                                   │
  │  │    μ = returns.mean() × 252          │                                   │
  │  │    Σ = returns.cov()  × 252          │                                   │
  │  │    max (ωᵀμ−rf)/√(ωᵀΣω)  ───────────┼──► scipy SLSQP                    │
  │  │    s.t. Σω=1, ω≥0                    │                                   │
  │  └─────────────────────────────────────┘                                   │
  │                                                                            │
  │  risk_parity.py                                                            │
  │  ┌─────────────────────────────────────┐                                   │
  │  │  @reg  [RiskParityStrategy]          │                                   │
  │  │─────────────────────────────────────│                                   │
  │  │  lookback_window: int = 60           │                                   │
  │  │  required_history_days → lookback+1  │                                   │
  │  │  compute_weights():                  │                                   │
  │  │    RC_i = ωᵢ·(Σω)ᵢ / ωᵀΣω           │                                   │
  │  │    min Σ(RC_i − 1/n)²  ─────────────┼──► scipy SLSQP                    │
  │  │    s.t. Σω=1, ωᵢ>0                   │                                   │
  │  └─────────────────────────────────────┘                                   │
  └───────────────────────────────────────────────────────────────────────────┘
```

### Signal Family (`core/strategies/signal/`)

```
  ┌───────────────────────────────────────────────────────────────────────────┐
  │  signal/                                                                   │
  │                                                                            │
  │  momentum.py                                                               │
  │  ┌─────────────────────────────────────────────────────────────────────┐  │
  │  │  @reg  [MomentumStrategy]                                            │  │
  │  │─────────────────────────────────────────────────────────────────────│  │
  │  │  lookback_period: int  = 252   trailing window in days               │  │
  │  │  top_k          : int  = 3     number of assets to go long           │  │
  │  │  long_only      : bool = True  False → long-short (market neutral)   │  │
  │  │                                                                      │  │
  │  │  required_history_days → lookback_period + 1                         │  │
  │  │                                                                      │  │
  │  │  compute_weights():                                                   │  │
  │  │    trailing_return_i = S_today / S_(today-lookback) − 1              │  │
  │  │    rank all assets descending by trailing return                     │  │
  │  │    long-only : wᵢ = 1/k  for top k,  0  for the rest                │  │
  │  │    long-short: wᵢ = +1/(2k) top k,  −1/(2k) bottom k               │  │
  │  └─────────────────────────────────────────────────────────────────────┘  │
  │                                                                            │
  │  mean_reversion.py                                                         │
  │  ┌─────────────────────────────────────────────────────────────────────┐  │
  │  │  @reg  [MeanReversionStrategy]                                       │  │
  │  │─────────────────────────────────────────────────────────────────────│  │
  │  │  lookback_window: int   = 20    window for MA and std                │  │
  │  │  z_threshold    : float = 2.0   signal trigger                       │  │
  │  │                                                                      │  │
  │  │  required_history_days → lookback_window + 1                         │  │
  │  │                                                                      │  │
  │  │  compute_weights():                                                   │  │
  │  │    MA_i  = mean(S_i over window)                                     │  │
  │  │    std_i = std(S_i over window)                                      │  │
  │  │    z_i   = (S_i − MA_i) / std_i                                      │  │
  │  │    signal: z_i < −threshold  →  BUY  (oversold)                     │  │
  │  │    weight ∝ 1/|z_i| for triggered assets  (normalised to sum 1)      │  │
  │  │    no signal → 100% cash                                             │  │
  │  └─────────────────────────────────────────────────────────────────────┘  │
  └───────────────────────────────────────────────────────────────────────────┘
```

---

## Full Inheritance Tree

```
  (IStrategy)   ← abstract, in base.py
       │
       ├── hedging/
       │     ├── [DeltaHedgeStrategy]       @reg  — uses MonteCarloPricer (P1)
       │     └── [DeltaGammaHedgeStrategy]  @reg  — composes DeltaHedgeStrategy
       │
       ├── allocation/
       │     ├── [EqualWeightStrategy]      @reg  — pure arithmetic (1/n)
       │     ├── [MinVarianceStrategy]      @reg  — scipy SLSQP: min ωᵀΣω
       │     ├── [MaxSharpeStrategy]        @reg  — scipy SLSQP: max Sharpe
       │     └── [RiskParityStrategy]       @reg  — scipy SLSQP: equal RC_i
       │
       └── signal/
             ├── [MomentumStrategy]         @reg  — rank by trailing return
             └── [MeanReversionStrategy]    @reg  — z-score signal
```

---

## Cross-Phase Dependency Map

```
  Phase 1                              Phase 2
  ───────                              ───────

  core/models/
  ┌─────────────────┐
  │  PricingResult  │◄════════════════ DeltaHedgeStrategy.compute_weights()
  │  (deltas array) │                  (reads .deltas from result)
  └─────────────────┘

  core/pricing/
  ┌─────────────────────┐
  │  MonteCarloPricer   │◄──────────── DeltaHedgeStrategy.__init__ / compute_weights
  │  .compute_deltas()  │              (instantiates and calls directly)
  └─────────────────────┘

  Phase 2 → Phase 1 (one-way):
    DeltaHedgeStrategy  ──────────────►  MonteCarloPricer
    DeltaGammaHedgeStrategy  ─────────►  DeltaHedgeStrategy  ──►  MonteCarloPricer

  Phase 1 → Phase 2: NONE
  (Phase 1 code has zero knowledge of strategies — correct layering)
```

---

## Data Flow Through a Strategy Call

```
  BacktestEngine (Phase 3 — not yet built)
       │
       │  1. prices = data_provider.get_prices(symbols, start, t)  ← IDataProvider (P1)
       │  2. strategy.compute_weights(t, prices)                   ← IStrategy (P2)
       │                    │
       │                    │  [inside compute_weights]
       │                    ├── reads price_history (only up to t — no look-ahead)
       │                    ├── applies strategy math (see each strategy above)
       │                    └── returns PortfolioWeights
       │
       │  3. reads pw.weights  → rebalance portfolio
       │  4. records in BacktestResult (P1 stub)
       ▼
  Final BacktestResult (filled by Phase 3)
```

---

## Dependency Count Summary

| Class | Depends On (internal) | Depended On By |
|---|---|---|
| `StrategyConfig` | — | `IStrategy`, `StrategyRegistry` |
| `PortfolioWeights` | stdlib date | all `compute_weights()` return it |
| `IStrategy` | `StrategyConfig`, `PortfolioWeights`, pandas | all 8 concrete strategies |
| `StrategyRegistry` | `IStrategy`, `StrategyConfig` | all 8 strategies (register), backtester (create), API (list) |
| `EqualWeightStrategy` | `IStrategy`, `StrategyRegistry` | tests, backtester |
| `MinVarianceStrategy` | `IStrategy`, `StrategyRegistry`, scipy | tests, backtester |
| `MaxSharpeStrategy` | `IStrategy`, `StrategyRegistry`, scipy | tests, backtester |
| `RiskParityStrategy` | `IStrategy`, `StrategyRegistry`, scipy | tests, backtester |
| `MomentumStrategy` | `IStrategy`, `StrategyRegistry` | tests, backtester |
| `MeanReversionStrategy` | `IStrategy`, `StrategyRegistry` | tests, backtester |
| `DeltaHedgeStrategy` | `IStrategy`, `StrategyRegistry`, **`MonteCarloPricer` (P1)** | `DeltaGammaHedgeStrategy`, tests |
| `DeltaGammaHedgeStrategy` | `IStrategy`, `StrategyRegistry`, **`DeltaHedgeStrategy`** | tests, backtester |
