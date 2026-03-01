# Phase 1 — Complete Class Dependency Map

> `backend/core/` · Python 3.10
> Generated after session 1 (all 46 tests passing)

---

## Legend

```
──────►   uses / imports / depends on
══════►   returns / produces
   ▲      inherits from (is-a)
───┤      composes (has-a / field of type)
[ ]       concrete class or dataclass
( )       abstract class / interface (ABC)
{ }       free function (no class)
```

---

## High-Level Overview — Three Modules, One Dependency Flow

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║   core/models/                core/data/               core/pricing/             ║
║   ─────────────               ──────────               ─────────────             ║
║   Pure data containers        Data source interface    Mathematical engine       ║
║   (dataclasses, no logic)     + two implementations    (BS + Monte Carlo)        ║
║                                                                                  ║
║   DataFeed     OHLCV          (IDataProvider)          { cholesky_decompose }    ║
║   Position     Portfolio          ▲     ▲              { gen_corr_normals   }    ║
║   VanillaOption                   │     │                      │                 ║
║   BasketOption            Simulated  Csv             [BlackScholesModel]         ║
║   PricingResult  ◄═══════════════════════════════════[MonteCarloPricer  ]        ║
║   BacktestResult  (stub)                                                         ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝

  Cross-module dependencies (the arrows that cross module boundaries):

    MonteCarloPricer  ──────►  PricingResult          (models → returned as output)
    MonteCarloPricer  ──────►  cholesky_decompose      (within pricing/)
    MonteCarloPricer  ──────►  generate_corr_normals   (within pricing/)
    SimulatedDataProvider  ──► numpy.linalg.cholesky   (external — inline, not via utils)

  Planned cross-module flows (Phases 2–4):

    IStrategy       ──────►  IDataProvider   (Phase 2 — strategies query prices)
    BacktestEngine  ──────►  IDataProvider   (Phase 3)
    BacktestEngine  ══════►  BacktestResult  (Phase 3 — fills the stub)
    GreeksCalc      ──────►  BlackScholesModel (Phase 4)
```

---

## Module 1 — `core/models/`  (Data Containers)

```
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  core/models/                                                                │
  │                                                                              │
  │  market_data.py                    options.py                                │
  │  ┌──────────────────┐              ┌────────────────────────────────┐        │
  │  │  [DataFeed]      │              │  [VanillaOption]               │        │
  │  │──────────────────│              │────────────────────────────────│        │
  │  │ symbol : str     │              │ underlying  : str              │        │
  │  │ date   : date    │              │ strike      : float            │        │
  │  │ price  : float   │              │ maturity    : float (years)    │        │
  │  └──────────────────┘              │ option_type : "call" | "put"  │        │
  │                                    └────────────────────────────────┘        │
  │  ┌──────────────────┐              ┌────────────────────────────────┐        │
  │  │  [OHLCV]         │              │  [BasketOption]                │        │
  │  │──────────────────│              │────────────────────────────────│        │
  │  │ symbol : str     │              │ underlyings : list[str]        │        │
  │  │ date   : date    │              │ weights     : list[float]      │        │
  │  │ open   : float   │              │ strike      : float            │        │
  │  │ high   : float   │              │ maturity    : float (years)    │        │
  │  │ low    : float   │              │ option_type : "call" | "put"  │        │
  │  │ close  : float   │              └────────────────────────────────┘        │
  │  │ volume : float   │                                                        │
  │  └──────────────────┘                                                        │
  │                                                                              │
  │  portfolio.py                                                                │
  │  ┌──────────────────┐                                                        │
  │  │  [Position]      │                                                        │
  │  │──────────────────│                                                        │
  │  │ symbol   : str   │                                                        │
  │  │ quantity : float │◄──────────────────────────────────────────────────┐   │
  │  │ price    : float │  composed-by                                       │   │
  │  └──────────────────┘                                                    │   │
  │                                                    ┌──────────────────────┴──┐│
  │                                                    │  [Portfolio]            ││
  │                                                    │─────────────────────────││
  │                                                    │ positions :             ││
  │                                                    │  dict[str, Position]    ││
  │                                                    │ cash : float            ││
  │                                                    │ date : date             ││
  │                                                    │ total_value(           ││
  │                                                    │   prices: dict[str,f]) ││
  │                                                    │  → float  (MtM equity  ││
  │                                                    │    + cash)              ││
  │                                                    └─────────────────────────┘│
  │                                                                              │
  │  results.py                                                                  │
  │  ┌─────────────────────────────────────────────────────────────────┐        │
  │  │  [PricingResult]                              ◄═══ MonteCarloPricer       │
  │  │─────────────────────────────────────────────────────────────────│        │
  │  │ price              : float                                       │        │
  │  │ std_error          : float     (σ / √N)                          │        │
  │  │ confidence_interval: tuple[float, float]  (95% CI)               │        │
  │  │ deltas             : np.ndarray  shape (n_assets,)               │        │
  │  └─────────────────────────────────────────────────────────────────┘        │
  │                                                                              │
  │  ┌─────────────────────────────────────────────────────────────────┐        │
  │  │  [BacktestResult]   stub — all fields have defaults             │        │
  │  │─────────────────────────────────────────────────────────────────│        │
  │  │ portfolio_values  : dict[str, float]  date → NAV                │        │
  │  │ benchmark_values  : dict[str, float] | None                     │        │
  │  │ weights_history   : dict[str, dict[str, float]]                 │        │
  │  │ trades_log        : list[dict]  one entry per trade             │        │
  │  │ risk_metrics      : dict  (filled by Phase 4)                   │        │
  │  │ computation_time_ms : float                                      │        │
  │  │                                                                  │        │
  │  │  ← BacktestEngine (Phase 3) fills all fields during the loop    │        │
  │  └─────────────────────────────────────────────────────────────────┘        │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 2 — `core/data/`  (Data Provider Interface)

```
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  core/data/                                                                  │
  │                                                                              │
  │  base.py                                                                     │
  │  ┌──────────────────────────────────────────────────────────────────────┐   │
  │  │  (IDataProvider)   — Abstract Base Class (ABC)                        │   │
  │  │──────────────────────────────────────────────────────────────────────│   │
  │  │                                                                       │   │
  │  │  @abstractmethod                                                      │   │
  │  │  get_prices(symbols: list[str],                                       │   │
  │  │             start_date: date,                                         │   │
  │  │             end_date: date) → pd.DataFrame                            │   │
  │  │       returns: DatetimeIndex rows × symbol columns (adj. close)       │   │
  │  │               missing values forward-filled by the provider           │   │
  │  │                                                                       │   │
  │  │  @abstractmethod                                                      │   │
  │  │  get_risk_free_rate(date: date) → float                               │   │
  │  │       returns: annualised rate (e.g. 0.05 for 5%)                    │   │
  │  │                                                                       │   │
  │  └──────────────────────────────────┬────────────────────────────────────┘  │
  │                                     │  concrete subclasses                  │
  │              ┌──────────────────────┼──────────────────────┐                │
  │              ▼                      ▼                      ▼                │
  │  ┌───────────────────────┐  ┌───────────────────┐  ┌────────────────────┐  │
  │  │  [SimulatedData-      │  │  [CsvDataProvider] │  │  YahooDataProvider │  │
  │  │   Provider]           │  │───────────────────│  │  (Phase 3)         │  │
  │  │───────────────────────│  │ filepath: str      │  │────────────────────│  │
  │  │ Constructor params:   │  │ risk_free_rate:    │  │ wraps yfinance     │  │
  │  │  spots: dict[str,f]  │  │  float = 0.05      │  │ real OHLCV from    │  │
  │  │  volatilities:       │  │                    │  │ the internet       │  │
  │  │    dict[str,float]   │  │ CSV format:        │  └────────────────────┘  │
  │  │  correlation:        │  │  Id, DateOfPrice,  │                          │
  │  │    np.ndarray        │  │  Value             │                          │
  │  │  drift: float = 0.0  │  │                    │                          │
  │  │  risk_free_rate: f   │  │ _load():           │                          │
  │  │  seed: int | None    │  │  pd.read_csv()     │                          │
  │  │                      │  │  pivot long→wide   │                          │
  │  │ get_prices():        │  │  ffill gaps        │                          │
  │  │  Correlated GBM:     │  │                    │                          │
  │  │  S_t = S₀ · exp(     │  │ get_risk_free_     │                          │
  │  │   (μ-σ²/2)·dt        │  │  rate() → 0.05    │                          │
  │  │   + σ·√dt · Z)       │  └───────────────────┘                          │
  │  │  Z = ε @ L.T         │                                                  │
  │  │  L = Cholesky(corr)  │──► numpy.linalg (internal)                      │
  │  │  → pd.DataFrame      │──► pandas                                        │
  │  └───────────────────────┘                                                  │
  │                                                                              │
  │  Design rule: IStrategy and BacktestEngine only import IDataProvider.       │
  │  Concrete provider is injected at startup — zero code change to swap it.    │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 3 — `core/pricing/`  (Mathematical Engine)

```
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  core/pricing/                                                               │
  │                                                                              │
  │  utils.py  (free functions — no class)                                       │
  │  ┌──────────────────────────────────────────────────────────────────────┐   │
  │  │                                                                       │   │
  │  │  { cholesky_decompose(correlation: np.ndarray) → L }                 │   │
  │  │       input : square correlation matrix, shape (n, n)                │   │
  │  │       output: lower-triangular L  s.t.  L @ L.T == correlation       │   │
  │  │       raises: ValueError if not positive definite                    │   │
  │  │                                                                       │   │
  │  │  { generate_correlated_normals(n_assets, n_samples, chol,            │   │
  │  │                                seed) → Z }                           │   │
  │  │       step 1: ε ~ N(0, I)   shape (n_samples, n_assets)              │   │
  │  │       step 2: Z = ε @ chol.T                                         │   │
  │  │       output: shape (n_samples, n_assets), cov = chol @ chol.T       │   │
  │  │                                                                       │   │
  │  └───────────────────────────────────────┬───────────────────────────────┘  │
  │                                          │ used by                          │
  │                                          ▼                                  │
  │  monte_carlo.py                                                              │
  │  ┌──────────────────────────────────────────────────────────────────────┐   │
  │  │  [MonteCarloPricer]                                                   │   │
  │  │──────────────────────────────────────────────────────────────────────│   │
  │  │ n_simulations     : int = 100_000                                     │   │
  │  │ seed              : int | None                                        │   │
  │  │ variance_reduction: "antithetic" | "none"                            │   │
  │  │                                                                       │   │
  │  │ price_basket_option(spots, weights, strike, maturity,                 │   │
  │  │                     risk_free_rate, volatilities, correlation)        │   │
  │  │  │                                                                    │   │
  │  │  ├── cholesky_decompose(correlation) ──────────────────────► utils   │   │
  │  │  ├── _simulate_payoffs()                                              │   │
  │  │  │     ├── generate_correlated_normals() ──────────────────► utils   │   │
  │  │  │     ├── GBM terminal:  S_T = S · exp((r-σ²/2)T + σ√T · Z)        │   │
  │  │  │     ├── Basket payoff: max(Σ ωᵢ·Sᵢᵀ − K,  0)                     │   │
  │  │  │     └── Antithetic:    avg(payoff(+Z), payoff(−Z))                │   │
  │  │  ├── compute_deltas() (see below)                                     │   │
  │  │  └══► PricingResult ────────────────────────────────────────► models │   │
  │  │                                                                       │   │
  │  │ compute_deltas(spots, weights, ..., bump_size=0.01) → np.ndarray     │   │
  │  │  │  Central finite difference per asset i:                           │   │
  │  │  │  Δᵢ = [V(Sᵢ + bump·Sᵢ) − V(Sᵢ − bump·Sᵢ)] / (2·bump·Sᵢ)       │   │
  │  │  │  Same seed reused for up/down bumps → Monte Carlo noise cancels   │   │
  │  │  └──► np.ndarray  shape (n_assets,)                                  │   │
  │  └──────────────────────────────────────────────────────────────────────┘   │
  │                                                                              │
  │  black_scholes.py  (no imports from core/ — fully standalone)               │
  │  ┌──────────────────────────────────────────────────────────────────────┐   │
  │  │  [BlackScholesModel]   all @staticmethod                             │   │
  │  │──────────────────────────────────────────────────────────────────────│   │
  │  │                                                                       │   │
  │  │  Internal helpers (private):                                          │   │
  │  │  _d1(S,K,T,r,σ)  =  [ln(S/K) + (r + σ²/2)·T] / (σ·√T)              │   │
  │  │  _d2(S,K,T,r,σ)  =  d1 − σ·√T                                       │   │
  │  │                                                                       │   │
  │  │  Public pricing:                                                      │   │
  │  │  call_price(S,K,T,r,σ) → float    S·N(d1) − K·e⁻ʳᵀ·N(d2)           │   │
  │  │  put_price(S,K,T,r,σ)  → float    C − S + K·e⁻ʳᵀ  (parity)         │   │
  │  │                                                                       │   │
  │  │  Greeks:                                                              │   │
  │  │  delta(…, option_type) → float    N(d1) call ∈[0,1]                  │   │
  │  │                                   N(d1)−1 put ∈[−1,0]                │   │
  │  │  gamma(…)              → float    n(d1)/(S·σ·√T)  always > 0        │   │
  │  │  vega(…)               → float    S·n(d1)·√T      always > 0        │   │
  │  │  theta(…, option_type) → float    daily decay ÷365  usually < 0     │   │
  │  │  rho(…, option_type)   → float    K·T·e⁻ʳᵀ·N(d2) call              │   │
  │  │                                                                       │   │
  │  │  Inverse problem:                                                     │   │
  │  │  implied_volatility(price, S,K,T,r, option_type)  → float            │   │
  │  │     Newton-Raphson:  σ_new = σ_old − (BS(σ)−price) / vega(σ)        │   │
  │  │     converges in < 20 iterations for typical market conditions        │   │
  │  │                                                                       │   │
  │  │  External only: scipy.stats.norm  (N/n),  math,  numpy               │   │
  │  └──────────────────────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Cross-Module Dependency Summary

```
  ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
  │   core/models/   │       │   core/data/      │       │  core/pricing/   │
  │                  │       │                  │       │                  │
  │  DataFeed        │       │  IDataProvider   │       │  utils.py        │
  │  OHLCV           │       │  (ABC)           │       │  ├─ cholesky_    │
  │  VanillaOption   │       │       ▲          │       │  │   decompose   │
  │  BasketOption    │       │       │          │       │  └─ gen_corr_    │
  │  Position        │       │  Simulated  Csv  │       │      normals     │
  │  Portfolio       │       │  DataProvider    │       │                  │
  │                  │       │                  │       │  BlackScholes-   │
  │  PricingResult ◄═╪═══════╪══════════════════╪═══════╪═ Model           │
  │  BacktestResult  │       │                  │       │  (standalone)    │
  │  (stub)          │       │                  │       │                  │
  └──────────────────┘       └──────────────────┘       │  MonteCarlo-     │
                                                         │  Pricer ────────►│
                                                         │   (uses utils,   │
                                                         │    returns       │
                                                         │    PricingResult)│
                                                         └──────────────────┘

  Arrow key for cross-module:
    ══════►  MonteCarloPricer returns PricingResult  (pricing → models)
    ──────►  Future: IStrategy uses IDataProvider    (Phase 2)
    ──────►  Future: BacktestEngine uses IDataProvider + fills BacktestResult (Phase 3)
```

---

## Dependency Topology (Sorted by Number of Dependents)

| Class / Function | Depends On (internal) | Depended On By |
|---|---|---|
| `DataFeed` | — | tests, demos |
| `OHLCV` | — | tests, demos |
| `Position` | — | `Portfolio` |
| `VanillaOption` | — | tests |
| `BasketOption` | — | `MonteCarloPricer` inputs |
| `cholesky_decompose` | numpy only | `MonteCarloPricer` |
| `generate_correlated_normals` | numpy only | `MonteCarloPricer` |
| `BlackScholesModel` | scipy, math, numpy only | Phase 4 GreeksCalculator |
| `Portfolio` | `Position` | `BacktestEngine` (Phase 3) |
| `PricingResult` | numpy only | `MonteCarloPricer` (produces it) |
| `BacktestResult` | — (stub) | `BacktestEngine` (Phase 3) |
| `IDataProvider` | pandas, stdlib | `SimulatedDataProvider`, `CsvDataProvider`, `IStrategy` (P2), `BacktestEngine` (P3) |
| `SimulatedDataProvider` | `IDataProvider`, numpy | tests, `BacktestEngine` |
| `CsvDataProvider` | `IDataProvider`, pandas | tests, `BacktestEngine` |
| `MonteCarloPricer` | `PricingResult`, `cholesky_decompose`, `generate_correlated_normals` | `DeltaHedgeStrategy` (P2), Phase 5 API |

---

## What This Means for Phase 2

Phase 2 adds `core/strategies/`. Every strategy will:

1. **Inherit** from `IStrategy` (the new ABC defined in Phase 2)
2. **Accept** an `IDataProvider` (injected — never imported directly)
3. **Call** `provider.get_prices()` to read historical prices
4. **Return** `PortfolioWeights` (new dataclass defined in Phase 2)

For `DeltaHedgeStrategy` specifically:

```
  DeltaHedgeStrategy
    ├── inherits IStrategy (Phase 2)
    ├── uses IDataProvider.get_prices()  (Phase 1)
    ├── calls MonteCarloPricer.compute_deltas()  (Phase 1)
    └── returns PortfolioWeights  (Phase 2)
```

No changes needed to any Phase 1 code.
