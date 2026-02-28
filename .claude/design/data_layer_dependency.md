# Data Layer — Dependency Graph

> Phase 1 · `backend/core/data/`

```
╔══════════════════════════════════════════════════════════════════════╗
║                   core/data/  — Dependency Graph                     ║
╚══════════════════════════════════════════════════════════════════════╝

  External
  ────────
  numpy: np        pandas: pd        csv file        yfinance
     │                  │                │               │
     │           ┌──────┘                │               │
     │           │                       │               │
     ▼           ▼                       ▼               ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                        data/base.py                              │
 │ ──────────────────────────────────────────────────────────────── │
 │  IDataProvider   (Abstract Base Class — ABC)                     │
 │                                                                  │
 │  @abstractmethod                                                 │
 │  get_prices(symbols, start_date, end_date) → pd.DataFrame        │
 │       └─ DatetimeIndex rows, one column per symbol (adj. close)  │
 │                                                                  │
 │  @abstractmethod                                                 │
 │  get_risk_free_rate(date) → float                                │
 └──────────────────────────┬───────────────────────────────────────┘
                            │  implemented by (subclass)
           ┌────────────────┼────────────────┐
           │                │                │
           ▼                ▼                ▼
 ┌──────────────────┐  ┌──────────────┐  ┌────────────────────────┐
 │  simulated.py    │  │ csv_loader.py│  │  yahoo.py  (Phase 3)   │
 │ ──────────────── │  │ ──────────── │  │ ────────────────────── │
 │ SimulatedData-   │  │ CsvData-     │  │ YahooDataProvider      │
 │ Provider         │  │ Provider     │  │                        │
 │                  │  │              │  │ · wraps yfinance       │
 │ Constructor:     │  │ · reads CSV  │  │ · fetches real OHLCV   │
 │  · spots: dict   │  │   with cols: │  │   from the internet    │
 │  · vols: dict    │  │   Id,        │  │ · get_risk_free_rate() │
 │  · corr: ndarray │  │   DateOfPrice│  │   → hardcoded 0.05     │
 │  · drift: float  │  │   Value      │  └────────────────────────┘
 │  · rfr: float    │  │              │
 │  · seed: int     │  │ · get_risk_  │
 │                  │  │   free_rate()│
 │ get_prices():    │  │   → 0.05     │
 │  GBM simulation  │  └──────────────┘
 │  ─────────────── │
 │  S_t = S_0 *     │
 │  exp((μ-σ²/2)dt  │
 │      +σ√dt·Z)    │
 │                  │
 │  Z drawn via     │
 │  Cholesky(corr)  │◄─── numpy
 │                  │
 │  returns         │
 │  pd.DataFrame    │──► pandas
 └──────────────────┘

                            │  consumed by
           ┌────────────────┼──────────────────┐
           ▼                ▼                  ▼
  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐
  │  IStrategy   │  │ BacktestEngine│  │ MonteCarloPricer  │
  │  (Phase 2)   │  │  (Phase 3)   │  │   (Phase 1)       │
  └──────────────┘  └──────────────┘  └───────────────────┘
  (always code against IDataProvider — never against concrete classes)

  Legend
  ──────
  ◄─── depends on / uses
   ──► produces / returns
   ▲   implements (subclass of)
```

## Key Design Point

Strategies and the backtester only ever call `provider.get_prices(...)` against the `IDataProvider` interface — they never import or reference concrete providers directly.

| Context | Provider used |
|---------|--------------|
| Tests | `SimulatedDataProvider` — fast, deterministic, no internet |
| CSV backtest | `CsvDataProvider` — reads Ensimag-format CSV |
| Production | `YahooDataProvider` — real market data via yfinance |

Zero code change needed in strategies or the backtester when swapping providers.
