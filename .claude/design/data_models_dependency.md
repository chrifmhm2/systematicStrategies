# Data Models — Dependency Graph

> Phase 1 · `backend/core/models/`

```
╔══════════════════════════════════════════════════════════════════════╗
║                    core/models/  — Dependency Graph                  ║
╚══════════════════════════════════════════════════════════════════════╝

  External
  ────────
  stdlib: date          numpy: np.ndarray          pandas: pd.DataFrame
      │                        │                           │
      │          ┌─────────────┘                           │
      │          │                                         │
      ▼          ▼                                         │
 ┌─────────────────────┐   ┌─────────────────────┐        │
 │   market_data.py    │   │     results.py       │        │
 │ ─────────────────── │   │ ─────────────────── │        │
 │  DataFeed           │   │  PricingResult       │        │
 │   · symbol: str     │   │   · price: float     │        │
 │   · date: date ◄────┘   │   · std_error: float │        │
 │   · price: float    │   │   · confidence_      │        │
 │                     │   │     interval: tuple  │        │
 │  OHLCV              │   │   · deltas: ndarray ◄┘        │
 │   · symbol: str     │   │                     │        │
 │   · date: date      │   │  BacktestResult     │        │
 │   · open: float     │   │   (stub — Phase 3)  │        │
 │   · high: float     │   └──────────┬──────────┘        │
 │   · low: float      │              │ produced by        │
 │   · close: float    │              │ BacktestEngine      │
 │   · volume: float   │              │ (Phase 3)          │
 └─────────────────────┘              │                    │
                                      │                    │
 ┌─────────────────────┐              │                    │
 │    portfolio.py     │              │                    │
 │ ─────────────────── │              │                    │
 │  Position           │              │                    │
 │   · symbol: str     │              │                    │
 │   · quantity: float │              │                    │
 │   · price: float    │              │                    │
 │          │          │              │                    │
 │          ▼          │              ▼                    │
 │  Portfolio          │       ┌─────────────────┐        │
 │   · positions:      │       │   data/base.py  │        │
 │     dict[str,       │       │ ─────────────── │        │
 │     Position] ──────┘       │  IDataProvider  │        │
 │   · cash: float             │  · get_prices() │◄───────┘
 │   · date: date              │    → DataFrame  │
 │   · total_value()           │  · get_risk_    │
 │     → float                 │    free_rate()  │
 └─────────────────────┘       │    → float      │
                               └─────────────────┘
 ┌─────────────────────┐              ▲
 │     options.py      │              │ implemented by
 │ ─────────────────── │       ┌──────┴──────────────┐
 │  VanillaOption      │       │  SimulatedProvider  │
 │   · underlying: str │       │  CsvDataProvider    │
 │   · strike: float   │       │  YahooDataProvider  │
 │   · maturity: float │       │  (Phase 3)          │
 │   · option_type:str │       └─────────────────────┘
 │                     │
 │  BasketOption       │
 │   · underlyings:    │
 │     list[str]       │
 │   · weights:        │
 │     list[float]     │
 │   · strike: float   │
 │   · maturity: float │
 └─────────────────────┘

  Legend
  ──────
  ◄─── field type dependency
   ──► "produces" / "used by" relationship
   ▲   "implements" (subtype)
```

## Key Points

- `market_data.py` and `options.py` are **leaf nodes** — no internal `core/` dependencies
- `Portfolio` depends on `Position` (same file)
- `PricingResult` is returned by `MonteCarloPricer` (Phase 1)
- `BacktestResult` is filled by `BacktestEngine` (Phase 3 — stub only in Phase 1)
- `IDataProvider` is the **central interface** — strategies and the backtester only talk to this, never to concrete providers directly
