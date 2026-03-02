# Phase 5 — FastAPI Backend Dependency Graph

> `backend/main.py` · `backend/config.py` · `backend/api/` · Python 3.12 + FastAPI
> Designed before session 5 (107/107 tests passing from Phases 1–4)

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
{ }       free function / module-level helper
⬡         external library (outside core/)
→ 400     HTTP error — bad business logic (unknown strategy, invalid range)
→ 422     HTTP error — Pydantic validation failure (automatic)
→ 500     HTTP error — unhandled exception (global handler)
```

---

## High-Level Layer Map

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  HTTP Client  (browser · frontend · curl · TestClient)                              ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
                              │  HTTP request (JSON body / query params)
                              ▼
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  backend/main.py                                                                     ║
║  ───────────────                                                                     ║
║  FastAPI(title="QuantForge API", version="1.0.0")                                   ║
║  ├── CORSMiddleware(allow_origins=["*"])                                              ║
║  ├── global exception handler → HTTP 500 {error, type}                              ║
║  └── include_router(…, prefix="/api") × 6 routers                                   ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
         │              │             │            │           │           │
         ▼              ▼             ▼            ▼           ▼           ▼
   /strategies     /backtest      /hedging      /risk        /data      /pricing
╔══════════╗  ╔══════════════╗  ╔══════════╗  ╔═══════╗  ╔═══════╗  ╔══════════╗
║strategies║  ║  backtest.py ║  ║hedging.py║  ║risk.py║  ║data.py║  ║pricing.py║
║  .py     ║  ║              ║  ║          ║  ║       ║  ║       ║  ║          ║
╚════╤═════╝  ╚══════╤═══════╝  ╚════╤═════╝  ╚══╤════╝  ╚══╤════╝  ╚═════╤════╝
     │               │               │            │          │             │
     ▼               ▼               ▼            ▼          ▼             ▼
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  core/  (Phases 1–4 — zero web deps)                                                ║
║  ─────                                                                               ║
║  strategies/   StrategyRegistry · IStrategy · all 8 strategies         (Phase 2)    ║
║  backtester/   BacktestConfig · BacktestEngine                         (Phase 3)    ║
║  data/         IDataProvider · SimulatedDataProvider · YahooDataProvider (P1 + P3) ║
║  risk/         PerformanceMetrics · VaRCalculator · GreeksCalculator   (Phase 4)    ║
║  pricing/      BlackScholesModel · MonteCarloPricer                    (Phase 1)    ║
║  models/       BacktestResult · PricingResult · Portfolio              (Phase 1)    ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## File Structure Created in Phase 5

```
backend/
├── main.py                       [P5-01]  FastAPI app + middleware + routers
├── config.py                     [P5-02]  Settings dataclass
└── api/
    ├── __init__.py               [P5-03]  empty
    ├── schemas.py                [P5-04]  all Pydantic request/response models
    └── routes/
        ├── __init__.py           [P5-03]  empty
        ├── strategies.py         [P5-05]  GET /strategies, GET /strategies/{id}
        ├── backtest.py           [P5-06]  POST /backtest, POST /backtest/compare
        ├── hedging.py            [P5-07]  POST /hedging/simulate
        ├── risk.py               [P5-08]  POST /risk/analyze
        ├── data.py               [P5-09]  GET /data/assets, GET /data/prices
        └── pricing.py            [P5-10]  POST /pricing/option

tests/
└── test_api.py                   [P5-13]  9 TestClient tests
```

---

## Module 1 — `backend/main.py`  [P5-01]

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  main.py                                                                      │
  │                                                                               │
  │  app = FastAPI(                                                                │
  │      title    = "QuantForge API",                                             │
  │      version  = "1.0.0",                                                      │
  │      docs_url = "/docs",      ← Swagger UI                                   │
  │  )                                                                            │
  │                                                                               │
  │  ── Middleware ────────────────────────────────────────────────────────────── │
  │                                                                               │
  │  CORSMiddleware(                                                               │
  │      allow_origins     = ["*"],    ← all origins (tightened in Phase 7)      │
  │      allow_methods     = ["*"],                                               │
  │      allow_headers     = ["*"],                                               │
  │      allow_credentials = True,                                                │
  │  )                                                                            │
  │                                                                               │
  │  ── Global exception handler [P5-11] ──────────────────────────────────────── │
  │                                                                               │
  │  @app.exception_handler(Exception)                                            │
  │  async def global_error_handler(request, exc):                                │
  │      return JSONResponse(                                                     │
  │          status_code = 500,                                                   │
  │          content     = {"error": str(exc), "type": type(exc).__name__},       │
  │      )                                                                        │
  │                                                                               │
  │  ── Router mounts ─────────────────────────────────────────────────────────── │
  │                                                                               │
  │  app.include_router(strategies_router, prefix="/api")                         │
  │  app.include_router(backtest_router,   prefix="/api")                         │
  │  app.include_router(hedging_router,    prefix="/api")                         │
  │  app.include_router(risk_router,       prefix="/api")                         │
  │  app.include_router(data_router,       prefix="/api")                         │
  │  app.include_router(pricing_router,    prefix="/api")                         │
  │                                                                               │
  │  ── Root health check ─────────────────────────────────────────────────────── │
  │                                                                               │
  │  GET /  →  {"status": "ok", "version": "1.0.0"}                              │
  │                                                                               │
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 2 — `backend/config.py`  [P5-02]

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  config.py                                                                    │
  │                                                                               │
  │  @dataclass                                                                   │
  │  class Settings:                                                              │
  │      debug               : bool = False                                       │
  │      default_data_source : str  = "yahoo"    # "yahoo" | "simulated"         │
  │      max_backtest_years  : int  = 10                                          │
  │                                                                               │
  │  settings = Settings()   ← singleton, imported by routes                     │
  │                                                                               │
  │  No pydantic-settings needed — plain dataclass is sufficient for Phase 5.    │
  │  Phase 7 can upgrade to pydantic-settings + .env file.                       │
  │                                                                               │
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 3 — `backend/api/schemas.py`  [P5-04]

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  schemas.py     (all models inherit from pydantic BaseModel)                  │
  │                                                                               │
  │  ── Request models ────────────────────────────────────────────────────────── │
  │                                                                               │
  │  [BacktestRequest]                              [P5-04a]                      │
  │  ┌──────────────────────────────────────────┐                                │
  │  │ strategy_id  : str                        │  "EqualWeightStrategy"         │
  │  │ symbols      : list[str]                  │  ["AAPL", "MSFT", ...]         │
  │  │ start_date   : date                       │  2022-01-01                    │
  │  │ end_date     : date                       │  2023-12-31                    │
  │  │ initial_value: float = 100_000            │                                │
  │  │ params       : dict  = {}                 │  strategy-specific kwargs      │
  │  │ data_source  : str   = "simulated"        │  "yahoo" | "simulated"         │
  │  │                                           │                                │
  │  │ @validator: start_date < end_date         │  → HTTP 422 if violated        │
  │  │ @validator: initial_value > 0             │  → HTTP 422 if violated        │
  │  │ @validator: symbols non-empty strings     │  → HTTP 422 if violated        │
  │  └──────────────────────────────────────────┘                                │
  │                                                                               │
  │  [CompareRequest]                               [P5-04c]                      │
  │  ┌──────────────────────────────────────────┐                                │
  │  │ strategies   : list[StrategySpec]         │                                │
  │  │     StrategySpec: {strategy_id, params}   │                                │
  │  │ symbols      : list[str]                  │                                │
  │  │ start_date   : date                       │                                │
  │  │ end_date     : date                       │                                │
  │  │ initial_value: float = 100_000            │                                │
  │  │ data_source  : str   = "simulated"        │                                │
  │  └──────────────────────────────────────────┘                                │
  │                                                                               │
  │  [HedgingRequest]                               [P5-04d]                      │
  │  ┌──────────────────────────────────────────┐                                │
  │  │ symbols           : list[str]             │                                │
  │  │ strike            : float                 │                                │
  │  │ maturity_years    : float                 │                                │
  │  │ risk_free_rate    : float = 0.05          │                                │
  │  │ volatilities      : list[float]           │  one per symbol                │
  │  │ correlation_matrix: list[list[float]]     │  n×n                           │
  │  │ initial_spots     : list[float]           │  starting prices               │
  │  │ n_simulations     : int   = 20_000        │  MC paths for pricing          │
  │  │ n_paths           : int   = 5             │  independent hedge simulations │
  │  │ rebalancing_freq  : str   = "weekly"      │                                │
  │  └──────────────────────────────────────────┘                                │
  │                                                                               │
  │  [OptionPricingRequest]                         [P5-04f]                      │
  │  ┌──────────────────────────────────────────┐                                │
  │  │ option_type : str   = "call"              │  "call" | "put"                │
  │  │ S           : float                       │  spot price                    │
  │  │ K           : float                       │  strike                        │
  │  │ T           : float                       │  maturity in years             │
  │  │ r           : float                       │  risk-free rate                │
  │  │ sigma       : float                       │  volatility                    │
  │  │ method      : str   = "bs"                │  "bs" | "mc"                   │
  │  │ n_simulations: int  = 10_000              │  used only when method="mc"    │
  │  └──────────────────────────────────────────┘                                │
  │                                                                               │
  │  [RiskAnalyzeRequest]                           [P5-04h]                      │
  │  ┌──────────────────────────────────────────┐                                │
  │  │ portfolio_values : dict[str, float]       │  {"2022-01-03": 100000, …}     │
  │  │ benchmark_values : dict[str, float] | None│                                │
  │  │ risk_free_rate   : float = 0.05           │                                │
  │  └──────────────────────────────────────────┘                                │
  │                                                                               │
  │  ── Response models ───────────────────────────────────────────────────────── │
  │                                                                               │
  │  [BacktestResponse]                             [P5-04b]                      │
  │  ┌──────────────────────────────────────────┐                                │
  │  │ strategy_name      : str                  │                                │
  │  │ portfolio_values   : dict[str, float]     │  {"2022-01-03": 100020, …}     │
  │  │ benchmark_values   : dict[str, float]|None│                                │
  │  │ weights_history    : dict[str,            │  {"2022-01-03":               │
  │  │                       dict[str, float]]   │    {"AAPL": 0.5, "MSFT": 0.5}}│
  │  │ risk_metrics       : dict                 │  NaN → None before returning   │
  │  │ trades_log         : list[dict]           │  [{date, symbol, qty, price,…}]│
  │  │ computation_time_ms: float                │                                │
  │  └──────────────────────────────────────────┘                                │
  │                                                                               │
  │  [HedgingResponse]                              [P5-04e]                      │
  │  ┌──────────────────────────────────────────┐                                │
  │  │ paths                  : list[dict]       │  one entry per simulated path  │
  │  │     each: {dates, portfolio_values,       │                                │
  │  │            option_values, pnl}            │                                │
  │  │ average_tracking_error : float            │  std of final P&L across paths │
  │  │ initial_option_price   : float            │  MC price at t=0               │
  │  │ initial_option_price_ci: list[float, float] │  95% CI                     │
  │  └──────────────────────────────────────────┘                                │
  │                                                                               │
  │  [OptionPricingResponse]                        [P5-04g]                      │
  │  ┌──────────────────────────────────────────┐                                │
  │  │ price              : float                │                                │
  │  │ std_error          : float | None         │  None when method="bs"         │
  │  │ confidence_interval: list[float] | None   │  None when method="bs"         │
  │  │ greeks             : dict                 │  {delta, gamma, vega, theta,   │
  │  │                                           │   rho} always computed via BS  │
  │  └──────────────────────────────────────────┘                                │
  │                                                                               │
  │  [StrategyInfo]                                 [P5-04i]                      │
  │  ┌──────────────────────────────────────────┐                                │
  │  │ id          : str   "EqualWeightStrategy" │                                │
  │  │ name        : str   "Equal Weight"        │                                │
  │  │ family      : str   "allocation"          │                                │
  │  │ description : str                         │                                │
  │  │ params      : dict  {}                    │  default constructor kwargs    │
  │  └──────────────────────────────────────────┘                                │
  │                                                                               │
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 4 — `api/routes/strategies.py`  [P5-05]

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  GET /api/strategies                                                          │
  │                                                                               │
  │  1. StrategyRegistry.list_strategies()  ──────────────────────► list[dict]   │
  │  2. Map each dict → StrategyInfo model                                       │
  │  3. return {"strategies": [...]}                                              │
  │                                                                               │
  │  Response 200:                                                                │
  │  {                                                                            │
  │    "strategies": [                                                            │
  │      {"id": "EqualWeightStrategy", "name": "Equal Weight",                   │
  │       "family": "allocation", "description": "…", "params": {}},             │
  │      …                                                                        │
  │    ]                                                                          │
  │  }                                                                            │
  │                                                                               │
  ├──────────────────────────────────────────────────────────────────────────────┤
  │  GET /api/strategies/{strategy_id}                                            │
  │                                                                               │
  │  1. strategies = StrategyRegistry.list_strategies()                           │
  │  2. find entry where entry["name"] == strategy_id   ← match by class name    │
  │  3. not found → raise HTTPException(404)                                      │
  │  4. return StrategyInfo                                                       │
  │                                                                               │
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 5 — `api/routes/backtest.py`  [P5-06]

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  POST /api/backtest                                                           │
  │                                                                               │
  │  Body: BacktestRequest (validated by Pydantic → 422 on schema error)         │
  │                                                                               │
  │  1. Validate business logic [P5-12]                                           │
  │       start_date < end_date          → HTTP 400 if not                       │
  │       strategy_id exists in registry → HTTP 400 if not                       │
  │       initial_value > 0              → caught by Pydantic validator (422)    │
  │                                                                               │
  │  2. Build strategy                                                            │
  │       strategy = StrategyRegistry.create(req.strategy_id, req.params)        │
  │       KeyError  → raise HTTPException(400, "Unknown strategy: …")           │
  │                                                                               │
  │  3. Build BacktestConfig                                                      │
  │       cfg = BacktestConfig(                                                   │
  │           symbols       = req.symbols,                                        │
  │           start_date    = req.start_date,                                     │
  │           end_date      = req.end_date,                                       │
  │           initial_value = req.initial_value,                                  │
  │           **req.params.get("backtest_config", {}),                            │
  │       )                                                                       │
  │                                                                               │
  │  4. Select data provider                                                      │
  │       "yahoo"     → YahooDataProvider()                                      │
  │       "simulated" → SimulatedDataProvider(                                   │
  │                         spots        = {s: 100.0 for s in symbols},          │
  │                         volatilities = {s: 0.20 for s in symbols},           │
  │                         correlation  = np.eye(n),                            │
  │                     )                                                         │
  │                                                                               │
  │  5. Run backtest                                                               │
  │       result = BacktestEngine(cfg).run(strategy, provider)                   │
  │       ← result.risk_metrics already populated by Phase 4                     │
  │                                                                               │
  │  6. Serialize to response   [key serialization decisions below]               │
  │       return BacktestResponse(...)                                            │
  │                                                                               │
  ├──────────────────────────────────────────────────────────────────────────────┤
  │  POST /api/backtest/compare                                                   │
  │                                                                               │
  │  Body: CompareRequest                                                         │
  │                                                                               │
  │  1. Validate: at least 2 strategies                                           │
  │  2. For each {strategy_id, params} in req.strategies:                        │
  │       run the POST /backtest logic above                                     │
  │  3. return list[BacktestResponse]                                             │
  │                                                                               │
  └──────────────────────────────────────────────────────────────────────────────┘

  ── Serialization: BacktestResult → BacktestResponse ────────────────────────────

  BacktestResult (Python internal)           BacktestResponse (JSON)
  ─────────────────────────────────          ───────────────────────
  portfolio_values : pd.Series          →    {str(date): float}
    DatetimeIndex → float                     key = "2022-01-03"
                                              value = 100020.0

  benchmark_values : pd.Series | None   →    {str(date): float} | None

  weights_history  : pd.DataFrame        →    {str(date): {symbol: float}}
    rows = rebalancing dates                  e.g. {"2022-01-03":
    cols = symbols                                   {"AAPL": 0.5, "MSFT": 0.5}}

  trades_log       : list[dict]          →    list[dict]  (already JSON-safe)

  risk_metrics     : dict                →    dict  (NaN → None)
    may contain float("nan")                  json.dumps can't handle NaN
                                              → replace with None before return

  ── Helper function (shared across routes) ───────────────────────────────────

  def _result_to_response(result: BacktestResult) -> BacktestResponse:
      # Series → date-keyed dict
      pv = {str(k.date()): v for k, v in result.portfolio_values.items()}
      bv = ({str(k.date()): v for k, v in result.benchmark_values.items()}
             if result.benchmark_values is not None else None)
      # DataFrame → nested dict
      wh = {}
      for ts, row in result.weights_history.iterrows():
          wh[str(ts.date())] = {col: row[col] for col in row.index}
      # NaN → None in risk_metrics
      rm = {k: (None if isinstance(v, float) and math.isnan(v) else v)
             for k, v in result.risk_metrics.items()}
      return BacktestResponse(
          strategy_name       = result.strategy_name,
          portfolio_values    = pv,
          benchmark_values    = bv,
          weights_history     = wh,
          risk_metrics        = rm,
          trades_log          = result.trades_log,
          computation_time_ms = result.computation_time_ms,
      )
```

---

## Module 6 — `api/routes/hedging.py`  [P5-07]

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  POST /api/hedging/simulate                                                   │
  │                                                                               │
  │  Body: HedgingRequest                                                         │
  │                                                                               │
  │  Concept: run n_paths independent simulated price paths through a delta       │
  │  hedge strategy. At each step compare option value to hedged portfolio.       │
  │                                                                               │
  │  1. Price the option at t=0 via MonteCarloPricer                             │
  │       spots_arr = np.array(req.initial_spots)                                 │
  │       basket = BasketOption(strike=req.strike, …)                             │
  │       pricing_result = MonteCarloPricer.price(basket, spots_arr, T, r, …)   │
  │                                                                               │
  │  2. Instantiate strategy                                                      │
  │       strat = DeltaHedgeStrategy(                                             │
  │           StrategyConfig(name="delta_hedge", description="…"),                │
  │           strike            = req.strike,                                     │
  │           maturity_years    = req.maturity_years,                             │
  │           volatilities      = req.volatilities,                               │
  │           correlation       = req.correlation_matrix,                         │
  │           risk_free_rate    = req.risk_free_rate,                             │
  │           n_simulations     = req.n_simulations,                              │
  │       )                                                                       │
  │                                                                               │
  │  3. Simulate n_paths independent paths                                        │
  │       for i in range(req.n_paths):                                            │
  │           provider = SimulatedDataProvider(                                   │
  │               spots        = dict(zip(req.symbols, req.initial_spots)),       │
  │               volatilities = dict(zip(req.symbols, req.volatilities)),        │
  │               correlation  = np.array(req.correlation_matrix),               │
  │               seed         = i,    ← different seed per path                  │
  │           )                                                                   │
  │           cfg    = BacktestConfig(symbols=req.symbols, …)                    │
  │           result = BacktestEngine(cfg).run(strat, provider)                  │
  │           paths.append({                                                      │
  │               "path_id":         i,                                           │
  │               "portfolio_values": {str(k.date()): v for k, v               │
  │                                    in result.portfolio_values.items()},       │
  │               "final_pnl":        result.portfolio_values.iloc[-1]           │
  │                                   - result.portfolio_values.iloc[0],          │
  │           })                                                                  │
  │                                                                               │
  │  4. Compute tracking error across paths                                       │
  │       final_pnls = [p["final_pnl"] for p in paths]                           │
  │       tracking_error = float(np.std(final_pnls))                             │
  │                                                                               │
  │  5. Return HedgingResponse                                                   │
  │                                                                               │
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 7 — `api/routes/risk.py`  [P5-08]

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  POST /api/risk/analyze                                                       │
  │                                                                               │
  │  Body: RiskAnalyzeRequest                                                     │
  │                                                                               │
  │  1. Deserialize portfolio_values dict → pd.Series (DatetimeIndex)            │
  │       pv = pd.Series(req.portfolio_values)                                   │
  │       pv.index = pd.to_datetime(pv.index)                                    │
  │                                                                               │
  │  2. Same for benchmark if provided                                            │
  │                                                                               │
  │  3. Call compute_all                                                          │
  │       metrics = PerformanceMetrics.compute_all(                               │
  │           values          = pv,                                               │
  │           benchmark       = bv,                                               │
  │           risk_free_rate  = req.risk_free_rate,                               │
  │       )                                                                       │
  │                                                                               │
  │  4. Replace NaN → None, return dict                                           │
  │                                                                               │
  │  Response 200: the metrics dict directly                                      │
  │  {                                                                            │
  │    "sharpe_ratio": 1.23,                                                      │
  │    "max_drawdown": -0.15,                                                     │
  │    "var_95": -0.021,                                                          │
  │    …                                                                          │
  │  }                                                                            │
  │                                                                               │
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 8 — `api/routes/data.py`  [P5-09]

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  GET /api/data/assets                                                         │
  │                                                                               │
  │  1. Import DEFAULT_UNIVERSE from core.data.yahoo                              │
  │  2. Return list of {"symbol": s} dicts                                        │
  │                                                                               │
  │  Response 200:                                                                │
  │  {"assets": [{"symbol": "AAPL"}, {"symbol": "MSFT"}, …]}                    │
  │                                                                               │
  ├──────────────────────────────────────────────────────────────────────────────┤
  │  GET /api/data/prices                                                         │
  │      ?symbols=AAPL,MSFT&start=2022-01-01&end=2023-12-31                      │
  │                                                                               │
  │  Query params:                                                                │
  │      symbols : str  (comma-separated)                                         │
  │      start   : date                                                           │
  │      end     : date                                                           │
  │                                                                               │
  │  1. Parse symbols list: symbols.split(",")                                   │
  │  2. provider = YahooDataProvider()                                            │
  │  3. df = provider.get_prices(symbols, start, end)   → pd.DataFrame           │
  │  4. Serialize: {symbol: {str(date): float}} nested dict                      │
  │       prices = {col: {str(idx.date()): df.loc[idx, col]                      │
  │                        for idx in df.index}                                   │
  │                 for col in df.columns}                                        │
  │  5. return {"prices": prices}                                                 │
  │                                                                               │
  │  Error handling:                                                              │
  │      yfinance failure → HTTP 502 Bad Gateway with descriptive message         │
  │                                                                               │
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 9 — `api/routes/pricing.py`  [P5-10]

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  POST /api/pricing/option                                                     │
  │                                                                               │
  │  Body: OptionPricingRequest                                                   │
  │                                                                               │
  │  Branch on method: ────────────────────────────────────────────────────────── │
  │                                                                               │
  │  method = "bs"                                                                │
  │  ───────────                                                                  │
  │  price = BlackScholesModel.call_price(S, K, T, r, sigma)                    │
  │          or  .put_price(…)                                                    │
  │  std_error          = None                                                    │
  │  confidence_interval = None                                                   │
  │                                                                               │
  │  method = "mc"                                                                │
  │  ───────────                                                                  │
  │  basket = BasketOption(strike=K, maturity=T, option_type=req.option_type,    │
  │                        weights=[1.0])                                         │
  │  result = MonteCarloPricer.price(                                            │
  │               basket, spots=np.array([S]), T=T, r=r,                         │
  │               sigma=np.array([sigma]),                                        │
  │               correlation=np.array([[1.0]]),                                  │
  │               n_simulations=req.n_simulations,                                │
  │           )                                                                   │
  │  price               = result.price                                           │
  │  std_error           = result.std_error                                       │
  │  confidence_interval = list(result.confidence_interval)                       │
  │                                                                               │
  │  Always compute full Greeks via BS: ──────────────────────────────────────── │
  │  greeks = {                                                                   │
  │      "delta": BlackScholesModel.delta(S, K, T, r, sigma, option_type),       │
  │      "gamma": BlackScholesModel.gamma(S, K, T, r, sigma),                    │
  │      "vega":  BlackScholesModel.vega(S, K, T, r, sigma),                     │
  │      "theta": BlackScholesModel.theta(S, K, T, r, sigma, option_type),       │
  │      "rho":   BlackScholesModel.rho(S, K, T, r, sigma, option_type),         │
  │  }                                                                            │
  │                                                                               │
  │  return OptionPricingResponse(price, std_error, confidence_interval, greeks) │
  │                                                                               │
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 10 — `tests/test_api.py`  [P5-13]

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  test_api.py   (uses FastAPI TestClient — no real HTTP, no network)          │
  │                                                                               │
  │  from fastapi.testclient import TestClient                                   │
  │  from main import app                                                         │
  │  client = TestClient(app)                                                     │
  │                                                                               │
  │  ── Shared test fixture ───────────────────────────────────────────────────── │
  │                                                                               │
  │  BACKTEST_BODY = {                                                            │
  │      "strategy_id":   "EqualWeightStrategy",                                 │
  │      "symbols":       ["A", "B"],                                             │
  │      "start_date":    "2022-01-01",                                           │
  │      "end_date":      "2022-12-31",                                           │
  │      "initial_value": 100_000,                                                │
  │      "data_source":   "simulated",                                            │
  │      "params":        {},                                                     │
  │  }                                                                            │
  │                                                                               │
  │  ── Tests ─────────────────────────────────────────────────────────────────── │
  │                                                                               │
  │  [P5-13a] GET /api/strategies  → 200  len(strategies) >= 6                  │
  │  [P5-13b] GET /api/strategies/EqualWeightStrategy                            │
  │                               → 200  id=="EqualWeightStrategy"               │
  │                                       family=="allocation"                    │
  │  [P5-13c] GET /api/strategies/nonexistent_xyz  → 404                         │
  │  [P5-13d] POST /api/backtest   → 200  "portfolio_values" in body             │
  │                                        "risk_metrics" in body                 │
  │                                        "trades_log" in body                   │
  │  [P5-13e] POST /api/backtest with unknown strategy_id  → 400                 │
  │  [P5-13f] POST /api/backtest/compare (2 strategies)    → 200  len(resp)==2   │
  │  [P5-13g] POST /api/pricing/option (BS ATM params)     → 200                 │
  │               S=100, K=100, T=1, r=0.05, sigma=0.2 → price ≈ 10.45          │
  │  [P5-13h] POST /api/risk/analyze  → 200  all of:                             │
  │               "sharpe_ratio", "max_drawdown", "var_95" in resp               │
  │  [P5-13i] GET /api/data/assets  → 200  len(assets) > 0                      │
  │                                                                               │
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

## Error Handling Decision Tree

```
  Request arrives
      │
      ├── Pydantic validation fails (wrong types, missing fields, validator errors)
      │       → FastAPI auto-returns HTTP 422 Unprocessable Entity
      │         body: {"detail": [{loc, msg, type}, …]}
      │
      ├── Business logic error (known category)
      │       → route raises HTTPException manually
      │       │
      │       ├── Unknown strategy_id         → HTTP 400
      │       ├── start_date >= end_date      → HTTP 400
      │       ├── Yahoo finance fetch failure → HTTP 502
      │       └── Strategy creation fails     → HTTP 400
      │
      └── Unexpected / unhandled exception
              → global_error_handler in main.py catches it
              → HTTP 500 {"error": str(exc), "type": type(exc).__name__}
```

---

## NaN Serialization Pattern

```
  Problem: Python float("nan") is not valid JSON.
  json.dumps({"x": float("nan")})  →  ValueError

  FastAPI's default JSONResponse calls json.dumps internally → crash.

  Solution: sanitize risk_metrics before returning:

  def _sanitize(d: dict) -> dict:
      return {
          k: (None if isinstance(v, float) and math.isnan(v) else v)
          for k, v in d.items()
      }

  Applied to: risk_metrics in BacktestResponse + /risk/analyze response.
  Not needed for: portfolio_values (always finite), trades_log (ints/floats/strings).

  Frontend interpretation: None (JSON null) → display as "—" or "N/A"
```

---

## Data Source Selection Pattern

```
  data_source field in requests → provider selection in routes

  "simulated"
  ───────────
  SimulatedDataProvider(
      spots        = {sym: 100.0 for sym in symbols},
      volatilities = {sym: 0.20  for sym in symbols},
      correlation  = np.eye(len(symbols)),
      drift        = 0.0,
      risk_free_rate = 0.05,
  )
  ← deterministic, reproducible, no internet
  ← used for all tests and /backtest with data_source="simulated"

  "yahoo"
  ───────
  YahooDataProvider()
  ← real historical prices via yfinance
  ← used in production / demos
  ← may fail if no internet → route catches and raises HTTP 502
```

---

## Full Endpoint Table

```
  Method  Path                        Auth  Body / Query         Response
  ──────  ──────────────────────────  ────  ───────────────────  ─────────────────────
  GET     /                           none  —                    {"status":"ok"}
  GET     /api/strategies             none  —                    {"strategies":[…]}
  GET     /api/strategies/{id}        none  —                    StrategyInfo | 404
  POST    /api/backtest               none  BacktestRequest      BacktestResponse
  POST    /api/backtest/compare       none  CompareRequest       list[BacktestResponse]
  POST    /api/hedging/simulate       none  HedgingRequest       HedgingResponse
  POST    /api/risk/analyze           none  RiskAnalyzeRequest   dict (metrics)
  GET     /api/data/assets            none  —                    {"assets":[…]}
  GET     /api/data/prices            none  symbols,start,end    {"prices":{…}}
  POST    /api/pricing/option         none  OptionPricingRequest OptionPricingResponse
  GET     /docs                       none  —                    Swagger UI (HTML)
```

---

## Cross-Phase Dependency Map for Phase 5

```
  Phase 5 (api/)
       │
       ├──────────────────────────────────────────────► Phase 2 (strategies/)
       │    routes/strategies.py uses StrategyRegistry
       │    routes/backtest.py   uses StrategyRegistry.create()
       │    routes/hedging.py    uses DeltaHedgeStrategy directly
       │
       ├──────────────────────────────────────────────► Phase 3 (backtester/)
       │    routes/backtest.py  uses BacktestConfig, BacktestEngine
       │    routes/hedging.py   uses BacktestConfig, BacktestEngine
       │
       ├──────────────────────────────────────────────► Phase 3 (data/)
       │    routes/backtest.py  uses YahooDataProvider / SimulatedDataProvider
       │    routes/data.py      uses YahooDataProvider, DEFAULT_UNIVERSE
       │
       ├──────────────────────────────────────────────► Phase 4 (risk/)
       │    routes/risk.py      uses PerformanceMetrics.compute_all()
       │    (risk_metrics already in BacktestResult from Phase 4 wiring)
       │
       └──────────────────────────────────────────────► Phase 1 (pricing/ + models/)
            routes/pricing.py  uses BlackScholesModel, MonteCarloPricer, BasketOption
            routes/hedging.py  uses MonteCarloPricer (for initial option price)

  Phases 1–4 → Phase 5: NONE
  (core/ has zero knowledge of FastAPI — correct layering)
```

---

## What This Means for Phase 6

Phase 6 adds the React frontend. It will:

1. **Call** all 10 endpoints above via Axios from `frontend/src/api/`
2. **Display** `BacktestResponse.portfolio_values` as a time-series chart (Recharts)
3. **Display** `risk_metrics` as a stats table
4. **Display** `weights_history` as a stacked bar / area chart
5. **Use** `GET /api/strategies` to populate strategy selector dropdown
6. **Use** `POST /api/pricing/option` for the options pricer page

No changes to Phase 5 code needed — the API contract is stable.
