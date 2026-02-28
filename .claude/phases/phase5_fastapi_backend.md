# Phase 5 — FastAPI Backend

> **Goal**: Expose the entire quant engine over HTTP. By the end of this phase you have a running API at `localhost:8000` with auto-generated docs at `/docs`, and every endpoint returns correct data.

**Prerequisites**: Phases 1–4 complete (full quant engine working).

---

## Application Entry Point (`backend/main.py`)

- [ ] **[P5-01]** Create `backend/main.py` with a `FastAPI` app:
  - Title `"QuantForge API"`, version `"1.0.0"`
  - `CORSMiddleware` with `allow_origins=["*"]` (tighten in Phase 7)
  - Mount all routers from `api/routes/` under the `/api` prefix
- [ ] **[P5-02]** Create `backend/config.py` with a `Settings` class (using `pydantic-settings` or plain dataclass):
  - `debug: bool = False`
  - `default_data_source: str = "yahoo"`
  - `max_backtest_years: int = 10`

---

## Pydantic Schemas (`backend/api/schemas.py`)

- [ ] **[P5-03]** Create `backend/api/__init__.py` and `backend/api/routes/__init__.py`
- [ ] **[P5-04]** Create `backend/api/schemas.py` — define all request/response Pydantic models:
  - **[P5-04a]** `BacktestRequest`: `strategy_id`, `symbols`, `start_date`, `end_date`, `initial_value`, `params: dict`, `data_source`
  - **[P5-04b]** `BacktestResponse`: `portfolio_values: dict[str, float]`, `benchmark_values: dict[str, float] | None`, `weights_history: dict[str, dict[str, float]]`, `risk_metrics: dict`, `trades_log: list[dict]`, `computation_time_ms: float`
  - **[P5-04c]** `CompareRequest`: `strategies: list[{strategy_id, params}]`, `symbols`, `start_date`, `end_date`, `initial_value`
  - **[P5-04d]** `HedgingRequest`: `option_type`, `weights`, `symbols`, `strike`, `maturity_years`, `risk_free_rate`, `volatilities`, `correlation_matrix`, `initial_spots`, `n_simulations`, `rebalancing_frequency`, `data_source`, `n_paths`
  - **[P5-04e]** `HedgingResponse`: `paths: list[dict]`, `average_tracking_error`, `initial_option_price`, `initial_option_price_ci`
  - **[P5-04f]** `OptionPricingRequest`: `option_type`, `S`, `K`, `T`, `r`, `sigma`, `method` (`"bs"` or `"mc"`)
  - **[P5-04g]** `OptionPricingResponse`: `price`, `std_error`, `confidence_interval`, `deltas`, `greeks: dict`
  - **[P5-04h]** `RiskAnalyzeRequest`: `portfolio_values: dict[str, float]`, `benchmark_values: dict[str, float] | None`, `risk_free_rate`
  - **[P5-04i]** `StrategyInfo`: `id`, `name`, `family`, `description`, `params: dict`

---

## API Routes

- [ ] **[P5-05]** Create `api/routes/strategies.py`
  - `GET /strategies` → call `StrategyRegistry.list_strategies()` → return `{"strategies": [StrategyInfo]}`
  - `GET /strategies/{strategy_id}` → return single `StrategyInfo` or 404

- [ ] **[P5-06]** Create `api/routes/backtest.py`
  - `POST /backtest` → validate `BacktestRequest`, build `BacktestConfig`, select `data_provider`, run `BacktestEngine.run(strategy, provider)`, convert `BacktestResult` to `BacktestResponse`
  - `POST /backtest/compare` → run `POST /backtest` logic in a loop for each strategy config; return list of `BacktestResponse`
  - For both: return HTTP 400 with a descriptive message if `strategy_id` is unknown or date range is invalid

- [ ] **[P5-07]** Create `api/routes/hedging.py`
  - `POST /hedging/simulate` → validate `HedgingRequest`, instantiate `DeltaHedgeStrategy` with the provided params, run `n_paths` independent simulations (using `SimulatedDataProvider` if `data_source="simulated"`), return `HedgingResponse`

- [ ] **[P5-08]** Create `api/routes/risk.py`
  - `POST /risk/analyze` → accept a `RiskAnalyzeRequest`, compute metrics via `compute_all()`, return the metrics dict

- [ ] **[P5-09]** Create `api/routes/data.py`
  - `GET /data/assets` → return the `DEFAULT_UNIVERSE` list with metadata (symbol, name if available)
  - `GET /data/prices?symbols=AAPL,MSFT&start=2020-01-01&end=2024-12-31` → fetch via `YahooDataProvider`, return as `{"prices": {"AAPL": {"2020-01-02": 300.0, ...}, ...}}`

- [ ] **[P5-10]** Create `api/routes/pricing.py`
  - `POST /pricing/option` → price a single option; if `method="bs"` use `BlackScholesModel`, if `method="mc"` use `MonteCarloPricer`; return `OptionPricingResponse` with full Greeks

---

## Error Handling

- [ ] **[P5-11]** Add a global exception handler in `main.py` that catches unhandled exceptions and returns HTTP 500 with `{"error": str(e), "type": type(e).__name__}`
- [ ] **[P5-12]** For `POST /backtest` and `POST /hedging/simulate`: validate that `start_date < end_date`, that all symbols are non-empty strings, and that `initial_value > 0`; return HTTP 422 with field-level error details (Pydantic validators handle this automatically)

---

## Tests (`backend/tests/`)

- [ ] **[P5-13]** Create `tests/test_api.py` using FastAPI's `TestClient`
  - **[P5-13a]** `GET /api/strategies` returns 200 with a list of at least 6 strategies
  - **[P5-13b]** `GET /api/strategies/EqualWeightStrategy` returns 200 with correct `id` and `family`
  - **[P5-13c]** `GET /api/strategies/nonexistent` returns 404
  - **[P5-13d]** `POST /api/backtest` with `EqualWeightStrategy` on 2 simulated assets returns 200 with `portfolio_values`, `risk_metrics`, and `trades_log`
  - **[P5-13e]** `POST /api/backtest` with unknown `strategy_id` returns 400
  - **[P5-13f]** `POST /api/backtest/compare` with 2 strategies returns a list of 2 results
  - **[P5-13g]** `POST /api/pricing/option` with BS method returns price close to 10.45 for ATM params
  - **[P5-13h]** `POST /api/risk/analyze` returns a dict with `sharpe_ratio`, `max_drawdown`, and `var_95`
  - **[P5-13i]** `GET /api/data/assets` returns a non-empty list

---

## How to Start the Server

```bash
cd backend
uvicorn main:app --reload --port 8000
# Visit http://localhost:8000/docs for the interactive API explorer
```

## How to Run Tests

```bash
cd backend
pytest tests/test_api.py -v
pytest tests/ --cov=. --cov-report=term-missing
```

---

## Definition of Done

- All tests in `[P5-13]` pass
- `uvicorn main:app --reload` starts without errors
- `http://localhost:8000/docs` renders the full OpenAPI UI with all endpoints documented
- A `curl` call to `POST /api/backtest` with a valid body returns a JSON response containing `portfolio_values` and `risk_metrics`
