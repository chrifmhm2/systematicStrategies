# Phase 1 — Core Engine Foundation

> **Goal**: Build the pure Python quant engine with no web dependencies. By the end of this phase you have a working option pricer and data layer you can call from a Python REPL or Jupyter notebook.

---

## Repository Bootstrap

- [ ] **[P1-01]** Create the `backend/` directory at the repo root
- [ ] **[P1-02]** Create `backend/pyproject.toml` with project name `quantforge`, Python `>=3.12`, runtime deps (`numpy`, `scipy`, `pandas`, `yfinance`), and dev deps (`pytest`, `pytest-cov`, `ruff`)
- [ ] **[P1-03]** Create `backend/core/__init__.py` (empty)
- [ ] **[P1-04]** Create `backend/tests/__init__.py` (empty)
- [ ] **[P1-05]** Verify `pip install -e ".[dev]"` (or `uv pip install -e ".[dev]"`) runs without errors

---

## Data Models (`backend/core/models/`)

- [ ] **[P1-06]** Create `core/models/__init__.py`
- [ ] **[P1-07]** Create `core/models/market_data.py`
  - `DataFeed` dataclass: `symbol: str`, `date: date`, `price: float`
  - `OHLCV` dataclass: `symbol`, `date`, `open`, `high`, `low`, `close`, `volume`
- [ ] **[P1-08]** Create `core/models/portfolio.py`
  - `Position` dataclass: `symbol: str`, `quantity: float`, `price: float`
  - `Portfolio` dataclass: `positions: dict[str, Position]`, `cash: float`, `date: date` with method `total_value(prices: dict[str, float]) -> float`
- [ ] **[P1-09]** Create `core/models/options.py`
  - `VanillaOption` dataclass: `underlying: str`, `strike: float`, `maturity: float`, `option_type: str` (`"call"` or `"put"`)
  - `BasketOption` dataclass: `underlyings: list[str]`, `weights: list[float]`, `strike: float`, `maturity: float`
- [ ] **[P1-10]** Create `core/models/results.py`
  - `PricingResult` dataclass: `price: float`, `std_error: float`, `confidence_interval: tuple[float, float]`, `deltas: np.ndarray`
  - `BacktestResult` as a stub (empty class body with `pass`) — will be completed in Phase 3

---

## Data Layer (`backend/core/data/`)

- [ ] **[P1-11]** Create `core/data/__init__.py`
- [ ] **[P1-12]** Create `core/data/base.py` — abstract `IDataProvider(ABC)` with two abstract methods:
  - `get_prices(symbols, start_date, end_date) -> pd.DataFrame` (DatetimeIndex, one column per symbol = adjusted close)
  - `get_risk_free_rate(date) -> float`
- [ ] **[P1-13]** Create `core/data/simulated.py` — `SimulatedDataProvider(IDataProvider)`
  - Constructor: `spots: dict[str, float]`, `volatilities: dict[str, float]`, `correlation: np.ndarray`, `drift: float = 0.0`, `risk_free_rate: float = 0.05`, `seed: int | None = None`
  - `get_prices`: generate daily GBM paths via Cholesky decomposition — formula: `S_t = S_0 * exp((μ - σ²/2)*dt + σ*√dt*Z)` where `Z` is drawn from the correlated normal
  - `get_risk_free_rate`: return the fixed rate from construction
- [ ] **[P1-14]** Create `core/data/csv_loader.py` — `CsvDataProvider(IDataProvider)`
  - Load CSV with columns `Id`, `DateOfPrice`, `Value` (same format as the original Ensimag project)
  - `get_risk_free_rate`: return a hardcoded fallback (e.g. `0.05`)

---

## Pricing Utilities (`backend/core/pricing/`)

- [ ] **[P1-15]** Create `core/pricing/__init__.py`
- [ ] **[P1-16]** Create `core/pricing/utils.py`
  - `cholesky_decompose(correlation: np.ndarray) -> np.ndarray` — thin wrapper around `np.linalg.cholesky` with a positive-definite guard
  - `generate_correlated_normals(n_assets: int, n_samples: int, chol: np.ndarray, seed: int | None) -> np.ndarray` — returns shape `(n_samples, n_assets)`

---

## Black-Scholes Model (`core/pricing/black_scholes.py`)

- [ ] **[P1-17]** Create the file and `BlackScholesModel` class (all static methods)
- [ ] **[P1-18]** Implement `call_price(S, K, T, r, sigma) -> float` using the closed-form formula via `scipy.stats.norm.cdf`
- [ ] **[P1-19]** Implement `put_price(S, K, T, r, sigma) -> float` using put-call parity: `C - S + K*exp(-rT)`
- [ ] **[P1-20]** Implement `delta(S, K, T, r, sigma, option_type) -> float` — `N(d1)` for call, `N(d1)-1` for put
- [ ] **[P1-21]** Implement `gamma(S, K, T, r, sigma) -> float` — `n(d1) / (S * sigma * sqrt(T))`
- [ ] **[P1-22]** Implement `vega(S, K, T, r, sigma) -> float` — `S * n(d1) * sqrt(T)`
- [ ] **[P1-23]** Implement `theta(S, K, T, r, sigma, option_type) -> float`
- [ ] **[P1-24]** Implement `rho(S, K, T, r, sigma, option_type) -> float`
- [ ] **[P1-25]** Implement `implied_volatility(price, S, K, T, r, option_type) -> float` — Newton-Raphson with `vega` as the derivative; raise `ValueError` if it fails to converge

---

## Monte Carlo Pricer (`core/pricing/monte_carlo.py`)

- [ ] **[P1-26]** Create the file and `MonteCarloPricer` class
- [ ] **[P1-27]** Implement `__init__(n_simulations=100_000, seed=None, variance_reduction="antithetic")`
- [ ] **[P1-28]** Implement `price_basket_option(spots, weights, strike, maturity, risk_free_rate, volatilities, correlation) -> PricingResult`
  - Simulate terminal prices: `S_i^T = S_i * exp((r - σ_i²/2)*T + σ_i*√T * Z_i)`
  - Basket payoff: `max(Σ ω_i * S_i^T - K, 0)`
  - Antithetic variates: draw `Z`, compute payoff with `+Z` and `-Z`, average before taking the mean across simulations
  - Discount: `price = mean(payoffs) * exp(-r*T)`
  - Fill `std_error`, `confidence_interval` (95%), and `deltas` (call `compute_deltas` internally)
- [ ] **[P1-29]** Implement `compute_deltas(spots, weights, strike, maturity, risk_free_rate, volatilities, correlation, bump_size=0.01) -> np.ndarray`
  - Central finite difference per asset: `Δ_i = (V(S_i+bump) - V(S_i-bump)) / (2 * bump * S_i)`
  - Reuse the same random seed for up/down bumps to reduce noise

---

## Tests (`backend/tests/`)

- [ ] **[P1-30]** Create `tests/test_black_scholes.py`
  - **[P1-30a]** ATM call: `S=100, K=100, T=1, r=0.05, σ=0.2` → price ≈ 10.45 (tolerance 0.01)
  - **[P1-30b]** Put-call parity holds: `C - P = S - K*exp(-rT)` for 5 different parameter sets
  - **[P1-30c]** Delta of a call is in `[0, 1]`; delta of a put is in `[-1, 0]`
  - **[P1-30d]** Gamma and vega are always positive
  - **[P1-30e]** Implied vol round-trip: compute price → extract IV → recompute price → match within `1e-4`
  - **[P1-30f]** Each Greek matches central finite-difference approximation within `1e-3`
- [ ] **[P1-31]** Create `tests/test_monte_carlo.py`
  - **[P1-31a]** Single-asset degenerate basket (weight=1, 1 asset): MC price ≈ BS price within 2 standard errors
  - **[P1-31b]** Antithetic variates produce lower `std_error` than plain MC (`variance_reduction="none"`) for the same `n_simulations`
  - **[P1-31c]** Basket call price is strictly positive for an ITM option
  - **[P1-31d]** All deltas are in `[0, 1]` for a basket call
  - **[P1-31e]** Price is roughly consistent across two different seeds (within 3 std errors)

---

## How to Run Tests

```bash
cd backend
pip install -e ".[dev]"
pytest tests/test_black_scholes.py tests/test_monte_carlo.py -v
pytest tests/ --cov=core --cov-report=term-missing
```

---

## Definition of Done

- All tests in `[P1-30]` and `[P1-31]` pass
- Coverage on `core/pricing/` and `core/data/` ≥ 80%
- The following REPL snippet runs and prints sensible values:

```python
from core.pricing.black_scholes import BlackScholesModel
from core.pricing.monte_carlo import MonteCarloPricer
import numpy as np

print(BlackScholesModel.call_price(100, 100, 1, 0.05, 0.2))  # ~10.45

pricer = MonteCarloPricer(n_simulations=50_000, seed=42)
result = pricer.price_basket_option(
    spots=np.array([100.0]),
    weights=np.array([1.0]),
    strike=100.0,
    maturity=1.0,
    risk_free_rate=0.05,
    volatilities=np.array([0.2]),
    correlation=np.array([[1.0]]),
)
print(result.price, result.std_error)  # ~10.45, small std_error
```
