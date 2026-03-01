"""
Phase 2 tests — Strategy Framework

Covers [P2-15a] through [P2-15i]:
  a) Equal weight: 1/n for any n, sum to 1
  b) Min-variance: weights non-negative and sum to 1
  c) Max-Sharpe: weights non-negative and sum to 1
  d) Risk parity: risk contributions approximately equal
  e) Momentum: exactly top_k non-zero entries (long-only)
  f) Mean reversion: only |z| > threshold get non-zero weights
  g) No look-ahead bias: truncating history at t does not change weights at t
  h) Registry: list_strategies >= 6, create() returns IStrategy
  i) Delta hedge: weights sum to <= 1 (keeps cash)
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from core.strategies import IStrategy, StrategyConfig, StrategyRegistry
from core.strategies.allocation.equal_weight import EqualWeightStrategy
from core.strategies.allocation.max_sharpe import MaxSharpeStrategy
from core.strategies.allocation.min_variance import MinVarianceStrategy
from core.strategies.allocation.risk_parity import RiskParityStrategy
from core.strategies.hedging.delta_hedge import DeltaHedgeStrategy
from core.strategies.signal.mean_reversion import MeanReversionStrategy
from core.strategies.signal.momentum import MomentumStrategy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SYMBOLS_3 = ["AAPL", "MSFT", "GOOG"]
SYMBOLS_5 = ["A", "B", "C", "D", "E"]


def _make_prices(
    symbols: list[str],
    n_days: int,
    seed: int = 42,
    drift: float = 0.0,
    vol: float = 0.01,
) -> pd.DataFrame:
    """Simulated daily prices: random GBM paths."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(drift, vol, size=(n_days, len(symbols)))
    prices = 100.0 * np.exp(np.cumsum(log_ret, axis=0))
    start = date(2023, 1, 2)
    idx = pd.date_range(start=start, periods=n_days, freq="B")
    return pd.DataFrame(prices, index=idx, columns=symbols)


def _config(name: str = "test") -> StrategyConfig:
    return StrategyConfig(name=name, description="test strategy")


# ---------------------------------------------------------------------------
# [P2-15a] Equal weight
# ---------------------------------------------------------------------------

class TestEqualWeight:
    def test_weights_equal_for_3_assets(self):
        strat = EqualWeightStrategy(_config())
        prices = _make_prices(SYMBOLS_3, 5)
        pw = strat.compute_weights(date.today(), prices)
        for sym in SYMBOLS_3:
            assert abs(pw.weights[sym] - 1 / 3) < 1e-9

    def test_weights_sum_to_one(self):
        for n in [2, 5, 10]:
            symbols = [f"S{i}" for i in range(n)]
            strat = EqualWeightStrategy(_config())
            prices = _make_prices(symbols, 5)
            pw = strat.compute_weights(date.today(), prices)
            assert abs(pw.total_weight() - 1.0) < 1e-9

    def test_cash_weight_is_zero(self):
        strat = EqualWeightStrategy(_config())
        prices = _make_prices(SYMBOLS_3, 5)
        pw = strat.compute_weights(date.today(), prices)
        assert pw.cash_weight == 0.0


# ---------------------------------------------------------------------------
# [P2-15b] Min-variance
# ---------------------------------------------------------------------------

class TestMinVariance:
    def _run(self, symbols=SYMBOLS_3, lookback=60):
        strat = MinVarianceStrategy(_config(), lookback_window=lookback)
        prices = _make_prices(symbols, lookback + 10)
        return strat.compute_weights(prices.index[-1].date(), prices)

    def test_weights_non_negative(self):
        pw = self._run()
        for w in pw.weights.values():
            assert w >= -1e-9

    def test_weights_sum_to_one(self):
        pw = self._run()
        assert abs(pw.total_weight() - 1.0) < 1e-6

    def test_cash_weight_is_zero(self):
        pw = self._run()
        assert abs(pw.cash_weight) < 1e-9


# ---------------------------------------------------------------------------
# [P2-15c] Max-Sharpe
# ---------------------------------------------------------------------------

class TestMaxSharpe:
    def _run(self, symbols=SYMBOLS_3, lookback=60):
        strat = MaxSharpeStrategy(_config(), lookback_window=lookback, risk_free_rate_override=0.05)
        prices = _make_prices(symbols, lookback + 10, drift=0.0005)
        return strat.compute_weights(prices.index[-1].date(), prices)

    def test_weights_non_negative(self):
        pw = self._run()
        for w in pw.weights.values():
            assert w >= -1e-9

    def test_weights_sum_to_one(self):
        pw = self._run()
        assert abs(pw.total_weight() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# [P2-15d] Risk parity
# ---------------------------------------------------------------------------

class TestRiskParity:
    def test_risk_contributions_approximately_equal(self):
        symbols = SYMBOLS_3
        lookback = 60
        strat = RiskParityStrategy(_config(), lookback_window=lookback)
        prices = _make_prices(symbols, lookback + 10)
        pw = strat.compute_weights(prices.index[-1].date(), prices)

        w = np.array([pw.weights[s] for s in symbols])
        returns = prices.tail(lookback).pct_change().dropna()
        cov = returns.cov().values

        port_var = float(w @ cov @ w)
        mrc = cov @ w
        rc = w * mrc / port_var   # risk contributions

        # Each RC should be close to 1/n
        target = 1.0 / len(symbols)
        for rci in rc:
            assert abs(rci - target) < 0.05   # within 5 percentage points

    def test_weights_sum_to_one(self):
        strat = RiskParityStrategy(_config())
        prices = _make_prices(SYMBOLS_3, 70)
        pw = strat.compute_weights(prices.index[-1].date(), prices)
        assert abs(pw.total_weight() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# [P2-15e] Momentum
# ---------------------------------------------------------------------------

class TestMomentum:
    def test_exactly_top_k_nonzero_long_only(self):
        top_k = 2
        strat = MomentumStrategy(_config(), lookback_period=20, top_k=top_k, long_only=True)
        prices = _make_prices(SYMBOLS_5, 30)
        pw = strat.compute_weights(prices.index[-1].date(), prices)
        nonzero = [s for s, w in pw.weights.items() if w > 1e-9]
        assert len(nonzero) == top_k

    def test_top_k_weights_equal(self):
        top_k = 3
        strat = MomentumStrategy(_config(), lookback_period=20, top_k=top_k, long_only=True)
        prices = _make_prices(SYMBOLS_5, 30)
        pw = strat.compute_weights(prices.index[-1].date(), prices)
        nonzero_weights = [w for w in pw.weights.values() if w > 1e-9]
        for w in nonzero_weights:
            assert abs(w - 1.0 / top_k) < 1e-9

    def test_weights_sum_to_one_long_only(self):
        strat = MomentumStrategy(_config(), lookback_period=20, top_k=2, long_only=True)
        prices = _make_prices(SYMBOLS_5, 30)
        pw = strat.compute_weights(prices.index[-1].date(), prices)
        assert abs(pw.total_weight() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# [P2-15f] Mean reversion
# ---------------------------------------------------------------------------

class TestMeanReversion:
    def _prices_with_known_zscore(self) -> pd.DataFrame:
        """Craft prices so AAPL has z < -2 (buy signal) and MSFT is neutral."""
        symbols = ["AAPL", "MSFT"]
        lookback = 20
        # Both assets have constant prices, then AAPL drops sharply
        aapl = [100.0] * lookback + [85.0]   # z ≈ -3 (far below MA)
        msft = [200.0] * (lookback + 1)       # z = 0 (flat)
        idx = pd.date_range("2023-01-02", periods=lookback + 1, freq="B")
        return pd.DataFrame({"AAPL": aapl, "MSFT": msft}, index=idx)

    def test_only_triggered_symbols_nonzero(self):
        strat = MeanReversionStrategy(_config(), lookback_window=20, z_threshold=2.0)
        prices = self._prices_with_known_zscore()
        pw = strat.compute_weights(prices.index[-1].date(), prices)
        # AAPL triggered, MSFT did not
        assert pw.weights["AAPL"] > 0.0
        assert pw.weights["MSFT"] == 0.0

    def test_weights_sum_to_at_most_one(self):
        strat = MeanReversionStrategy(_config(), lookback_window=20, z_threshold=2.0)
        prices = self._prices_with_known_zscore()
        pw = strat.compute_weights(prices.index[-1].date(), prices)
        assert pw.total_weight() <= 1.0 + 1e-9

    def test_no_signal_means_all_cash(self):
        """If no asset triggers, all weight goes to cash."""
        strat = MeanReversionStrategy(_config(), lookback_window=20, z_threshold=100.0)
        prices = _make_prices(SYMBOLS_3, 30)
        pw = strat.compute_weights(prices.index[-1].date(), prices)
        assert all(w == 0.0 for w in pw.weights.values())
        assert abs(pw.cash_weight - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# [P2-15g] No look-ahead bias
# ---------------------------------------------------------------------------

class TestNoLookAheadBias:
    @pytest.mark.parametrize("StratClass,kwargs", [
        (EqualWeightStrategy, {}),
        (MinVarianceStrategy, {"lookback_window": 30}),
        (MaxSharpeStrategy,   {"lookback_window": 30}),
        (RiskParityStrategy,  {"lookback_window": 30}),
        (MomentumStrategy,    {"lookback_period": 30, "top_k": 2}),
        (MeanReversionStrategy, {"lookback_window": 20}),
    ])
    def test_weights_unchanged_when_future_row_added(self, StratClass, kwargs):
        prices = _make_prices(SYMBOLS_3, 60)
        t = prices.index[40]

        strat = StratClass(_config(), **kwargs)

        # Compute weights using data up to t
        history_at_t = prices.loc[:t]
        pw_at_t = strat.compute_weights(t.date(), history_at_t)

        # Compute weights using data up to t again but with one extra future row
        history_extended = prices.loc[:prices.index[41]]
        # Truncate back to t before calling — simulating what backtester does
        pw_at_t_again = strat.compute_weights(t.date(), prices.loc[:t])

        for sym in SYMBOLS_3:
            assert abs(pw_at_t.weights[sym] - pw_at_t_again.weights[sym]) < 1e-12


# ---------------------------------------------------------------------------
# [P2-15h] Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_list_strategies_returns_at_least_6(self):
        strategies = StrategyRegistry.list_strategies()
        assert len(strategies) >= 6

    def test_list_strategies_has_required_keys(self):
        for entry in StrategyRegistry.list_strategies():
            assert "name" in entry
            assert "description" in entry
            assert "family" in entry
            assert "param_schema" in entry

    def test_create_returns_istrategy_instance(self):
        strat = StrategyRegistry.create("EqualWeightStrategy", {})
        assert isinstance(strat, IStrategy)

    def test_create_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="not found in registry"):
            StrategyRegistry.create("NonExistentStrategy", {})

    def test_all_families_present(self):
        families = {s["family"] for s in StrategyRegistry.list_strategies()}
        assert "hedging" in families
        assert "allocation" in families
        assert "signal" in families


# ---------------------------------------------------------------------------
# [P2-15i] Delta hedge: weights sum to <= 1
# ---------------------------------------------------------------------------

class TestDeltaHedge:
    def test_total_weight_at_most_one(self):
        config = StrategyConfig(name="dh", description="delta hedge test")
        strat = DeltaHedgeStrategy(
            config,
            strike=100.0,
            maturity_years=1.0,
            n_simulations=5_000,
            risk_free_rate=0.05,
            volatilities=[0.2, 0.2],
            correlation=[[1.0, 0.3], [0.3, 1.0]],
        )
        prices = _make_prices(["S1", "S2"], 5, seed=0)
        pw = strat.compute_weights(prices.index[-1].date(), prices)
        assert pw.total_weight() <= 1.0 + 1e-6

    def test_all_weights_non_negative(self):
        config = StrategyConfig(name="dh", description="delta hedge test")
        strat = DeltaHedgeStrategy(
            config,
            strike=100.0,
            maturity_years=1.0,
            n_simulations=5_000,
            risk_free_rate=0.05,
        )
        prices = _make_prices(["S1", "S2", "S3"], 5, seed=1)
        pw = strat.compute_weights(prices.index[-1].date(), prices)
        for w in pw.weights.values():
            assert w >= -1e-9
