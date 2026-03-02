"""
Tests for Phase 3 — BacktestEngine, RebalancingOracles, TransactionCostModel.

Labels: [P3-15a] through [P3-15g]
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from core.backtester.costs import TransactionCostModel
from core.backtester.engine import BacktestConfig, BacktestEngine
from core.backtester.rebalancing import PeriodicRebalancing, ThresholdRebalancing
from core.data.simulated import SimulatedDataProvider
from core.models.portfolio import Portfolio, Position
from core.strategies.allocation.equal_weight import EqualWeightStrategy
from core.strategies.base import StrategyConfig

# ── Shared fixtures ────────────────────────────────────────────────────────────

SYMBOLS = ["AAPL", "MSFT"]
START = date(2023, 1, 2)
END = date(2023, 6, 30)


def _make_provider(seed: int = 0) -> SimulatedDataProvider:
    return SimulatedDataProvider(
        spots={"AAPL": 150.0, "MSFT": 280.0},
        volatilities={"AAPL": 0.20, "MSFT": 0.25},
        correlation=np.array([[1.0, 0.5], [0.5, 1.0]]),
        seed=seed,
    )


def _make_strategy(name: str = "EW") -> EqualWeightStrategy:
    return EqualWeightStrategy(StrategyConfig(name=name, description=""))


def _make_config(**overrides) -> BacktestConfig:
    defaults = dict(
        initial_value=100_000.0,
        start_date=START,
        end_date=END,
        symbols=SYMBOLS,
        rebalancing_frequency="weekly",
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


# ── [P3-15a] Self-financing constraint ─────────────────────────────────────────


class TestSelfFinancing:
    """
    [P3-15a] The backtest engine must be self-financing:
    portfolio value immediately after rebalancing = value before − transaction costs.

    The BacktestEngine itself asserts this invariant on every rebalancing step
    (see the `assert` in engine.py). So if it runs without error, the constraint holds.
    """

    def test_backtest_runs_without_self_financing_violation(self):
        provider = _make_provider(seed=1)
        strategy = _make_strategy()
        config = _make_config()
        engine = BacktestEngine(config)
        result = engine.run(strategy, provider)
        # If we reach here, no AssertionError was raised → invariant holds.
        assert len(result.portfolio_values) > 0

    def test_final_value_less_than_zero_cost_run(self):
        """Portfolio with costs must be worth less than a zero-cost portfolio."""
        provider = _make_provider(seed=2)
        strategy_cost = _make_strategy("EW-cost")
        strategy_free = _make_strategy("EW-free")

        result_cost = BacktestEngine(_make_config(transaction_cost_bps=10, slippage_bps=5)).run(
            strategy_cost, provider
        )
        result_free = BacktestEngine(_make_config(transaction_cost_bps=0, slippage_bps=0)).run(
            strategy_free, provider
        )

        assert result_free.portfolio_values.iloc[-1] >= result_cost.portfolio_values.iloc[-1]


# ── [P3-15b] Transaction costs reduce final value ──────────────────────────────


class TestTransactionCosts:
    """[P3-15b] Zero-cost portfolio has strictly higher final value than non-zero cost."""

    def test_zero_cost_outperforms_nonzero_cost(self):
        provider_cost = _make_provider(seed=42)
        provider_free = _make_provider(seed=42)  # same seed → same prices

        result_cost = BacktestEngine(
            _make_config(transaction_cost_bps=10, slippage_bps=5)
        ).run(_make_strategy("EW-cost"), provider_cost)

        result_free = BacktestEngine(
            _make_config(transaction_cost_bps=0, slippage_bps=0)
        ).run(_make_strategy("EW-free"), provider_free)

        # Same prices, same strategy, only cost differs
        assert result_free.portfolio_values.iloc[-1] > result_cost.portfolio_values.iloc[-1]

    def test_cost_model_basic(self):
        model = TransactionCostModel(commission_bps=10, slippage_bps=5, min_commission=1.0)
        # 15 bps of $10,000 = $15
        assert abs(model.compute_cost(10_000) - 15.0) < 1e-9

    def test_cost_model_min_commission(self):
        model = TransactionCostModel(commission_bps=10, slippage_bps=5, min_commission=1.0)
        # 15 bps of $1 = 0.0015 → floored to min_commission = 1.0
        assert model.compute_cost(1.0) == 1.0


# ── [P3-15c] No look-ahead bias ────────────────────────────────────────────────


class TestNoLookAheadBias:
    """
    [P3-15c] The strategy must never receive price data beyond the current backtest date.

    Approach: wrap the strategy's compute_weights to record the price_history
    DataFrame index on every call, then assert all dates ≤ current_date.
    """

    def test_price_history_never_contains_future_dates(self):
        provider = _make_provider(seed=3)
        real_strategy = _make_strategy()
        received_histories: list[tuple[date, pd.DatetimeIndex]] = []

        # Wrap compute_weights to spy on price_history
        original_compute = real_strategy.compute_weights

        def spy_compute(current_date, price_history, current_portfolio=None):
            received_histories.append((current_date, price_history.index.copy()))
            return original_compute(current_date, price_history, current_portfolio)

        real_strategy.compute_weights = spy_compute  # type: ignore[method-assign]

        config = _make_config()
        BacktestEngine(config).run(real_strategy, provider)

        assert len(received_histories) > 0, "Strategy was never called — adjust date range"
        for call_date, idx in received_histories:
            for ts in idx:
                assert ts.date() <= call_date, (
                    f"Look-ahead bias detected: strategy received {ts.date()} "
                    f"at current_date={call_date}"
                )


# ── [P3-15d] Date range ────────────────────────────────────────────────────────


class TestDateRange:
    """[P3-15d] portfolio_values index is within [start_date, end_date]."""

    def test_portfolio_values_within_date_range(self):
        provider = _make_provider(seed=4)
        result = BacktestEngine(_make_config()).run(_make_strategy(), provider)

        pv = result.portfolio_values
        assert not pv.empty
        assert pv.index[0].date() >= START
        assert pv.index[-1].date() <= END

    def test_portfolio_values_is_series(self):
        provider = _make_provider(seed=5)
        result = BacktestEngine(_make_config()).run(_make_strategy(), provider)
        assert isinstance(result.portfolio_values, pd.Series)

    def test_benchmark_values_is_series(self):
        provider = _make_provider(seed=5)
        result = BacktestEngine(_make_config()).run(_make_strategy(), provider)
        assert isinstance(result.benchmark_values, pd.Series)
        assert len(result.benchmark_values) == len(result.portfolio_values)


# ── [P3-15e] Trades log ────────────────────────────────────────────────────────


class TestTradesLog:
    """[P3-15e] Each trade dict contains the required keys."""

    REQUIRED_KEYS = {"date", "symbol", "quantity", "price", "cost"}

    def test_trades_have_required_keys(self):
        provider = _make_provider(seed=6)
        result = BacktestEngine(_make_config()).run(_make_strategy(), provider)
        assert len(result.trades_log) > 0, "No trades were logged"
        for trade in result.trades_log:
            missing = self.REQUIRED_KEYS - set(trade.keys())
            assert not missing, f"Trade missing keys: {missing}. Trade: {trade}"

    def test_trade_cost_is_positive(self):
        provider = _make_provider(seed=6)
        result = BacktestEngine(_make_config()).run(_make_strategy(), provider)
        for trade in result.trades_log:
            assert trade["cost"] >= 0, f"Negative cost in trade: {trade}"

    def test_trade_price_is_positive(self):
        provider = _make_provider(seed=6)
        result = BacktestEngine(_make_config()).run(_make_strategy(), provider)
        for trade in result.trades_log:
            assert trade["price"] > 0, f"Non-positive price in trade: {trade}"


# ── [P3-15f] Periodic rebalancing — weekly ─────────────────────────────────────


class TestPeriodicRebalancing:
    """[P3-15f] should_rebalance returns True only on Mondays for weekly frequency."""

    def test_weekly_fires_on_monday(self):
        oracle = PeriodicRebalancing("weekly")
        monday = date(2023, 1, 2)   # Monday
        assert oracle.should_rebalance(monday, None) is True

    def test_weekly_does_not_fire_on_other_days(self):
        oracle = PeriodicRebalancing("weekly")
        monday = date(2023, 1, 2)
        oracle.should_rebalance(monday, None)  # consume the Monday

        for days_ahead in range(1, 5):  # Tue–Fri
            other = monday + timedelta(days=days_ahead)
            assert oracle.should_rebalance(other, None) is False, (
                f"Weekly oracle fired on {other} (weekday={other.weekday()})"
            )

    def test_daily_fires_every_day(self):
        oracle = PeriodicRebalancing("daily")
        for i in range(5):
            d = date(2023, 1, 2) + timedelta(days=i)
            assert oracle.should_rebalance(d, None) is True

    def test_monthly_fires_once_per_month(self):
        oracle = PeriodicRebalancing("monthly")
        # First call in January → fires
        jan1 = date(2023, 1, 2)
        assert oracle.should_rebalance(jan1, None) is True
        # Later in January → does not fire
        jan15 = date(2023, 1, 16)
        assert oracle.should_rebalance(jan15, None) is False
        # First call in February → fires
        feb1 = date(2023, 2, 1)
        assert oracle.should_rebalance(feb1, None) is True

    def test_invalid_frequency_raises(self):
        with pytest.raises(ValueError, match="frequency"):
            PeriodicRebalancing("yearly")

    def test_weekly_oracle_integrated_in_backtest(self):
        """Weights history rebalancing dates should all be Mondays (weekday=0)."""
        provider = _make_provider(seed=7)
        result = BacktestEngine(_make_config(rebalancing_frequency="weekly")).run(
            _make_strategy(), provider
        )
        if not result.weights_history.empty:
            for ts in result.weights_history.index:
                assert ts.weekday() == 0, (
                    f"Rebalancing happened on non-Monday: {ts} (weekday={ts.weekday()})"
                )


# ── [P3-15g] Threshold rebalancing ────────────────────────────────────────────


class TestThresholdRebalancing:
    """[P3-15g] Threshold oracle fires only when drift exceeds threshold."""

    def _make_portfolio(self, aapl_w: float, msft_w: float, total: float = 100_000.0):
        aapl_price, msft_price = 150.0, 280.0
        aapl_qty = aapl_w * total / aapl_price
        msft_qty = msft_w * total / msft_price
        return Portfolio(
            positions={
                "AAPL": Position("AAPL", aapl_qty, aapl_price),
                "MSFT": Position("MSFT", msft_qty, msft_price),
            },
            cash=0.0,
        )

    def test_no_rebalance_when_within_threshold(self):
        oracle = ThresholdRebalancing(threshold=0.05)
        oracle.update_target_weights({"AAPL": 0.50, "MSFT": 0.50})

        # Portfolio at exact target weights → drift = 0 → no rebalance
        portfolio = self._make_portfolio(aapl_w=0.50, msft_w=0.50)
        assert oracle.should_rebalance(date(2023, 6, 1), portfolio) is False

    def test_rebalance_when_drift_exceeds_threshold(self):
        oracle = ThresholdRebalancing(threshold=0.05)
        oracle.update_target_weights({"AAPL": 0.50, "MSFT": 0.50})

        # AAPL has drifted to 62% (12pp drift > 5% threshold) → rebalance
        portfolio = self._make_portfolio(aapl_w=0.62, msft_w=0.38)
        assert oracle.should_rebalance(date(2023, 6, 1), portfolio) is True

    def test_first_call_always_rebalances(self):
        oracle = ThresholdRebalancing(threshold=0.05)
        # No target weights set → first call must return True
        portfolio = self._make_portfolio(0.5, 0.5)
        assert oracle.should_rebalance(date(2023, 1, 2), portfolio) is True

    def test_threshold_oracle_integrated_in_backtest(self):
        """With a very tight threshold, the engine should trigger many rebalances."""
        provider_tight = _make_provider(seed=8)
        provider_loose = _make_provider(seed=8)

        result_tight = BacktestEngine(
            _make_config(rebalancing_frequency="threshold", threshold=0.001)
        ).run(_make_strategy("tight"), provider_tight)

        result_loose = BacktestEngine(
            _make_config(rebalancing_frequency="threshold", threshold=0.99)
        ).run(_make_strategy("loose"), provider_loose)

        # Tight threshold → more rebalances → more trades
        assert len(result_tight.trades_log) > len(result_loose.trades_log)
