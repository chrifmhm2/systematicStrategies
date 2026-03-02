"""
Phase 4 tests — Risk Analytics
[P4-25a] through [P4-25h]
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from core.risk.greeks import GreeksCalculator
from core.risk.metrics import PerformanceMetrics
from core.risk.var import VaRCalculator


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def rising_series() -> pd.Series:
    """Monotonically increasing portfolio NAV from 100 to 200 (101 points)."""
    return pd.Series(range(100, 201), dtype=float)


@pytest.fixture
def random_returns() -> pd.Series:
    """Reproducible random daily returns — 500 observations."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.001, 0.02, 500))


# ── [P4-25a] Total return ─────────────────────────────────────────────────────


def test_total_return_doubles(rising_series: pd.Series) -> None:
    """A series from 100 to 200 must give total_return == 1.0 (100% gain)."""
    assert PerformanceMetrics.total_return(rising_series) == pytest.approx(1.0)


# ── [P4-25b] Max drawdown ─────────────────────────────────────────────────────


def test_max_drawdown_no_drawdown(rising_series: pd.Series) -> None:
    """A monotonically increasing series has zero drawdown."""
    assert PerformanceMetrics.max_drawdown(rising_series) == pytest.approx(0.0)


def test_max_drawdown_fifty_percent() -> None:
    """A peak of 200 followed by a drop to 100 gives a 50% drawdown."""
    values = pd.Series([100.0, 200.0, 100.0])
    assert PerformanceMetrics.max_drawdown(values) == pytest.approx(-0.5)


# ── [P4-25c] Sharpe with zero volatility ──────────────────────────────────────


def test_sharpe_ratio_zero_vol_returns_nan() -> None:
    """A constant portfolio value has zero volatility — Sharpe must be NaN, not crash."""
    constant = pd.Series([100.0] * 100)
    result = PerformanceMetrics.sharpe_ratio(constant)
    assert math.isnan(result)


# ── [P4-25d] VaR ordering ─────────────────────────────────────────────────────


def test_var_99_more_negative_than_var_95(random_returns: pd.Series) -> None:
    """99% VaR captures a more extreme tail than 95% VaR — it must be more negative."""
    var_95 = VaRCalculator.historical_var(random_returns, confidence=0.95)
    var_99 = VaRCalculator.historical_var(random_returns, confidence=0.99)
    assert var_99 < var_95


# ── [P4-25e] CVaR ≤ VaR ──────────────────────────────────────────────────────


def test_cvar_more_negative_than_or_equal_to_var(random_returns: pd.Series) -> None:
    """CVaR is the average of returns below VaR, so it is always ≤ VaR."""
    var_95 = VaRCalculator.historical_var(random_returns, confidence=0.95)
    cvar_95 = VaRCalculator.historical_cvar(random_returns, confidence=0.95)
    assert cvar_95 <= var_95


# ── [P4-25f] Rolling VaR length ───────────────────────────────────────────────


def test_rolling_var_same_length(random_returns: pd.Series) -> None:
    """rolling_var must return a Series of the same length as input."""
    window = 100
    result = VaRCalculator.rolling_var(random_returns, window=window)
    assert len(result) == len(random_returns)


def test_rolling_var_nan_prefix(random_returns: pd.Series) -> None:
    """The first window-1 entries must be NaN (not enough history)."""
    window = 100
    result = VaRCalculator.rolling_var(random_returns, window=window)
    assert result.iloc[: window - 1].isna().all()
    assert result.iloc[window:].notna().all()


# ── [P4-25g] compute_all required keys ───────────────────────────────────────


def test_compute_all_has_required_keys() -> None:
    """compute_all must return a dict containing every key used by the API layer."""
    values = pd.Series(range(100, 400), dtype=float)
    required_keys = {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "var_95",
        "cvar_95",
        "win_rate",
        "turnover",
    }
    result = PerformanceMetrics.compute_all(values)
    assert required_keys.issubset(result.keys())


# ── [P4-25h] Greeks surface ───────────────────────────────────────────────────


def test_greeks_surface_keys_and_delta_range() -> None:
    """Surface dict must have the right keys; call deltas must be in [0, 1]."""
    spot_range = np.linspace(80, 120, 5)
    vol_range = np.linspace(0.1, 0.4, 4)

    result = GreeksCalculator.compute_greeks_surface(
        spot_range=spot_range,
        vol_range=vol_range,
        strike=100.0,
        maturity=1.0,
        risk_free_rate=0.05,
    )

    assert set(result.keys()) == {"spots", "vols", "delta", "gamma", "vega", "theta"}

    delta = np.array(result["delta"])
    assert delta.shape == (len(spot_range), len(vol_range))
    assert np.all((delta >= 0) & (delta <= 1)), "Call delta must be in [0, 1]"
