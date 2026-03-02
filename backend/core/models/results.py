from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from core.backtester.engine import BacktestConfig


@dataclass
class PricingResult:
    """
    Output of the Monte Carlo pricer for a basket option.

    Attributes
    ----------
    price : float
        Discounted expected payoff (the fair value estimate).
    std_error : float
        Standard error of the Monte Carlo estimate.
    confidence_interval : tuple[float, float]
        95% confidence interval (lower, upper).
    deltas : np.ndarray
        Finite-difference delta for each underlying, shape (n_assets,).
    """

    price: float
    std_error: float
    confidence_interval: tuple[float, float]
    deltas: np.ndarray


@dataclass
class BacktestResult:
    """
    Full output of a backtest run.

    Populated by BacktestEngine (Phase 3).
    risk_metrics is filled by Phase 4 risk analytics.
    """

    portfolio_values: pd.Series = field(default_factory=pd.Series)
    # DatetimeIndex → portfolio NAV (float)

    benchmark_values: pd.Series | None = None
    # DatetimeIndex → benchmark NAV (equal-weight buy-and-hold)

    weights_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    # DatetimeIndex (rebalancing dates) × symbol columns → weight at each rebalancing

    trades_log: list[dict] = field(default_factory=list)
    # one entry per trade: {date, symbol, quantity, price, cost}

    risk_metrics: dict = field(default_factory=dict)
    # populated by Phase 4 risk analytics after the loop completes

    config: BacktestConfig | None = None
    # the BacktestConfig used to produce this result

    strategy_name: str = ""
    # name of the strategy that produced this result

    computation_time_ms: float = 0.0
