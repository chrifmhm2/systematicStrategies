from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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

    Populated by BacktestEngine in Phase 3.
    Fields are filled incrementally during the backtest loop.
    """

    portfolio_values: dict[str, float] = field(default_factory=dict)
    # date string → portfolio NAV

    benchmark_values: dict[str, float] | None = None
    # date string → benchmark NAV (buy-and-hold equal weight)

    weights_history: dict[str, dict[str, float]] = field(default_factory=dict)
    # date string → {symbol: weight}

    trades_log: list[dict] = field(default_factory=list)
    # one entry per trade: {date, symbol, side, quantity, price, cost}

    risk_metrics: dict = field(default_factory=dict)
    # populated by Phase 4 risk analytics after the loop completes

    computation_time_ms: float = 0.0
