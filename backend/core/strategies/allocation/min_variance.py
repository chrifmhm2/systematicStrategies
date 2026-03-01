from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from core.strategies.base import IStrategy, PortfolioWeights, StrategyConfig
from core.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class MinVarianceStrategy(IStrategy):
    """
    Minimum Global Variance (MGV) portfolio.

    Solves:
        min  ωᵀ Σ ω
        s.t. Σ ωᵢ = 1,  ωᵢ ≥ 0

    where Σ is the sample covariance matrix estimated from the rolling window.

    Ignores expected returns entirely — only cares about risk.
    Well-suited for risk-averse investors or when return forecasts are unreliable.
    """

    DESCRIPTION = "Minimum variance portfolio — minimises portfolio volatility"
    FAMILY = "allocation"

    def __init__(
        self,
        config: StrategyConfig,
        lookback_window: int = 60,
    ) -> None:
        super().__init__(config)
        self.lookback_window = lookback_window

    @property
    def required_history_days(self) -> int:
        return self.lookback_window + 1

    def compute_weights(
        self,
        current_date: date,
        price_history: pd.DataFrame,
        current_portfolio: object | None = None,
    ) -> PortfolioWeights:
        symbols = list(price_history.columns)
        n = len(symbols)

        returns = price_history.tail(self.lookback_window).pct_change().dropna()
        cov = returns.cov().values

        # Objective: portfolio variance ωᵀ Σ ω
        def portfolio_variance(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n
        w0 = np.ones(n) / n  # start from equal weight

        result = minimize(
            portfolio_variance,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        weights_arr = result.x if result.success else w0
        weights_arr = np.clip(weights_arr, 0.0, 1.0)
        weights_arr /= weights_arr.sum()  # renormalise for numerical safety

        return PortfolioWeights(
            weights={sym: float(weights_arr[i]) for i, sym in enumerate(symbols)},
            cash_weight=0.0,
            timestamp=current_date,
        )

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "lookback_window": {
                "type": "integer",
                "default": 60,
                "description": "Number of trading days used to estimate the covariance matrix",
            },
            "rebalancing_frequency": {
                "type": "string",
                "default": "monthly",
                "enum": ["daily", "weekly", "monthly"],
            },
        }
