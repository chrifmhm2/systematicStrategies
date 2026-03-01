from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from core.strategies.base import IStrategy, PortfolioWeights, StrategyConfig
from core.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class MaxSharpeStrategy(IStrategy):
    """
    Maximum Sharpe Ratio (Tangency) portfolio.

    Solves:
        max  (ωᵀμ - rf) / sqrt(ωᵀΣω)
        s.t. Σ ωᵢ = 1,  ωᵢ ≥ 0

    where μ = annualised mean returns, Σ = covariance matrix, rf = risk-free rate.

    This is the portfolio on the efficient frontier tangent to the Capital Market Line.
    It maximises return per unit of risk — the best risk-adjusted allocation.
    """

    DESCRIPTION = "Maximum Sharpe ratio (tangency) portfolio"
    FAMILY = "allocation"

    def __init__(
        self,
        config: StrategyConfig,
        lookback_window: int = 60,
        risk_free_rate_override: float | None = None,
    ) -> None:
        super().__init__(config)
        self.lookback_window = lookback_window
        self.risk_free_rate_override = risk_free_rate_override

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
        mu = returns.mean().values * 252      # annualise daily means
        cov = returns.cov().values * 252      # annualise covariance

        rf = self.risk_free_rate_override if self.risk_free_rate_override is not None else 0.05

        # Minimise negative Sharpe ratio
        def neg_sharpe(w: np.ndarray) -> float:
            port_return = float(w @ mu)
            port_vol = float(np.sqrt(w @ cov @ w))
            if port_vol < 1e-10:
                return 0.0
            return -(port_return - rf) / port_vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n
        w0 = np.ones(n) / n

        result = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        weights_arr = result.x if result.success else w0
        weights_arr = np.clip(weights_arr, 0.0, 1.0)
        weights_arr /= weights_arr.sum()

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
                "description": "Rolling window (days) for return and covariance estimation",
            },
            "risk_free_rate_override": {
                "type": "number",
                "default": None,
                "description": "Override risk-free rate (uses provider rate if null)",
            },
            "rebalancing_frequency": {
                "type": "string",
                "default": "monthly",
                "enum": ["daily", "weekly", "monthly"],
            },
        }
