from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from core.strategies.base import IStrategy, PortfolioWeights, StrategyConfig
from core.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class RiskParityStrategy(IStrategy):
    """
    Risk Parity (Equal Risk Contribution) portfolio.

    Each asset contributes equally to total portfolio variance:
        RC_i = ωᵢ · (Σω)ᵢ / (ωᵀΣω)  for all i

    Objective:
        min  Σᵢ (RC_i - 1/n)²

    Unlike min-variance, risk parity does not concentrate in the lowest-vol
    asset — it diversifies risk across the whole universe.
    """

    DESCRIPTION = "Risk parity — each asset contributes equally to portfolio risk"
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

        target_rc = 1.0 / n  # each asset contributes 1/n of total risk

        def risk_parity_objective(w: np.ndarray) -> float:
            port_var = float(w @ cov @ w)
            if port_var < 1e-12:
                return 0.0
            # Marginal risk contributions: (Σω)ᵢ
            mrc = cov @ w
            # Risk contributions: RC_i = ωᵢ · MRC_i / port_var
            rc = w * mrc / port_var
            return float(np.sum((rc - target_rc) ** 2))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(1e-6, 1.0)] * n  # strictly positive — avoids division by zero in RC
        w0 = np.ones(n) / n

        result = minimize(
            risk_parity_objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 2000},
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
                "description": "Rolling window (days) for covariance estimation",
            },
            "rebalancing_frequency": {
                "type": "string",
                "default": "monthly",
                "enum": ["daily", "weekly", "monthly"],
            },
        }
