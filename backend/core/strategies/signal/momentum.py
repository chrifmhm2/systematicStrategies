from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from core.strategies.base import IStrategy, PortfolioWeights, StrategyConfig
from core.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class MomentumStrategy(IStrategy):
    """
    Cross-sectional momentum strategy.

    Ranks all assets by their trailing return over `lookback_period` days.

    Long-only mode (default):
        Buy equal weight in the top `top_k` assets, hold 0 in the rest.

    Long-short mode:
        Long +1/(2k) in the top k, short -1/(2k) in the bottom k.
        Net exposure is zero (market-neutral).

    Momentum is one of the most robust and widely documented anomalies in
    financial markets (Jegadeesh & Titman 1993).
    """

    DESCRIPTION = "Cross-sectional momentum — buy top-k assets by trailing return"
    FAMILY = "signal"

    def __init__(
        self,
        config: StrategyConfig,
        lookback_period: int = 252,
        top_k: int = 3,
        long_only: bool = True,
    ) -> None:
        super().__init__(config)
        self.lookback_period = lookback_period
        self.top_k = top_k
        self.long_only = long_only

    @property
    def required_history_days(self) -> int:
        return self.lookback_period + 1

    def compute_weights(
        self,
        current_date: date,
        price_history: pd.DataFrame,
        current_portfolio: object | None = None,
    ) -> PortfolioWeights:
        symbols = list(price_history.columns)
        n = len(symbols)
        k = min(self.top_k, n)

        # Trailing return: (price_today / price_lookback_days_ago) - 1
        window = price_history.tail(self.lookback_period + 1)
        trailing_returns = (window.iloc[-1] / window.iloc[0]) - 1.0

        ranked = trailing_returns.sort_values(ascending=False)
        top_k_symbols = list(ranked.index[:k])
        bottom_k_symbols = list(ranked.index[-k:]) if not self.long_only else []

        weights: dict[str, float] = {sym: 0.0 for sym in symbols}

        if self.long_only:
            for sym in top_k_symbols:
                weights[sym] = 1.0 / k
            cash_weight = 0.0
        else:
            for sym in top_k_symbols:
                weights[sym] = 1.0 / (2 * k)
            for sym in bottom_k_symbols:
                weights[sym] -= 1.0 / (2 * k)  # short
            cash_weight = 0.0

        return PortfolioWeights(
            weights=weights,
            cash_weight=cash_weight,
            timestamp=current_date,
        )

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "lookback_period": {
                "type": "integer",
                "default": 252,
                "description": "Trailing window in days for return ranking",
            },
            "top_k": {
                "type": "integer",
                "default": 3,
                "description": "Number of top-ranked assets to go long",
            },
            "long_only": {
                "type": "boolean",
                "default": True,
                "description": "True = long-only, False = long-short (market neutral)",
            },
            "rebalancing_frequency": {
                "type": "string",
                "default": "monthly",
                "enum": ["daily", "weekly", "monthly"],
            },
        }
