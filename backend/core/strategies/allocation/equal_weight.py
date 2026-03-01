from __future__ import annotations

from datetime import date

import pandas as pd

from core.strategies.base import IStrategy, PortfolioWeights, StrategyConfig
from core.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class EqualWeightStrategy(IStrategy):
    """
    Assigns equal weight 1/n to every asset in the universe.

    The simplest possible allocation strategy — no estimation risk,
    surprisingly competitive against complex optimised portfolios in practice
    (known as the '1/N' portfolio in the academic literature).
    """

    DESCRIPTION = "Equal weight 1/N across all assets, rebalanced periodically"
    FAMILY = "allocation"

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)

    @property
    def required_history_days(self) -> int:
        return 1

    def compute_weights(
        self,
        current_date: date,
        price_history: pd.DataFrame,
        current_portfolio: object | None = None,
    ) -> PortfolioWeights:
        symbols = list(price_history.columns)
        w = 1.0 / len(symbols)
        return PortfolioWeights(
            weights={sym: w for sym in symbols},
            cash_weight=0.0,
            timestamp=current_date,
        )

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "rebalancing_frequency": {
                "type": "string",
                "default": "weekly",
                "enum": ["daily", "weekly", "monthly"],
                "description": "How often to rebalance back to equal weight",
            }
        }
