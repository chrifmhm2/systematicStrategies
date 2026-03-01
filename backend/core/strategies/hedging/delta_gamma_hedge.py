from __future__ import annotations

from datetime import date

import pandas as pd

from core.strategies.base import IStrategy, PortfolioWeights, StrategyConfig
from core.strategies.hedging.delta_hedge import DeltaHedgeStrategy
from core.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class DeltaGammaHedgeStrategy(IStrategy):
    """
    Delta-Gamma hedging strategy (stub — falls back to delta hedge).

    Full gamma hedging requires a second instrument (e.g. a vanilla option)
    to neutralise the portfolio's Γ exposure. This is planned for a future
    iteration. For now, compute_weights() delegates to DeltaHedgeStrategy.
    """

    DESCRIPTION = "Delta-Gamma hedge (Gamma neutralisation planned — currently delegates to Delta hedge)"
    FAMILY = "hedging"

    def __init__(
        self,
        config: StrategyConfig,
        **kwargs,
    ) -> None:
        super().__init__(config)
        self._delta_hedge = DeltaHedgeStrategy(config, **kwargs)

    @property
    def required_history_days(self) -> int:
        return 1

    def compute_weights(
        self,
        current_date: date,
        price_history: pd.DataFrame,
        current_portfolio: object | None = None,
    ) -> PortfolioWeights:
        # Delegate to delta hedge until gamma instrument is introduced
        return self._delta_hedge.compute_weights(current_date, price_history, current_portfolio)

    @classmethod
    def get_param_schema(cls) -> dict:
        return DeltaHedgeStrategy.get_param_schema()
