from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from core.strategies.base import IStrategy, PortfolioWeights, StrategyConfig
from core.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class MeanReversionStrategy(IStrategy):
    """
    Z-score mean reversion strategy.

    For each asset computes a z-score:
        z_i = (S_i - MA_i) / std_i

    where MA and std are estimated over `lookback_window` days.

    Signal:
        z < -threshold  →  asset is cheap relative to recent history  →  BUY
        z > +threshold  →  asset is expensive relative to recent history  →  AVOID / SHORT

    Weights are inversely proportional to |z_i| among triggered symbols,
    normalised to sum to 1. Stronger deviation → lower weight (more
    cautious sizing as the asset is more extreme).

    Mean reversion exploits the tendency of prices to revert toward
    their short-term moving average (statistical arbitrage, pairs trading).
    """

    DESCRIPTION = "Z-score mean reversion — buy oversold, avoid/short overbought assets"
    FAMILY = "signal"

    def __init__(
        self,
        config: StrategyConfig,
        lookback_window: int = 20,
        z_threshold: float = 2.0,
    ) -> None:
        super().__init__(config)
        self.lookback_window = lookback_window
        self.z_threshold = z_threshold

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
        window = price_history.tail(self.lookback_window)

        ma = window.mean()
        std = window.std()
        current_prices = price_history.iloc[-1]

        # Avoid division by zero for constant price series
        std = std.replace(0.0, np.nan)
        z_scores = (current_prices - ma) / std

        # Buy signal: z < -threshold (oversold)
        long_signals = {sym: z_scores[sym] for sym in symbols if z_scores[sym] < -self.z_threshold}

        weights: dict[str, float] = {sym: 0.0 for sym in symbols}

        if long_signals:
            # Weight inversely proportional to |z| — more extreme = smaller position
            inv_z = {sym: 1.0 / abs(z) for sym, z in long_signals.items()}
            total = sum(inv_z.values())
            for sym, w in inv_z.items():
                weights[sym] = w / total

        return PortfolioWeights(
            weights=weights,
            cash_weight=1.0 - sum(weights.values()),
            timestamp=current_date,
        )

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "lookback_window": {
                "type": "integer",
                "default": 20,
                "description": "Window (days) for computing moving average and standard deviation",
            },
            "z_threshold": {
                "type": "number",
                "default": 2.0,
                "description": "Z-score threshold to trigger a signal",
            },
            "rebalancing_frequency": {
                "type": "string",
                "default": "weekly",
                "enum": ["daily", "weekly", "monthly"],
            },
        }
