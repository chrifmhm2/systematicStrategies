from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass
class StrategyConfig:
    name: str
    description: str
    rebalancing_frequency: str = "weekly"
    transaction_cost_bps: float = 10.0


@dataclass
class PortfolioWeights:
    weights: dict[str, float]  # symbol -> weight in [0, 1] (or negative for short)
    cash_weight: float
    timestamp: date

    def total_weight(self) -> float:
        return sum(self.weights.values()) + self.cash_weight


class IStrategy(ABC):
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    @abstractmethod
    def compute_weights(
        self,
        current_date: date,
        price_history: pd.DataFrame,
        current_portfolio: object | None = None,
    ) -> PortfolioWeights:
        """Return target portfolio weights given price history up to current_date.

        Parameters
        ----------
        current_date:
            The date for which weights are being computed.
        price_history:
            DataFrame of shape (n_days, n_assets), columns = symbols, index = dates.
            Contains only data up to and including current_date (no look-ahead).
        current_portfolio:
            Optional current Portfolio object (used by hedging strategies).
        """

    @property
    @abstractmethod
    def required_history_days(self) -> int:
        """Minimum number of historical data rows needed before this strategy can trade."""

    @classmethod
    @abstractmethod
    def get_param_schema(cls) -> dict:
        """Return a JSON-schema-like dict describing the strategy's config parameters.

        Used by the frontend to render a dynamic form.
        """
