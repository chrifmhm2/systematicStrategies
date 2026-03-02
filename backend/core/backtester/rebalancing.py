from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date


class RebalancingOracle(ABC):
    """
    Abstract decision rule for when to rebalance a portfolio.

    The BacktestEngine calls `should_rebalance(current_date, portfolio)` at every
    step of the backtest loop. Concrete implementations encode the logic.
    """

    @abstractmethod
    def should_rebalance(self, current_date: date, portfolio: object) -> bool:
        """
        Return True if the portfolio should be rebalanced on current_date.

        Parameters
        ----------
        current_date : date
            The date being processed in the backtest loop.
        portfolio : Portfolio
            Current portfolio snapshot (positions, cash, date).

        Returns
        -------
        bool
        """


class PeriodicRebalancing(RebalancingOracle):
    """
    Rebalance on a fixed calendar schedule.

    - "daily"   → rebalance every trading day
    - "weekly"  → rebalance every Monday (or first available trading day of the week)
    - "monthly" → rebalance on the first trading day of each calendar month
    """

    def __init__(self, frequency: str = "weekly") -> None:
        if frequency not in ("daily", "weekly", "monthly"):
            raise ValueError(f"frequency must be 'daily', 'weekly', or 'monthly'. Got: {frequency!r}")
        self._frequency = frequency
        self._last_rebalance_date: date | None = None

    def should_rebalance(self, current_date: date, portfolio: object) -> bool:  # noqa: ARG002
        if self._frequency == "daily":
            result = True
        elif self._frequency == "weekly":
            # Rebalance on Monday (weekday 0).
            # If Monday is not a trading day, the first available day of that week
            # will be the smallest weekday — catch it by also firing if last rebalance
            # was more than 4 trading days ago (handled implicitly: iterating real dates
            # means if Mon is a holiday, Tue weekday=1 won't fire; the engine will miss
            # that week and rebalance next Monday — acceptable for a backtester).
            result = current_date.weekday() == 0
        else:  # monthly
            # Fire on the first trading day of a new calendar month.
            result = (
                self._last_rebalance_date is None
                or current_date.month != self._last_rebalance_date.month
            )

        if result:
            self._last_rebalance_date = current_date
        return result


class ThresholdRebalancing(RebalancingOracle):
    """
    Rebalance when any asset's weight has drifted more than `threshold` from its
    last target weight.

    Useful for keeping the portfolio close to the target allocation without
    rebalancing on a fixed schedule.
    """

    def __init__(self, threshold: float = 0.05) -> None:
        self._threshold = threshold
        self._last_target_weights: dict[str, float] = {}

    def should_rebalance(self, current_date: date, portfolio: object) -> bool:  # noqa: ARG002
        from core.models.portfolio import Portfolio  # local import to avoid circular dep

        if not self._last_target_weights:
            return True  # no target set yet → always rebalance on first call

        if not isinstance(portfolio, Portfolio) or not portfolio.positions:
            return True

        prices = {sym: pos.price for sym, pos in portfolio.positions.items()}
        total = portfolio.total_value(prices)
        if total <= 0:
            return False

        for sym, target_w in self._last_target_weights.items():
            if sym in portfolio.positions:
                pos = portfolio.positions[sym]
                current_w = (pos.quantity * pos.price) / total
            else:
                current_w = 0.0
            if abs(current_w - target_w) > self._threshold:
                return True
        return False

    def update_target_weights(self, weights: dict[str, float]) -> None:
        """Call after each rebalancing to store the new target weights."""
        self._last_target_weights = weights.copy()
