from dataclasses import dataclass, field
from datetime import date


@dataclass
class Position:
    """A holding in a single asset."""

    symbol: str
    quantity: float   # number of units / shares held
    price: float      # last known price used to value the position


@dataclass
class Portfolio:
    """
    Snapshot of the full portfolio at a given date.

    `positions` maps symbol → Position.
    `cash` is the uninvested cash balance (can be negative if leveraged).
    """

    positions: dict[str, Position] = field(default_factory=dict)
    cash: float = 0.0
    date: date = field(default_factory=date.today)

    def total_value(self, prices: dict[str, float]) -> float:
        """
        Mark-to-market portfolio value using the provided price map.

        Parameters
        ----------
        prices : dict[str, float]
            Current prices keyed by symbol.

        Returns
        -------
        float
            Sum of (quantity × current_price) for all positions plus cash.
        """
        equity = sum(
            pos.quantity * prices[symbol]
            for symbol, pos in self.positions.items()
            if symbol in prices
        )
        return equity + self.cash
