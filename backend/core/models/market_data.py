from dataclasses import dataclass
from datetime import date


@dataclass
class DataFeed:
    """A single adjusted-close price observation for one symbol on one date."""

    symbol: str
    date: date
    price: float


@dataclass
class OHLCV:
    """Full candlestick bar: Open, High, Low, Close, Volume."""

    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
