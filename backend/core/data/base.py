from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class IDataProvider(ABC):
    """
    Abstract interface for all data sources.

    Strategies and the backtester always code against this interface —
    never against a concrete provider (SimulatedDataProvider, YahooDataProvider, etc.).
    This ensures data source independence: swap providers without touching strategy code.
    """

    @abstractmethod
    def get_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Return adjusted-close prices for the requested symbols and date range.

        Parameters
        ----------
        symbols : list[str]
            Tickers to fetch (e.g. ["AAPL", "MSFT"]).
        start_date : date
            First date (inclusive).
        end_date : date
            Last date (inclusive).

        Returns
        -------
        pd.DataFrame
            DatetimeIndex rows, one column per symbol, values = adjusted close price.
            Missing values (non-trading days, gaps) are forward-filled by the provider.
        """

    @abstractmethod
    def get_risk_free_rate(self, date: date) -> float:
        """
        Return the annualised risk-free rate on a given date.

        Parameters
        ----------
        date : date
            The date for which the rate is requested.

        Returns
        -------
        float
            Annualised risk-free rate (e.g. 0.05 for 5%).
        """
