from datetime import date

import pandas as pd

from core.data.base import IDataProvider

# Fallback rate used when no external rate source is available
_DEFAULT_RISK_FREE_RATE = 0.05


class CsvDataProvider(IDataProvider):
    """
    Loads price data from a CSV file in the Ensimag format.

    Expected CSV columns:
        Id            — asset identifier / ticker
        DateOfPrice   — date string parseable by pandas (e.g. "2020-01-02")
        Value         — adjusted close price (float)

    Example:
        Id,DateOfPrice,Value
        AAPL,2020-01-02,300.35
        MSFT,2020-01-02,158.96
        AAPL,2020-01-03,299.80
        ...
    """

    def __init__(self, filepath: str, risk_free_rate: float = _DEFAULT_RISK_FREE_RATE) -> None:
        """
        Parameters
        ----------
        filepath : str
            Path to the CSV file.
        risk_free_rate : float
            Fixed annualised risk-free rate to return (default 0.05).
        """
        self._rfr = risk_free_rate
        self._df = self._load(filepath)

    # ------------------------------------------------------------------
    # IDataProvider interface
    # ------------------------------------------------------------------

    def get_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Return adjusted-close prices for the requested symbols and date range.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex, one column per symbol, forward-filled for missing days.
        """
        missing = [s for s in symbols if s not in self._df.columns]
        if missing:
            raise ValueError(f"Symbols not found in CSV: {missing}")

        mask = (self._df.index >= pd.Timestamp(start_date)) & (
            self._df.index <= pd.Timestamp(end_date)
        )
        result = self._df.loc[mask, symbols].copy()

        # Forward-fill gaps (weekends / holidays recorded in the file)
        result = result.ffill()

        return result

    def get_risk_free_rate(self, date: date) -> float:  # noqa: ARG002
        return self._rfr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(filepath: str) -> pd.DataFrame:
        """
        Parse the CSV and pivot to wide format: DatetimeIndex × symbol columns.
        """
        raw = pd.read_csv(filepath, parse_dates=["DateOfPrice"])
        raw = raw.rename(columns={"Id": "symbol", "DateOfPrice": "date", "Value": "price"})
        raw = raw.sort_values("date")

        # Pivot: rows = dates, columns = symbols, values = price
        wide = raw.pivot_table(index="date", columns="symbol", values="price", aggfunc="last")
        wide.index = pd.DatetimeIndex(wide.index)
        wide.columns.name = None   # remove the "symbol" label from columns axis

        return wide
