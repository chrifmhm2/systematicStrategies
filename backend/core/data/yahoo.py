from __future__ import annotations

from datetime import date

import pandas as pd

from core.data.base import IDataProvider

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "HD", "UNH",
    "DIS", "BAC", "XOM", "PFE", "BRK-B",
]


class YahooDataProvider(IDataProvider):
    """
    Fetches adjusted-close prices from Yahoo Finance via yfinance.

    Features
    --------
    - Optional in-memory cache: repeated calls with the same parameters skip the HTTP request.
    - Forward-fills NaN values (weekends, holidays, missing data).
    - get_risk_free_rate() fetches the 13-week T-bill (^IRX); falls back to 5% on failure.

    Parameters
    ----------
    cache : bool
        If True (default), cache results in memory keyed by (symbols, start, end).
    """

    def __init__(self, cache: bool = True) -> None:
        self._cache: dict[tuple, pd.DataFrame] = {}
        self._use_cache = cache

    def get_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Download adjusted-close prices from Yahoo Finance.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex rows, one column per symbol, forward-filled.
        """
        import yfinance as yf

        key = (tuple(sorted(symbols)), start_date, end_date)
        if self._use_cache and key in self._cache:
            return self._cache[key]

        raw = yf.download(
            symbols,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )

        # yfinance returns multi-level columns for multiple tickers
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw["Close"]
        else:
            # Single ticker: flat columns like Open, High, Low, Close, Volume
            df = raw[["Close"]]
            df.columns = symbols

        # Ensure all requested symbols are present
        for sym in symbols:
            if sym not in df.columns:
                df[sym] = float("nan")

        df = df[symbols]  # enforce column order
        df = df.ffill()   # forward-fill gaps (weekends, holidays)

        if self._use_cache:
            self._cache[key] = df

        return df

    def get_risk_free_rate(self, as_of_date: date) -> float:  # noqa: ARG002
        """
        Return the annualised risk-free rate.

        Attempts to fetch the 13-week T-bill yield (^IRX) from Yahoo Finance.
        Falls back to 5% (0.05) if the fetch fails for any reason.

        Returns
        -------
        float
            Annualised rate as a decimal (e.g. 0.05 for 5%).
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker("^IRX")
            hist = ticker.history(period="5d")
            if hist.empty:
                return 0.05
            latest = float(hist["Close"].iloc[-1])
            return latest / 100.0  # ^IRX is quoted as percentage
        except Exception:  # noqa: BLE001
            return 0.05
