from datetime import date, timedelta

import numpy as np
import pandas as pd

from core.data.base import IDataProvider


class SimulatedDataProvider(IDataProvider):
    """
    Generates synthetic price paths using correlated Geometric Brownian Motion (GBM).

    Formula per asset i at each daily step:
        S_t = S_{t-1} * exp((μ - σ_i²/2) * dt + σ_i * √dt * Z_i)

    where Z = L @ ε,  L = Cholesky(correlation),  ε ~ N(0, I)

    Used in tests and demos — deterministic when a seed is provided, no internet required.
    """

    def __init__(
        self,
        spots: dict[str, float],
        volatilities: dict[str, float],
        correlation: np.ndarray,
        drift: float = 0.0,
        risk_free_rate: float = 0.05,
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        spots : dict[str, float]
            Initial price per symbol, e.g. {"AAPL": 150.0, "MSFT": 300.0}.
        volatilities : dict[str, float]
            Annual volatility per symbol, e.g. {"AAPL": 0.25, "MSFT": 0.20}.
        correlation : np.ndarray
            Correlation matrix, shape (n_assets, n_assets). Must be positive definite.
        drift : float
            Annual drift μ applied to all assets (default 0.0).
        risk_free_rate : float
            Fixed annualised risk-free rate returned by get_risk_free_rate().
        seed : int | None
            Random seed for reproducibility. None = non-deterministic.
        """
        symbols = list(spots.keys())
        if list(volatilities.keys()) != symbols:
            raise ValueError("spots and volatilities must have the same keys in the same order.")
        if correlation.shape != (len(symbols), len(symbols)):
            raise ValueError(
                f"correlation must be ({len(symbols)}, {len(symbols)}), "
                f"got {correlation.shape}."
            )

        self._symbols = symbols
        self._spots = np.array([spots[s] for s in symbols], dtype=float)
        self._vols = np.array([volatilities[s] for s in symbols], dtype=float)
        self._chol = np.linalg.cholesky(correlation)   # L s.t. L @ L.T = corr
        self._drift = drift
        self._rfr = risk_free_rate
        self._seed = seed

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
        Simulate daily GBM price paths between start_date and end_date (inclusive).

        Returns
        -------
        pd.DataFrame
            DatetimeIndex, columns = requested symbols, values = simulated prices.
        """
        # Build a daily date range (calendar days — strategies will filter to business days)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        n_steps = len(dates)
        n_assets = len(self._symbols)

        rng = np.random.default_rng(self._seed)
        dt = 1 / 252  # one trading day in years

        # Draw independent standard normals: shape (n_steps, n_assets)
        eps = rng.standard_normal((n_steps, n_assets))

        # Introduce correlation via Cholesky: Z_t = eps_t @ L.T
        Z = eps @ self._chol.T   # shape (n_steps, n_assets)

        # Daily log-returns: r_t = (μ - σ²/2)*dt + σ*√dt*Z_t
        log_returns = (
            (self._drift - 0.5 * self._vols**2) * dt
            + self._vols * np.sqrt(dt) * Z
        )  # shape (n_steps, n_assets)

        # Cumulative product from S_0
        # prices[0] = S_0 * exp(log_return[0]), etc.
        prices = self._spots * np.exp(np.cumsum(log_returns, axis=0))

        df = pd.DataFrame(prices, index=dates, columns=self._symbols)

        # Filter to requested symbols only
        missing = [s for s in symbols if s not in df.columns]
        if missing:
            raise ValueError(f"Symbols not available in this provider: {missing}")

        return df[symbols]

    def get_risk_free_rate(self, date: date) -> float:  # noqa: ARG002
        return self._rfr
