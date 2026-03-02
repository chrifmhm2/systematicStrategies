from __future__ import annotations

import numpy as np
import pandas as pd

from core.pricing.black_scholes import BlackScholesModel


class GreeksCalculator:
    """Compute Black-Scholes Greeks over (spot, vol) surfaces and over time."""

    # ── [P4-23] Greeks surface ──────────────────────────────────────────────

    @staticmethod
    def compute_greeks_surface(
        spot_range: list[float] | np.ndarray,
        vol_range: list[float] | np.ndarray,
        strike: float,
        maturity: float,
        risk_free_rate: float,
    ) -> dict:
        """
        Compute delta, gamma, vega, theta over a (spot, vol) meshgrid.

        Parameters
        ----------
        spot_range      : Underlying spot prices to evaluate (e.g. np.linspace(80, 120, 20))
        vol_range       : Implied vols to evaluate (e.g. np.linspace(0.1, 0.5, 10))
        strike          : Option strike price
        maturity        : Time to expiry in years
        risk_free_rate  : Annualised risk-free rate

        Returns
        -------
        dict with keys: spots, vols, delta, gamma, vega, theta
        Each surface matrix has shape (len(spot_range), len(vol_range)).
        """
        spots = np.asarray(spot_range, dtype=float)
        vols = np.asarray(vol_range, dtype=float)

        n_s, n_v = len(spots), len(vols)
        delta_arr = np.empty((n_s, n_v))
        gamma_arr = np.empty((n_s, n_v))
        vega_arr = np.empty((n_s, n_v))
        theta_arr = np.empty((n_s, n_v))

        for i, S in enumerate(spots):
            for j, sigma in enumerate(vols):
                try:
                    delta_arr[i, j] = BlackScholesModel.delta(S, strike, maturity, risk_free_rate, sigma)
                    gamma_arr[i, j] = BlackScholesModel.gamma(S, strike, maturity, risk_free_rate, sigma)
                    vega_arr[i, j] = BlackScholesModel.vega(S, strike, maturity, risk_free_rate, sigma)
                    theta_arr[i, j] = BlackScholesModel.theta(S, strike, maturity, risk_free_rate, sigma)
                except (ValueError, ZeroDivisionError):
                    delta_arr[i, j] = np.nan
                    gamma_arr[i, j] = np.nan
                    vega_arr[i, j] = np.nan
                    theta_arr[i, j] = np.nan

        return {
            "spots": spots.tolist(),
            "vols": vols.tolist(),
            "delta": delta_arr.tolist(),
            "gamma": gamma_arr.tolist(),
            "vega": vega_arr.tolist(),
            "theta": theta_arr.tolist(),
        }

    # ── [P4-24] Greeks over time ────────────────────────────────────────────

    @staticmethod
    def compute_greeks_over_time(
        price_history: pd.Series,
        strike: float,
        risk_free_rate: float,
        sigma: float,
    ) -> pd.DataFrame:
        """
        Compute BS Greeks at each date in price_history with shrinking maturity.

        The total option life spans the full price_history.  At step i of n,
        the remaining maturity is (n - i) / 252 years, floored at 1/252.

        Returns
        -------
        pd.DataFrame with columns: delta, gamma, vega, theta, rho
        Index matches price_history.index.
        """
        n = len(price_history)
        records = []

        for i, (date, S) in enumerate(price_history.items()):
            T = max((n - i) / 252, 1 / 252)
            try:
                row = {
                    "delta": BlackScholesModel.delta(S, strike, T, risk_free_rate, sigma),
                    "gamma": BlackScholesModel.gamma(S, strike, T, risk_free_rate, sigma),
                    "vega": BlackScholesModel.vega(S, strike, T, risk_free_rate, sigma),
                    "theta": BlackScholesModel.theta(S, strike, T, risk_free_rate, sigma),
                    "rho": BlackScholesModel.rho(S, strike, T, risk_free_rate, sigma),
                }
            except (ValueError, ZeroDivisionError):
                row = {k: np.nan for k in ("delta", "gamma", "vega", "theta", "rho")}

            records.append(row)

        return pd.DataFrame(records, index=price_history.index)
