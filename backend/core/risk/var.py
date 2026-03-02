from __future__ import annotations

import pandas as pd
from scipy.stats import norm


class VaRCalculator:
    """Value at Risk and Expected Shortfall (CVaR) calculators."""

    # ── [P4-18] Historical VaR ──────────────────────────────────────────────

    @staticmethod
    def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Sort returns, take the (1 - confidence) quantile.

        Returned as a negative number — e.g. -0.02 means the portfolio
        loses at least 2% on the worst 5% of trading days.
        """
        if len(returns) == 0:
            return float("nan")
        return float(returns.quantile(1 - confidence))

    # ── [P4-19] Parametric VaR ──────────────────────────────────────────────

    @staticmethod
    def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Normal-distribution VaR: μ - z * σ  where z = norm.ppf(confidence).

        Assumes returns are normally distributed.
        """
        if len(returns) < 2:
            return float("nan")
        mu = float(returns.mean())
        sigma = float(returns.std())
        z = norm.ppf(confidence)
        return mu - z * sigma

    # ── [P4-20] Historical CVaR (Expected Shortfall) ────────────────────────

    @staticmethod
    def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Mean of returns below the VaR threshold (Expected Shortfall).

        Always ≤ historical_var — it averages the actual tail losses.
        """
        if len(returns) == 0:
            return float("nan")
        var = VaRCalculator.historical_var(returns, confidence)
        tail = returns[returns <= var]
        if len(tail) == 0:
            return var
        return float(tail.mean())

    # ── [P4-21] Rolling VaR ─────────────────────────────────────────────────

    @staticmethod
    def rolling_var(
        returns: pd.Series, window: int = 252, confidence: float = 0.95
    ) -> pd.Series:
        """
        Historical VaR computed at each date using a rolling window.

        Returns a Series of the same length as `returns`, with NaN for the
        first `window - 1` entries (not enough history yet).
        """
        return returns.rolling(window=window).quantile(1 - confidence)
