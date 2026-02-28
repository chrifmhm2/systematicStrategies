from typing import Literal

import numpy as np

from core.models.results import PricingResult
from core.pricing.utils import cholesky_decompose, generate_correlated_normals


class MonteCarloPricer:
    """
    Monte Carlo pricer for European basket call options.

    Simulates N correlated GBM terminal prices and averages discounted payoffs.
    Supports antithetic variates for variance reduction.

    Usage:
        pricer = MonteCarloPricer(n_simulations=100_000, seed=42)
        result = pricer.price_basket_option(
            spots=np.array([100.0, 100.0]),
            weights=np.array([0.5, 0.5]),
            strike=100.0,
            maturity=1.0,
            risk_free_rate=0.05,
            volatilities=np.array([0.2, 0.2]),
            correlation=np.array([[1.0, 0.5], [0.5, 1.0]]),
        )
        print(result.price, result.std_error)
    """

    # ------------------------------------------------------------------
    # [P1-27] Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        n_simulations: int = 100_000,
        seed: int | None = None,
        variance_reduction: Literal["antithetic", "none"] = "antithetic",
    ) -> None:
        """
        Parameters
        ----------
        n_simulations : int
            Number of Monte Carlo paths (or pairs, if antithetic). Default 100_000.
        seed : int | None
            Random seed for reproducibility.
        variance_reduction : "antithetic" | "none"
            "antithetic" — draw Z and -Z, average both payoffs before taking the mean.
                           Halves variance for the same n_simulations.
            "none"       — plain Monte Carlo, no variance reduction.
        """
        self.n_simulations = n_simulations
        self.seed = seed
        self.variance_reduction = variance_reduction

    # ------------------------------------------------------------------
    # [P1-28] Price basket option
    # ------------------------------------------------------------------

    def price_basket_option(
        self,
        spots: np.ndarray,
        weights: np.ndarray,
        strike: float,
        maturity: float,
        risk_free_rate: float,
        volatilities: np.ndarray,
        correlation: np.ndarray,
    ) -> PricingResult:
        """
        Price a European basket call option via Monte Carlo simulation.

        Terminal price for asset i:
            S_i^T = S_i * exp( (r - σ_i²/2)*T  +  σ_i*√T * Z_i )

        Basket payoff:
            payoff = max( Σ ω_i * S_i^T  -  K,  0 )

        Price:
            V = e^(-rT) * E[payoff]

        Parameters
        ----------
        spots       : np.ndarray  shape (n_assets,) — current prices
        weights     : np.ndarray  shape (n_assets,) — basket weights (sum to 1)
        strike      : float       — K
        maturity    : float       — T in years
        risk_free_rate : float    — r (annualised)
        volatilities: np.ndarray  shape (n_assets,) — annual vols
        correlation : np.ndarray  shape (n_assets, n_assets)

        Returns
        -------
        PricingResult
            price, std_error, 95% confidence_interval, deltas per asset
        """
        chol = cholesky_decompose(correlation)
        n_assets = len(spots)

        payoffs = self._simulate_payoffs(
            spots, weights, strike, maturity, risk_free_rate, volatilities, chol, self.seed
        )

        discount = np.exp(-risk_free_rate * maturity)
        discounted = payoffs * discount

        price = float(np.mean(discounted))
        std_error = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))
        z95 = 1.96
        ci = (price - z95 * std_error, price + z95 * std_error)

        deltas = self.compute_deltas(
            spots, weights, strike, maturity, risk_free_rate, volatilities, correlation
        )

        return PricingResult(
            price=price,
            std_error=std_error,
            confidence_interval=ci,
            deltas=deltas,
        )

    # ------------------------------------------------------------------
    # [P1-29] Finite-difference deltas
    # ------------------------------------------------------------------

    def compute_deltas(
        self,
        spots: np.ndarray,
        weights: np.ndarray,
        strike: float,
        maturity: float,
        risk_free_rate: float,
        volatilities: np.ndarray,
        correlation: np.ndarray,
        bump_size: float = 0.01,
    ) -> np.ndarray:
        """
        Central finite-difference delta per asset.

            Δ_i = ( V(S_i + bump*S_i) - V(S_i - bump*S_i) ) / (2 * bump * S_i)

        The same random seed is reused for up/down bumps to cancel noise.

        Parameters
        ----------
        bump_size : float
            Fractional bump applied to each spot (default 1%).

        Returns
        -------
        np.ndarray  shape (n_assets,)
        """
        chol = cholesky_decompose(correlation)
        discount = np.exp(-risk_free_rate * maturity)
        deltas = np.zeros(len(spots))

        for i in range(len(spots)):
            bump = bump_size * spots[i]

            spots_up = spots.copy()
            spots_up[i] += bump

            spots_dn = spots.copy()
            spots_dn[i] -= bump

            # Use the same seed for both bumps → noise cancels in the difference
            payoffs_up = self._simulate_payoffs(
                spots_up, weights, strike, maturity, risk_free_rate, volatilities, chol, self.seed
            )
            payoffs_dn = self._simulate_payoffs(
                spots_dn, weights, strike, maturity, risk_free_rate, volatilities, chol, self.seed
            )

            v_up = float(np.mean(payoffs_up)) * discount
            v_dn = float(np.mean(payoffs_dn)) * discount

            deltas[i] = (v_up - v_dn) / (2 * bump)

        return deltas

    # ------------------------------------------------------------------
    # Internal simulation helper
    # ------------------------------------------------------------------

    def _simulate_payoffs(
        self,
        spots: np.ndarray,
        weights: np.ndarray,
        strike: float,
        maturity: float,
        risk_free_rate: float,
        volatilities: np.ndarray,
        chol: np.ndarray,
        seed: int | None,
    ) -> np.ndarray:
        """
        Simulate basket payoffs for one set of spot prices.

        Antithetic variates:
            Draw Z ~ N(0, I), compute payoff(+Z) and payoff(-Z),
            return their average per pair → halves variance.

        Returns
        -------
        np.ndarray  shape (n_simulations,) — one payoff per path (or pair)
        """
        n_assets = len(spots)

        Z = generate_correlated_normals(n_assets, self.n_simulations, chol, seed)
        # shape (n_simulations, n_assets)

        # GBM terminal prices: S_i^T = S_i * exp((r - σ_i²/2)*T + σ_i*√T*Z_i)
        drift_term = (risk_free_rate - 0.5 * volatilities**2) * maturity
        diffusion_term = volatilities * np.sqrt(maturity)

        def terminal_prices(z: np.ndarray) -> np.ndarray:
            return spots * np.exp(drift_term + diffusion_term * z)

        def basket_payoff(z: np.ndarray) -> np.ndarray:
            S_T = terminal_prices(z)                           # (n_sim, n_assets)
            basket = S_T @ weights                             # (n_sim,)
            return np.maximum(basket - strike, 0.0)

        if self.variance_reduction == "antithetic":
            payoffs_pos = basket_payoff(Z)
            payoffs_neg = basket_payoff(-Z)
            return (payoffs_pos + payoffs_neg) / 2.0
        else:
            return basket_payoff(Z)
