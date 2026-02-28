import math
from typing import Literal

import numpy as np
from scipy.stats import norm


class BlackScholesModel:
    """
    Closed-form Black-Scholes pricing for European vanilla options.

    All methods are static — no instance needed:
        BlackScholesModel.call_price(S=100, K=100, T=1, r=0.05, sigma=0.2)

    Assumptions:
        - European exercise (no early exercise)
        - Constant volatility and risk-free rate
        - No dividends
        - Continuous compounding
    """

    # ------------------------------------------------------------------
    # Internal helpers — d1 and d2
    # ------------------------------------------------------------------

    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        d1 = [ ln(S/K) + (r + σ²/2)*T ] / (σ*√T)

        Measures how far in-the-money the option is, adjusted for time and vol.
        """
        return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        d2 = d1 - σ*√T

        Risk-neutral probability that the option expires in-the-money.
        """
        d1 = BlackScholesModel._d1(S, K, T, r, sigma)
        return d1 - sigma * math.sqrt(T)

    # ------------------------------------------------------------------
    # [P1-18] Call price
    # ------------------------------------------------------------------

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Price a European call option.

        C = S * N(d1) - K * e^(-rT) * N(d2)

        Parameters
        ----------
        S     : Current underlying price
        K     : Strike price
        T     : Time to expiry in years
        r     : Annualised risk-free rate (e.g. 0.05)
        sigma : Annualised volatility (e.g. 0.20)

        Returns
        -------
        float : Fair value of the call option
        """
        d1 = BlackScholesModel._d1(S, K, T, r, sigma)
        d2 = BlackScholesModel._d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

    # ------------------------------------------------------------------
    # [P1-19] Put price — put-call parity
    # ------------------------------------------------------------------

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Price a European put option using put-call parity.

        P = C - S + K * e^(-rT)

        Equivalent to: P = K * e^(-rT) * N(-d2) - S * N(-d1)
        """
        C = BlackScholesModel.call_price(S, K, T, r, sigma)
        return C - S + K * math.exp(-r * T)

    # ------------------------------------------------------------------
    # [P1-20] Delta
    # ------------------------------------------------------------------

    @staticmethod
    def delta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call",
    ) -> float:
        """
        Delta — sensitivity of option price to a $1 move in the underlying.

        Call delta = N(d1)         ∈ [0, 1]
        Put  delta = N(d1) - 1     ∈ [-1, 0]

        Interpretation: a delta of 0.55 means the option price moves ~$0.55
        for every $1 move in the underlying.
        """
        d1 = BlackScholesModel._d1(S, K, T, r, sigma)
        if option_type == "call":
            return norm.cdf(d1)
        return norm.cdf(d1) - 1.0

    # ------------------------------------------------------------------
    # [P1-21] Gamma
    # ------------------------------------------------------------------

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Gamma — rate of change of delta per $1 move in the underlying.

        Γ = n(d1) / (S * σ * √T)

        Same for calls and puts. Always positive.
        High gamma → delta changes rapidly → option is highly sensitive near expiry.
        """
        d1 = BlackScholesModel._d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))

    # ------------------------------------------------------------------
    # [P1-22] Vega
    # ------------------------------------------------------------------

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Vega — sensitivity of option price to a 1-unit move in volatility.

        ν = S * n(d1) * √T

        Same for calls and puts. Always positive.
        Returned per unit of sigma (not per 1% move).
        """
        d1 = BlackScholesModel._d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * math.sqrt(T)

    # ------------------------------------------------------------------
    # [P1-23] Theta
    # ------------------------------------------------------------------

    @staticmethod
    def theta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call",
    ) -> float:
        """
        Theta — daily time decay of the option price (per calendar day).

        Θ_call = -[S*n(d1)*σ / (2*√T)] - r*K*e^(-rT)*N(d2)
        Θ_put  = -[S*n(d1)*σ / (2*√T)] + r*K*e^(-rT)*N(-d2)

        Divided by 365 to express as daily decay.
        Almost always negative — options lose value as time passes.
        """
        d1 = BlackScholesModel._d1(S, K, T, r, sigma)
        d2 = BlackScholesModel._d2(S, K, T, r, sigma)

        common = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))

        if option_type == "call":
            return (common - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
        return (common + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365

    # ------------------------------------------------------------------
    # [P1-24] Rho
    # ------------------------------------------------------------------

    @staticmethod
    def rho(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call",
    ) -> float:
        """
        Rho — sensitivity of option price to a 1-unit move in the risk-free rate.

        ρ_call =  K * T * e^(-rT) * N(d2)
        ρ_put  = -K * T * e^(-rT) * N(-d2)

        Calls have positive rho (higher rates → higher call value).
        Puts have negative rho (higher rates → lower put value).
        """
        d2 = BlackScholesModel._d2(S, K, T, r, sigma)
        if option_type == "call":
            return K * T * math.exp(-r * T) * norm.cdf(d2)
        return -K * T * math.exp(-r * T) * norm.cdf(-d2)

    # ------------------------------------------------------------------
    # [P1-25] Implied volatility — Newton-Raphson
    # ------------------------------------------------------------------

    @staticmethod
    def implied_volatility(
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: Literal["call", "put"] = "call",
        initial_guess: float = 0.2,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> float:
        """
        Find the implied volatility σ such that BS(σ) == price.

        Uses Newton-Raphson iteration:
            σ_new = σ_old - (BS(σ_old) - market_price) / vega(σ_old)

        Parameters
        ----------
        price         : Observed market price of the option
        S, K, T, r    : Same as call_price / put_price
        option_type   : "call" or "put"
        initial_guess : Starting σ for the iteration (default 0.2)
        max_iterations: Maximum Newton-Raphson steps before giving up
        tolerance     : Convergence threshold on |BS(σ) - price|

        Returns
        -------
        float : Implied volatility

        Raises
        ------
        ValueError
            If the algorithm fails to converge within max_iterations.
        """
        pricer = (
            BlackScholesModel.call_price
            if option_type == "call"
            else BlackScholesModel.put_price
        )

        sigma = initial_guess

        for i in range(max_iterations):
            bs_price = pricer(S, K, T, r, sigma)
            diff = bs_price - price

            if abs(diff) < tolerance:
                return sigma

            v = BlackScholesModel.vega(S, K, T, r, sigma)

            if abs(v) < 1e-10:
                # Vega too small — Newton-Raphson becomes unstable
                raise ValueError(
                    f"Implied volatility did not converge: vega ≈ 0 at iteration {i}. "
                    "The option may be too deep in/out of the money."
                )

            sigma -= diff / v

            # Keep sigma in a sensible range
            sigma = max(1e-6, min(sigma, 10.0))

        raise ValueError(
            f"Implied volatility did not converge after {max_iterations} iterations. "
            f"Last σ = {sigma:.6f}, last |error| = {abs(diff):.2e}."
        )
