"""
Tests for MonteCarloPricer — [P1-31a] through [P1-31e]
"""

import numpy as np
import pytest
from core.pricing.black_scholes import BlackScholesModel as BS
from core.pricing.monte_carlo import MonteCarloPricer


# ── Shared setup ──────────────────────────────────────────────────────

N_SIM = 200_000   # enough for tight tolerance without being slow

# Single-asset degenerate basket (weight=1, 1 asset) — should match BS
SINGLE = dict(
    spots=np.array([100.0]),
    weights=np.array([1.0]),
    strike=100.0,
    maturity=1.0,
    risk_free_rate=0.05,
    volatilities=np.array([0.2]),
    correlation=np.array([[1.0]]),
)

# Two-asset basket
BASKET = dict(
    spots=np.array([100.0, 100.0]),
    weights=np.array([0.5, 0.5]),
    strike=95.0,
    maturity=1.0,
    risk_free_rate=0.05,
    volatilities=np.array([0.2, 0.25]),
    correlation=np.array([[1.0, 0.5], [0.5, 1.0]]),
)


# ── [P1-31a] Single-asset MC ≈ BS price ──────────────────────────────

def test_single_asset_matches_black_scholes():
    """
    Degenerate basket (1 asset, weight=1): MC price must be within
    2 standard errors of the Black-Scholes closed-form price.
    """
    pricer = MonteCarloPricer(n_simulations=N_SIM, seed=42)
    result = pricer.price_basket_option(**SINGLE)

    bs_price = BS.call_price(
        S=SINGLE["spots"][0],
        K=SINGLE["strike"],
        T=SINGLE["maturity"],
        r=SINGLE["risk_free_rate"],
        sigma=SINGLE["volatilities"][0],
    )

    assert abs(result.price - bs_price) < 2 * result.std_error, (
        f"MC price {result.price:.4f} more than 2 std errors from "
        f"BS price {bs_price:.4f} (std_error={result.std_error:.4f})"
    )


# ── [P1-31b] Antithetic reduces std_error vs plain MC ────────────────

def test_antithetic_reduces_variance():
    """
    Antithetic variates must produce a lower std_error than plain MC
    for the same n_simulations.
    """
    pricer_anti  = MonteCarloPricer(n_simulations=N_SIM, seed=0, variance_reduction="antithetic")
    pricer_plain = MonteCarloPricer(n_simulations=N_SIM, seed=0, variance_reduction="none")

    r_anti  = pricer_anti.price_basket_option(**SINGLE)
    r_plain = pricer_plain.price_basket_option(**SINGLE)

    assert r_anti.std_error < r_plain.std_error, (
        f"Antithetic std_error {r_anti.std_error:.6f} not < "
        f"plain std_error {r_plain.std_error:.6f}"
    )


# ── [P1-31c] ITM basket call price > 0 ───────────────────────────────

def test_itm_basket_price_positive():
    """Basket call price is strictly positive for an ITM option."""
    itm = {**BASKET, "strike": 80.0}   # spots ~100, strike 80 → deep ITM
    pricer = MonteCarloPricer(n_simulations=50_000, seed=1)
    result = pricer.price_basket_option(**itm)
    assert result.price > 0, f"ITM basket price {result.price:.4f} not positive"


# ── [P1-31d] All deltas in [0, 1] for basket call ────────────────────

def test_basket_call_deltas_in_range():
    """All deltas must be in [0, 1] for a basket call."""
    pricer = MonteCarloPricer(n_simulations=50_000, seed=2)
    result = pricer.price_basket_option(**BASKET)
    for i, d in enumerate(result.deltas):
        assert 0.0 <= d <= 1.0, f"Delta[{i}] = {d:.4f} not in [0, 1]"


# ── [P1-31e] Price consistent across seeds ───────────────────────────

def test_price_consistent_across_seeds():
    """
    Two independent runs (different seeds) must agree within 3 std errors
    of the first run — prices must converge as N grows.
    """
    pricer_a = MonteCarloPricer(n_simulations=N_SIM, seed=10)
    pricer_b = MonteCarloPricer(n_simulations=N_SIM, seed=99)

    r_a = pricer_a.price_basket_option(**BASKET)
    r_b = pricer_b.price_basket_option(**BASKET)

    assert abs(r_a.price - r_b.price) < 3 * r_a.std_error, (
        f"Prices diverge across seeds: {r_a.price:.4f} vs {r_b.price:.4f} "
        f"(3×std_error = {3 * r_a.std_error:.4f})"
    )


# ── Bonus: PricingResult fields sanity ───────────────────────────────

def test_pricing_result_fields():
    """confidence_interval is (lower, upper) and contains the price."""
    pricer = MonteCarloPricer(n_simulations=50_000, seed=3)
    result = pricer.price_basket_option(**BASKET)

    lo, hi = result.confidence_interval
    assert lo < result.price < hi, "Price not inside its own 95% CI"
    assert result.std_error > 0
    assert len(result.deltas) == len(BASKET["spots"])
