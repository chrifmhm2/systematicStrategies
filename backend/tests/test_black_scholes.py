"""
Tests for BlackScholesModel — [P1-30a] through [P1-30f]
"""

import math
import pytest
import numpy as np
from core.pricing.black_scholes import BlackScholesModel as BS


# ── Fixtures / shared params ──────────────────────────────────────────

ATM = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)

PARAM_SETS = [
    dict(S=100, K=100, T=1.0, r=0.05, sigma=0.20),
    dict(S=120, K=100, T=0.5, r=0.03, sigma=0.15),
    dict(S=80,  K=100, T=2.0, r=0.04, sigma=0.30),
    dict(S=100, K=110, T=1.0, r=0.05, sigma=0.25),
    dict(S=150, K=130, T=0.25,r=0.02, sigma=0.18),
]


# ── [P1-30a] ATM call price ───────────────────────────────────────────

def test_atm_call_price():
    """ATM call: S=100, K=100, T=1, r=0.05, σ=0.2 → price ≈ 10.45"""
    price = BS.call_price(**ATM)
    assert abs(price - 10.45) < 0.01, f"Expected ~10.45, got {price:.4f}"


# ── [P1-30b] Put-call parity ──────────────────────────────────────────

@pytest.mark.parametrize("p", PARAM_SETS)
def test_put_call_parity(p):
    """C - P = S - K * e^(-rT) for all parameter sets."""
    C = BS.call_price(**p)
    P = BS.put_price(**p)
    lhs = C - P
    rhs = p["S"] - p["K"] * math.exp(-p["r"] * p["T"])
    assert abs(lhs - rhs) < 1e-8, f"Parity violated: {lhs:.6f} != {rhs:.6f}"


# ── [P1-30c] Delta bounds ─────────────────────────────────────────────

@pytest.mark.parametrize("p", PARAM_SETS)
def test_call_delta_in_range(p):
    d = BS.delta(**p, option_type="call")
    assert 0.0 <= d <= 1.0, f"Call delta {d:.4f} not in [0, 1]"


@pytest.mark.parametrize("p", PARAM_SETS)
def test_put_delta_in_range(p):
    d = BS.delta(**p, option_type="put")
    assert -1.0 <= d <= 0.0, f"Put delta {d:.4f} not in [-1, 0]"


# ── [P1-30d] Gamma and Vega always positive ───────────────────────────

@pytest.mark.parametrize("p", PARAM_SETS)
def test_gamma_positive(p):
    g = BS.gamma(**p)
    assert g > 0, f"Gamma {g:.6f} not positive"


@pytest.mark.parametrize("p", PARAM_SETS)
def test_vega_positive(p):
    v = BS.vega(**p)
    assert v > 0, f"Vega {v:.4f} not positive"


# ── [P1-30e] Implied vol round-trip ──────────────────────────────────

@pytest.mark.parametrize("p", PARAM_SETS)
def test_implied_vol_roundtrip_call(p):
    """Compute call price → extract IV → recompute price → match within 1e-4."""
    price = BS.call_price(**p)
    iv = BS.implied_volatility(price, p["S"], p["K"], p["T"], p["r"], "call")
    price2 = BS.call_price(p["S"], p["K"], p["T"], p["r"], iv)
    assert abs(price2 - price) < 1e-4, f"IV round-trip error: {abs(price2 - price):.2e}"


@pytest.mark.parametrize("p", PARAM_SETS)
def test_implied_vol_roundtrip_put(p):
    """Same round-trip for put options."""
    price = BS.put_price(**p)
    iv = BS.implied_volatility(price, p["S"], p["K"], p["T"], p["r"], "put")
    price2 = BS.put_price(p["S"], p["K"], p["T"], p["r"], iv)
    assert abs(price2 - price) < 1e-4, f"IV round-trip error: {abs(price2 - price):.2e}"


# ── [P1-30f] Greeks vs central finite differences ────────────────────

def _fd_greek(fn, param_name, p, h=1e-4):
    """Central finite difference: (f(x+h) - f(x-h)) / (2h)."""
    p_up = {**p, param_name: p[param_name] + h}
    p_dn = {**p, param_name: p[param_name] - h}
    return (fn(**p_up) - fn(**p_dn)) / (2 * h)


def test_delta_vs_fd():
    """Call delta matches dC/dS via finite difference within 1e-3."""
    p = ATM
    analytic = BS.delta(**p, option_type="call")
    fd = _fd_greek(BS.call_price, "S", p, h=0.01)
    assert abs(analytic - fd) < 1e-3, f"Delta FD error: {abs(analytic - fd):.2e}"


def test_gamma_vs_fd():
    """Gamma matches d²C/dS² via finite difference within 1e-3."""
    p = ATM
    analytic = BS.gamma(**p)
    h = 0.01
    fd = (BS.call_price(**{**p, "S": p["S"] + h})
          - 2 * BS.call_price(**p)
          + BS.call_price(**{**p, "S": p["S"] - h})) / h**2
    assert abs(analytic - fd) < 1e-3, f"Gamma FD error: {abs(analytic - fd):.2e}"


def test_vega_vs_fd():
    """Vega matches dC/dσ via finite difference within 1e-3."""
    p = ATM
    analytic = BS.vega(**p)
    fd = _fd_greek(BS.call_price, "sigma", p, h=1e-4)
    assert abs(analytic - fd) < 1e-3, f"Vega FD error: {abs(analytic - fd):.2e}"


def test_rho_vs_fd():
    """Rho matches dC/dr via finite difference within 1e-3."""
    p = ATM
    analytic = BS.rho(**p, option_type="call")
    fd = _fd_greek(BS.call_price, "r", p, h=1e-4)
    assert abs(analytic - fd) < 1e-3, f"Rho FD error: {abs(analytic - fd):.2e}"
