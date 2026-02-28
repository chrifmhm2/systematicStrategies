"""
Phase 1 — Interactive Demo & Verification
==========================================
Run from backend/ with:
    .venv/bin/python demos/phase1_demo.py

Produces 4 figures:
    1. GBM simulated price paths (SimulatedDataProvider)
    2. Black-Scholes call price vs spot (with Greeks)
    3. Monte Carlo vs Black-Scholes convergence
    4. MC payoff distribution + 95% confidence interval
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import date

# Make sure core/ is importable when running from backend/
sys.path.insert(0, ".")

from core.data import SimulatedDataProvider
from core.pricing import BlackScholesModel as BS, MonteCarloPricer

# ── Global style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4d",
    "axes.labelcolor":  "#c8ccd8",
    "xtick.color":      "#c8ccd8",
    "ytick.color":      "#c8ccd8",
    "text.color":       "#c8ccd8",
    "grid.color":       "#2a2d3d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "legend.facecolor": "#1a1d27",
    "legend.edgecolor": "#3a3d4d",
    "font.size":        11,
})

BLUE   = "#4c9be8"
GREEN  = "#50fa7b"
ORANGE = "#ffb86c"
RED    = "#ff5555"
PURPLE = "#bd93f9"
YELLOW = "#f1fa8c"


# ══════════════════════════════════════════════════════════════════════
# Figure 1 — GBM simulated price paths
# ══════════════════════════════════════════════════════════════════════

def plot_gbm_paths():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 1 — GBM Simulated Price Paths", fontsize=14, color=YELLOW, y=1.01)

    corr = np.array([[1.0, 0.7], [0.7, 1.0]])
    provider = SimulatedDataProvider(
        spots={"AAPL": 150.0, "MSFT": 300.0},
        volatilities={"AAPL": 0.25, "MSFT": 0.20},
        correlation=corr,
        drift=0.05,
        risk_free_rate=0.05,
        seed=42,
    )

    df = provider.get_prices(["AAPL", "MSFT"], date(2023, 1, 1), date(2024, 12, 31))

    colors = {"AAPL": BLUE, "MSFT": GREEN}
    for ax, symbol in zip(axes, ["AAPL", "MSFT"]):
        series = df[symbol]
        ax.plot(series.index, series.values, color=colors[symbol], linewidth=1.5)
        ax.axhline(series.iloc[0], color="#555", linewidth=0.8, linestyle="--", label="Start price")
        ax.fill_between(series.index, series.values, series.iloc[0],
                        alpha=0.1, color=colors[symbol])
        ax.set_title(f"{symbol}  (σ={'25%' if symbol=='AAPL' else '20%'}, corr=0.70)",
                     color=colors[symbol])
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.grid(True)
        ret = (series.iloc[-1] / series.iloc[0] - 1) * 100
        ax.text(0.02, 0.95, f"Total return: {ret:+.1f}%",
                transform=ax.transAxes, color=colors[symbol], fontsize=10,
                verticalalignment="top")

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# Figure 2 — Black-Scholes call price vs spot + Greeks
# ══════════════════════════════════════════════════════════════════════

def plot_bs_surface():
    K, T, r, sigma = 100.0, 1.0, 0.05, 0.20
    spots = np.linspace(60, 160, 300)

    prices = [BS.call_price(S, K, T, r, sigma) for S in spots]
    deltas = [BS.delta(S, K, T, r, sigma, "call") for S in spots]
    gammas = [BS.gamma(S, K, T, r, sigma) for S in spots]
    vegas  = [BS.vega(S, K, T, r, sigma) for S in spots]

    intrinsic = np.maximum(spots - K, 0)
    time_value = np.array(prices) - intrinsic

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Figure 2 — Black-Scholes Call  (K={K}, T={T}y, r={r*100:.0f}%, σ={sigma*100:.0f}%)",
        fontsize=14, color=YELLOW
    )

    # ── Price decomposition
    ax = axes[0, 0]
    ax.plot(spots, prices, color=BLUE, linewidth=2, label="Call price")
    ax.plot(spots, intrinsic, color=ORANGE, linewidth=1.2, linestyle="--", label="Intrinsic value")
    ax.fill_between(spots, intrinsic, prices, alpha=0.15, color=GREEN, label="Time value")
    ax.axvline(K, color=RED, linewidth=0.8, linestyle=":", label=f"Strike K={K}")
    ax.set_title("Call price = intrinsic + time value", color=BLUE)
    ax.set_xlabel("Spot price S")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=9)
    ax.grid(True)

    # ── Delta
    ax = axes[0, 1]
    ax.plot(spots, deltas, color=GREEN, linewidth=2)
    ax.axvline(K, color=RED, linewidth=0.8, linestyle=":", label=f"Strike K={K}")
    ax.axhline(0.5, color="#555", linewidth=0.8, linestyle="--", label="ATM Δ≈0.5")
    ax.set_title("Delta Δ  (shares needed to hedge)", color=GREEN)
    ax.set_xlabel("Spot price S")
    ax.set_ylabel("Delta")
    ax.legend(fontsize=9)
    ax.grid(True)

    # ── Gamma
    ax = axes[1, 0]
    ax.plot(spots, gammas, color=PURPLE, linewidth=2)
    ax.axvline(K, color=RED, linewidth=0.8, linestyle=":", label=f"Strike K={K}")
    ax.set_title("Gamma Γ  (how fast delta changes)", color=PURPLE)
    ax.set_xlabel("Spot price S")
    ax.set_ylabel("Gamma")
    ax.legend(fontsize=9)
    ax.grid(True)

    # ── Vega
    ax = axes[1, 1]
    ax.plot(spots, vegas, color=ORANGE, linewidth=2)
    ax.axvline(K, color=RED, linewidth=0.8, linestyle=":", label=f"Strike K={K}")
    ax.set_title("Vega ν  (sensitivity to volatility)", color=ORANGE)
    ax.set_xlabel("Spot price S")
    ax.set_ylabel("Vega")
    ax.legend(fontsize=9)
    ax.grid(True)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# Figure 3 — Monte Carlo vs BS convergence
# ══════════════════════════════════════════════════════════════════════

def plot_mc_convergence():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs_price = BS.call_price(S, K, T, r, sigma)

    n_values = [500, 1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000]
    mc_prices, mc_errors = [], []

    for n in n_values:
        pricer = MonteCarloPricer(n_simulations=n, seed=42)
        result = pricer.price_basket_option(
            spots=np.array([S]),
            weights=np.array([1.0]),
            strike=K,
            maturity=T,
            risk_free_rate=r,
            volatilities=np.array([sigma]),
            correlation=np.array([[1.0]]),
        )
        mc_prices.append(result.price)
        mc_errors.append(result.std_error)

    mc_prices = np.array(mc_prices)
    mc_errors = np.array(mc_errors)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 3 — Monte Carlo Convergence to Black-Scholes", fontsize=14, color=YELLOW)

    # ── Price convergence
    ax = axes[0]
    ax.semilogx(n_values, mc_prices, "o-", color=BLUE, linewidth=2, markersize=6, label="MC price")
    ax.fill_between(n_values,
                    mc_prices - 1.96 * mc_errors,
                    mc_prices + 1.96 * mc_errors,
                    alpha=0.2, color=BLUE, label="95% CI")
    ax.axhline(bs_price, color=GREEN, linewidth=1.5, linestyle="--",
               label=f"BS price = {bs_price:.4f}")
    ax.set_xlabel("Number of simulations")
    ax.set_ylabel("Option price ($)")
    ax.set_title("MC price converges to BS as N → ∞")
    ax.legend(fontsize=9)
    ax.grid(True)

    # ── Std error decay ~ 1/√N
    ax = axes[1]
    ax.loglog(n_values, mc_errors, "o-", color=ORANGE, linewidth=2, markersize=6, label="Std error")
    # theoretical 1/√N line
    n_arr = np.array(n_values, dtype=float)
    ax.loglog(n_values, mc_errors[0] * np.sqrt(n_values[0] / n_arr), "--",
              color=RED, linewidth=1.2, label="~1/√N (theoretical)")
    ax.set_xlabel("Number of simulations")
    ax.set_ylabel("Standard error")
    ax.set_title("Std error decays as 1/√N")
    ax.legend(fontsize=9)
    ax.grid(True)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# Figure 4 — Payoff distribution + antithetic comparison
# ══════════════════════════════════════════════════════════════════════

def plot_payoff_distribution():
    basket = dict(
        spots=np.array([100.0, 100.0]),
        weights=np.array([0.5, 0.5]),
        strike=95.0,
        maturity=1.0,
        risk_free_rate=0.05,
        volatilities=np.array([0.2, 0.25]),
        correlation=np.array([[1.0, 0.5], [0.5, 1.0]]),
    )

    pricer_anti  = MonteCarloPricer(n_simulations=100_000, seed=42, variance_reduction="antithetic")
    pricer_plain = MonteCarloPricer(n_simulations=100_000, seed=42, variance_reduction="none")

    r_anti  = pricer_anti.price_basket_option(**basket)
    r_plain = pricer_plain.price_basket_option(**basket)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 4 — Basket Option Pricing & Antithetic Variates", fontsize=14, color=YELLOW)

    # ── Pricing result comparison
    ax = axes[0]
    methods = ["Plain MC", "Antithetic MC"]
    prices  = [r_plain.price, r_anti.price]
    errors  = [r_plain.std_error, r_anti.std_error]
    colors  = [ORANGE, BLUE]
    bars = ax.bar(methods, prices, color=colors, width=0.4, alpha=0.85)
    ax.errorbar(methods, prices, yerr=[1.96 * e for e in errors],
                fmt="none", color="white", capsize=8, linewidth=2)
    for bar, price, err in zip(bars, prices, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{price:.4f}\n±{err:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Option price ($)")
    ax.set_title("Price ± 95% CI  (same N=100k simulations)")
    ax.grid(True, axis="y")

    # ── Std error comparison
    ax = axes[1]
    reduction_pct = (1 - r_anti.std_error / r_plain.std_error) * 100
    bars = ax.bar(methods, errors, color=colors, width=0.4, alpha=0.85)
    for bar, err in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
                f"{err:.5f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Standard error")
    ax.set_title(f"Antithetic reduces std error by {reduction_pct:.1f}%", color=GREEN)
    ax.grid(True, axis="y")
    ax.text(0.5, 0.85, f"Variance reduction: {reduction_pct:.1f}%",
            transform=ax.transAxes, ha="center", color=GREEN, fontsize=12,
            bbox=dict(boxstyle="round", facecolor="#1a1d27", edgecolor=GREEN))

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# Main — run all figures
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating Phase 1 visualizations...")

    figs = [
        ("GBM price paths",          plot_gbm_paths),
        ("Black-Scholes surface",     plot_bs_surface),
        ("Monte Carlo convergence",   plot_mc_convergence),
        ("Payoff distribution",       plot_payoff_distribution),
    ]

    for name, fn in figs:
        print(f"  → {name}")
        fig = fn()
        filename = f"demos/{name.lower().replace(' ', '_')}.png"
        fig.savefig(filename, dpi=140, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"     saved to {filename}")

    print("\nDone. Open the .png files to view the charts.")
    plt.show()
