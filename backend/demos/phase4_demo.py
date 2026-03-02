"""
Phase 4 Demo — Risk Analytics
===============================
Run from backend/ with:
    .venv/bin/python demos/phase4_demo.py

Produces 3 figures saved to demos/:
    phase4_metrics_comparison.png  — Bar chart of key metrics across 3 strategies
    phase4_rolling_var.png         — Rolling 60-day VaR (95%) for each strategy
    phase4_greeks_surface.png      — Delta / Gamma / Vega / Theta heatmaps over (spot, vol)

Terminal output shows the full risk metrics table and VaR breakdown.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from datetime import date

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from core.backtester.engine import BacktestConfig, BacktestEngine
from core.data.simulated import SimulatedDataProvider
from core.risk.metrics import PerformanceMetrics
from core.risk.var import VaRCalculator
from core.risk.greeks import GreeksCalculator
from core.strategies.allocation.equal_weight import EqualWeightStrategy
from core.strategies.allocation.min_variance import MinVarianceStrategy
from core.strategies.signal.momentum import MomentumStrategy
from core.strategies.base import StrategyConfig

# ── Global style (matches phase1/phase2/phase3) ───────────────────────────────

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
GRAY   = "#6272a4"

STRATEGY_COLORS = {"EqualWeight": BLUE, "Momentum": GREEN, "MinVariance": ORANGE}


def section(title: str) -> None:
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


# ── Shared setup (same seed/data as phase3_demo) ──────────────────────────────

SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]

DATA = SimulatedDataProvider(
    spots       = {"AAPL": 150.0, "MSFT": 300.0, "GOOG": 120.0, "AMZN": 180.0, "NVDA": 200.0},
    volatilities= {"AAPL": 0.25,  "MSFT": 0.22,  "GOOG": 0.28,  "AMZN": 0.30,  "NVDA": 0.40},
    correlation = np.array([
        [1.00, 0.70, 0.60, 0.50, 0.35],
        [0.70, 1.00, 0.65, 0.55, 0.30],
        [0.60, 0.65, 1.00, 0.60, 0.25],
        [0.50, 0.55, 0.60, 1.00, 0.25],
        [0.35, 0.30, 0.25, 0.25, 1.00],
    ]),
    drift         = 0.05,
    risk_free_rate= 0.05,
    seed          = 7,
)

START = date(2022, 1, 3)
END   = date(2023, 12, 29)

BASE_CFG = dict(
    symbols              = SYMBOLS,
    start_date           = START,
    end_date             = END,
    transaction_cost_bps = 10.0,
    slippage_bps         = 5.0,
    rebalancing_frequency= "weekly",
)

strategies = {
    "EqualWeight": EqualWeightStrategy(StrategyConfig(name="ew",  description="Equal weight")),
    "Momentum":    MomentumStrategy(StrategyConfig(name="mom", description="Momentum"),
                                    lookback_period=60, top_k=2, long_only=True),
    "MinVariance": MinVarianceStrategy(StrategyConfig(name="mv",  description="Min Variance"),
                                       lookback_window=60),
}

print("  Running backtests …", end=" ", flush=True)
results = {}
for name, strat in strategies.items():
    results[name] = BacktestEngine(BacktestConfig(**BASE_CFG)).run(strat, DATA)
print("done.")


# ── 1. Full risk metrics table ────────────────────────────────────────────────

section("1. Full Risk Metrics — all strategies")

METRIC_LABELS = {
    "total_return":         ("Total Return",          "%"),
    "annualized_return":    ("Annualized Return",     "%"),
    "annualized_volatility":("Annualized Volatility", "%"),
    "sharpe_ratio":         ("Sharpe Ratio",          "x"),
    "sortino_ratio":        ("Sortino Ratio",         "x"),
    "max_drawdown":         ("Max Drawdown",          "%"),
    "calmar_ratio":         ("Calmar Ratio",          "x"),
    "win_rate":             ("Win Rate",              "%"),
    "profit_factor":        ("Profit Factor",         "x"),
    "var_95":               ("VaR 95% (daily)",       "%"),
    "cvar_95":              ("CVaR 95% (daily)",      "%"),
    "tracking_error":       ("Tracking Error",        "%"),
    "information_ratio":    ("Information Ratio",     "x"),
    "turnover":             ("Avg Turnover",          "%"),
}

col_w = 16
header = f"  {'Metric':28s}" + "".join(f"{n:>{col_w}}" for n in strategies)
print(header)
print("  " + "-" * (28 + col_w * len(strategies)))

for key, (label, unit) in METRIC_LABELS.items():
    row = f"  {label:28s}"
    for name, r in results.items():
        v = r.risk_metrics.get(key, float("nan"))
        if np.isnan(v):
            cell = "     —    "
        elif unit == "%":
            cell = f"{v*100:>+.2f}%"
        else:
            cell = f"{v:>+.4f} "
        row += f"{cell:>{col_w}}"
    print(row)


# ── 2. VaR / CVaR breakdown ───────────────────────────────────────────────────

section("2. VaR / CVaR at multiple confidence levels")

conf_levels = [0.90, 0.95, 0.99]
header2 = f"  {'Strategy':16s}" + "".join(
    f"{'VaR '+str(int(c*100))+'%':>12}  {'CVaR '+str(int(c*100))+'%':>12}"
    for c in conf_levels
)
print(header2)
print("  " + "-" * (16 + 28 * len(conf_levels)))

for name, r in results.items():
    returns = r.portfolio_values.pct_change().dropna()
    row = f"  {name:16s}"
    for c in conf_levels:
        var  = VaRCalculator.historical_var(returns, confidence=c)
        cvar = VaRCalculator.historical_cvar(returns, confidence=c)
        row += f"{var*100:>+10.2f}%  {cvar*100:>+10.2f}%  "
    print(row)

print("\n  Interpretation: VaR 99% is more negative than VaR 95% (deeper tail).")
print("  CVaR is always ≤ VaR — it's the average loss once VaR is breached.")


# ── 3. Chart: Metrics comparison bar chart ───────────────────────────────────

section("3. Generating charts …")

metric_groups = {
    "Return & Risk":     ["annualized_return", "annualized_volatility", "max_drawdown"],
    "Risk-Adjusted":     ["sharpe_ratio",       "sortino_ratio",         "calmar_ratio"],
    "Trade Quality":     ["win_rate",            "profit_factor"],
    "VaR":              ["var_95",              "cvar_95"],
}

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Phase 4 — Risk Analytics: Strategy Comparison", fontsize=15, y=0.99)

strat_names = list(strategies.keys())
colors      = [STRATEGY_COLORS[n] for n in strat_names]
x           = np.arange(len(strat_names))

for ax, (group_name, keys) in zip(axes.flat, metric_groups.items()):
    n_metrics = len(keys)
    width = 0.7 / n_metrics
    for i, key in enumerate(keys):
        vals = [results[n].risk_metrics.get(key, np.nan) for n in strat_names]
        # Scale to % for readability where unit is %
        if key in ("annualized_return", "annualized_volatility", "max_drawdown",
                   "win_rate", "var_95", "cvar_95"):
            vals = [v * 100 if not np.isnan(v) else 0.0 for v in vals]
            unit_label = "%"
        else:
            vals = [v if not np.isnan(v) else 0.0 for v in vals]
            unit_label = ""

        label = METRIC_LABELS[key][0]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      label=f"{label}{' (' + unit_label + ')' if unit_label else ''}",
                      color=colors, alpha=0.85)

    ax.set_title(group_name)
    ax.set_xticks(x)
    ax.set_xticklabels(strat_names, rotation=15, ha="right")
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, axis="y")

plt.tight_layout()
out1 = "demos/phase4_metrics_comparison.png"
fig.savefig(out1, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"  Saved → {out1}")
plt.close(fig)


# ── 4. Chart: Rolling 60-day VaR ─────────────────────────────────────────────

fig2, ax = plt.subplots(figsize=(14, 5))
fig2.suptitle("Phase 4 — Rolling 60-Day Historical VaR (95%)", fontsize=14)

for name, r in results.items():
    returns = r.portfolio_values.pct_change()
    rolling = VaRCalculator.rolling_var(returns, window=60, confidence=0.95) * 100
    ax.plot(rolling.index, rolling.values, label=name,
            color=STRATEGY_COLORS[name], lw=1.6)

ax.set_ylabel("VaR 95% — Daily Loss (%)")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True)
ax.invert_yaxis()   # losses are negative; flip so "worse" is lower

plt.tight_layout()
out2 = "demos/phase4_rolling_var.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight", facecolor=fig2.get_facecolor())
print(f"  Saved → {out2}")
plt.close(fig2)


# ── 5. Chart: Greeks surface heatmaps ────────────────────────────────────────

spot_range = np.linspace(70, 130, 40)
vol_range  = np.linspace(0.05, 0.60, 40)

surface = GreeksCalculator.compute_greeks_surface(
    spot_range    = spot_range,
    vol_range     = vol_range,
    strike        = 100.0,
    maturity      = 1.0,
    risk_free_rate= 0.05,
)

greek_keys   = ["delta", "gamma", "vega", "theta"]
greek_labels = ["Delta", "Gamma", "Vega", "Theta (per day)"]
cmaps        = ["RdYlGn", "plasma", "viridis", "RdYlGn_r"]

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
fig3.suptitle(
    "Phase 4 — Black-Scholes Greeks Surface\n"
    "(K=100, T=1yr, r=5%, Call option | x-axis: Vol, y-axis: Spot)",
    fontsize=13, y=1.01,
)

for ax, key, label, cmap in zip(axes3.flat, greek_keys, greek_labels, cmaps):
    mat = np.array(surface[key])  # shape (n_spots, n_vols)
    im = ax.pcolormesh(
        vol_range, spot_range, mat,
        cmap=cmap, shading="auto",
    )
    cbar = fig3.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#c8ccd8")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#c8ccd8")

    ax.set_title(label)
    ax.set_xlabel("Implied Volatility")
    ax.set_ylabel("Spot Price")

    # ATM line
    ax.axhline(100, color="white", lw=0.8, linestyle="--", alpha=0.5, label="ATM (S=K)")
    ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
out3 = "demos/phase4_greeks_surface.png"
fig3.savefig(out3, dpi=130, bbox_inches="tight", facecolor=fig3.get_facecolor())
print(f"  Saved → {out3}")
plt.close(fig3)

print("\nDone. Phase 4 is fully operational.\n")
