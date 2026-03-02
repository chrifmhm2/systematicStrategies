"""
Phase 5 Demo — FastAPI Backend
================================
Run from backend/ with:
    .venv/bin/python demos/phase5_demo.py

Exercises the live API via FastAPI TestClient (no server needed).
Produces 3 figures saved to demos/:
    phase5_strategy_comparison.png  — Portfolio curves for 3 strategies via /api/backtest/compare
    phase5_option_pricing.png       — BS call/put prices + delta across strikes via /api/pricing/option
    phase5_risk_dashboard.png       — Risk metrics bar chart for each strategy via /api/risk/analyze

Terminal output shows endpoint responses and key numbers.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from fastapi.testclient import TestClient

from main import app

# ── Global style (matches phase1–phase4) ──────────────────────────────────────

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
CYAN   = "#8be9fd"

client = TestClient(app)


def section(title: str) -> None:
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GET /api/strategies  — list all registered strategies
# ─────────────────────────────────────────────────────────────────────────────

section("1. GET /api/strategies")

resp = client.get("/api/strategies")
assert resp.status_code == 200
strategies = resp.json()["strategies"]
print(f"  {len(strategies)} strategies registered:")
for s in strategies:
    print(f"    [{s['family']:12s}]  {s['id']:30s}  {s['description'][:50]}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  POST /api/backtest/compare — run 3 strategies side by side
# ─────────────────────────────────────────────────────────────────────────────

section("2. POST /api/backtest/compare — 3 strategies on 5 simulated assets")

compare_payload = {
    "strategies": [
        {"strategy_id": "EqualWeightStrategy",  "params": {}},
        {"strategy_id": "MomentumStrategy",     "params": {"lookback_period": 60, "top_k": 2}},
        {"strategy_id": "MinVarianceStrategy",  "params": {"lookback_window": 60}},
    ],
    "symbols":       ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"],
    "start_date":    "2022-01-03",
    "end_date":      "2023-12-29",
    "initial_value": 100_000.0,
    "data_source":   "simulated",
}

print("  Running compare … ", end="", flush=True)
resp = client.post("/api/backtest/compare", json=compare_payload)
assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
compare_results = resp.json()
print(f"done — {len(compare_results)} results returned.")

STRAT_NAMES  = ["EqualWeight", "Momentum", "MinVariance"]
STRAT_COLORS = [BLUE, GREEN, ORANGE]

print(f"\n  {'Strategy':20s}  {'Total Ret':>10}  {'Sharpe':>8}  {'MaxDD':>8}  {'Days':>6}")
print("  " + "-" * 60)
for name, result in zip(STRAT_NAMES, compare_results):
    pv      = result["portfolio_values"]
    metrics = result["risk_metrics"]
    n_days  = len(pv)
    tr      = (metrics.get("total_return") or 0) * 100
    sharpe  = metrics.get("sharpe_ratio") or float("nan")
    mdd     = (metrics.get("max_drawdown") or 0) * 100
    print(f"  {name:20s}  {tr:>+9.2f}%  {sharpe:>8.3f}  {mdd:>+8.2f}%  {n_days:>6}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  POST /api/pricing/option — price calls across a range of strikes
# ─────────────────────────────────────────────────────────────────────────────

section("3. POST /api/pricing/option — BS call/put prices and delta vs strike")

strikes   = np.linspace(70, 130, 25)
call_prices, put_prices, call_deltas, put_deltas = [], [], [], []

for K in strikes:
    r_call = client.post("/api/pricing/option", json={
        "option_type": "call", "S": 100.0, "K": float(K),
        "T": 1.0, "r": 0.05, "sigma": 0.20, "method": "bs",
    })
    r_put = client.post("/api/pricing/option", json={
        "option_type": "put",  "S": 100.0, "K": float(K),
        "T": 1.0, "r": 0.05, "sigma": 0.20, "method": "bs",
    })
    assert r_call.status_code == 200
    assert r_put.status_code  == 200
    call_prices.append(r_call.json()["price"])
    put_prices.append(r_put.json()["price"])
    call_deltas.append(r_call.json()["greeks"]["delta"])
    put_deltas.append(r_put.json()["greeks"]["delta"])

atm_call = client.post("/api/pricing/option", json={
    "option_type": "call", "S": 100.0, "K": 100.0,
    "T": 1.0, "r": 0.05, "sigma": 0.20, "method": "bs",
}).json()

print(f"  ATM call (S=K=100, T=1, r=5%, σ=20%):")
print(f"    Price  = {atm_call['price']:.4f}")
greeks = atm_call["greeks"]
for g, v in greeks.items():
    if v is not None:
        print(f"    {g:6s} = {v:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  POST /api/risk/analyze — risk dashboard for each strategy
# ─────────────────────────────────────────────────────────────────────────────

section("4. POST /api/risk/analyze — risk metrics per strategy")

risk_results = {}
for name, result in zip(STRAT_NAMES, compare_results):
    payload = {
        "portfolio_values": result["portfolio_values"],
        "risk_free_rate":   0.05,
    }
    r = client.post("/api/risk/analyze", json=payload)
    assert r.status_code == 200
    risk_results[name] = r.json()

RISK_KEYS   = ["annualized_return", "annualized_volatility", "sharpe_ratio",
               "sortino_ratio", "max_drawdown", "var_95"]
RISK_LABELS = ["Ann. Return", "Ann. Vol", "Sharpe", "Sortino", "Max DD", "VaR 95%"]
RISK_UNITS  = ["%", "%", "x", "x", "%", "%"]

print(f"\n  {'Metric':20s}" + "".join(f"  {n:>14}" for n in STRAT_NAMES))
print("  " + "-" * (20 + 18 * len(STRAT_NAMES)))
for key, label, unit in zip(RISK_KEYS, RISK_LABELS, RISK_UNITS):
    row = f"  {label:20s}"
    for name in STRAT_NAMES:
        v = risk_results[name].get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            row += f"  {'—':>14}"
        elif unit == "%":
            row += f"  {v*100:>+13.2f}%"
        else:
            row += f"  {v:>+14.4f}"
    print(row)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Chart 1 — Portfolio value curves from /api/backtest/compare
# ─────────────────────────────────────────────────────────────────────────────

section("5. Generating charts …")

fig1, ax1 = plt.subplots(figsize=(14, 6))
fig1.suptitle("Phase 5 — Strategy Comparison via POST /api/backtest/compare", fontsize=14)

for name, result, color in zip(STRAT_NAMES, compare_results, STRAT_COLORS):
    pv   = result["portfolio_values"]
    dates = pd.to_datetime(list(pv.keys()))
    vals  = np.array(list(pv.values())) / 100_000.0 * 100  # index base 100
    ax1.plot(dates, vals, label=name, color=color, lw=1.8)

ax1.axhline(100, color=GRAY, lw=0.8, linestyle="--", alpha=0.6, label="Benchmark (base 100)")
ax1.set_ylabel("Portfolio Value (Base 100)")
ax1.set_xlabel("Date")
ax1.legend()
ax1.grid(True)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

plt.tight_layout()
out1 = "demos/phase5_strategy_comparison.png"
fig1.savefig(out1, dpi=130, bbox_inches="tight", facecolor=fig1.get_facecolor())
print(f"  Saved → {out1}")
plt.close(fig1)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Chart 2 — Option prices and delta via /api/pricing/option
# ─────────────────────────────────────────────────────────────────────────────

fig2, (ax_price, ax_delta) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle(
    "Phase 5 — Black-Scholes Prices & Delta via POST /api/pricing/option\n"
    "(S=100, T=1yr, r=5%, σ=20%)",
    fontsize=13,
)

# Price panel
ax_price.plot(strikes, call_prices, color=BLUE,   lw=2.0, label="Call price")
ax_price.plot(strikes, put_prices,  color=RED,    lw=2.0, label="Put price")
ax_price.axvline(100, color=GRAY, lw=0.8, linestyle="--", alpha=0.6, label="ATM (K=100)")
ax_price.fill_between(strikes, call_prices, alpha=0.08, color=BLUE)
ax_price.fill_between(strikes, put_prices,  alpha=0.08, color=RED)
ax_price.set_xlabel("Strike (K)")
ax_price.set_ylabel("Option Price")
ax_price.set_title("Call / Put Prices vs Strike")
ax_price.legend()
ax_price.grid(True)

# Delta panel
ax_delta.plot(strikes, call_deltas, color=BLUE,   lw=2.0, label="Call delta")
ax_delta.plot(strikes, put_deltas,  color=RED,    lw=2.0, label="Put delta")
ax_delta.axvline(100, color=GRAY, lw=0.8, linestyle="--", alpha=0.6, label="ATM")
ax_delta.axhline(0,   color=GRAY, lw=0.5, alpha=0.4)
ax_delta.axhline(0.5, color=GREEN, lw=0.8, linestyle=":", alpha=0.5, label="Δ = 0.5 (ATM call)")
ax_delta.set_xlabel("Strike (K)")
ax_delta.set_ylabel("Delta")
ax_delta.set_title("Delta vs Strike")
ax_delta.legend()
ax_delta.grid(True)

plt.tight_layout()
out2 = "demos/phase5_option_pricing.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight", facecolor=fig2.get_facecolor())
print(f"  Saved → {out2}")
plt.close(fig2)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Chart 3 — Risk dashboard via /api/risk/analyze
# ─────────────────────────────────────────────────────────────────────────────

fig3, axes3 = plt.subplots(2, 3, figsize=(16, 9))
fig3.suptitle("Phase 5 — Risk Dashboard via POST /api/risk/analyze", fontsize=14)

x = np.arange(len(STRAT_NAMES))
bar_colors = STRAT_COLORS

for ax, key, label, unit in zip(axes3.flat, RISK_KEYS, RISK_LABELS, RISK_UNITS):
    vals = []
    for name in STRAT_NAMES:
        v = risk_results[name].get(key)
        vals.append((v or 0.0) * (100 if unit == "%" else 1))

    bars = ax.bar(x, vals, color=bar_colors, alpha=0.85, width=0.5)

    # value labels on bars
    for bar, v in zip(bars, vals):
        h = bar.get_height()
        y_pos = h + (abs(max(vals, default=1) - min(vals, default=0)) * 0.02)
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{v:+.2f}{'%' if unit == '%' else ''}",
                ha="center", va="bottom", fontsize=9, color="#c8ccd8")

    ax.set_title(label)
    ax.set_xticks(x)
    ax.set_xticklabels(STRAT_NAMES, rotation=15, ha="right", fontsize=9)
    ax.axhline(0, color=GRAY, lw=0.7)
    ax.grid(True, axis="y")

plt.tight_layout()
out3 = "demos/phase5_risk_dashboard.png"
fig3.savefig(out3, dpi=130, bbox_inches="tight", facecolor=fig3.get_facecolor())
print(f"  Saved → {out3}")
plt.close(fig3)

print("\nDone. Phase 5 (FastAPI Backend) is fully operational.\n")
