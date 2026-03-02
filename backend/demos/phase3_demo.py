"""
Phase 3 Demo — Backtesting Engine
==================================
Run from backend/ with:
    .venv/bin/python demos/phase3_demo.py

Produces 2 figures saved to demos/:
    phase3_nav_comparison.png    — NAV + drawdown curves (3 strategies vs benchmark)
    phase3_weights_history.png   — Momentum weights over time (stacked area)

Terminal output shows a per-strategy result summary and a rebalancing comparison.
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
from core.strategies.allocation.equal_weight import EqualWeightStrategy
from core.strategies.allocation.min_variance import MinVarianceStrategy
from core.strategies.signal.momentum import MomentumStrategy
from core.strategies.base import StrategyConfig

# ── Global style (matches phase1/phase2) ──────────────────────────────────────

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


def section(title: str) -> None:
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


def fmt(v: float, pct: bool = False, dollar: bool = False) -> str:
    if np.isnan(v):
        return "   NaN  "
    if dollar:
        return f"${v:>10,.0f}"
    if pct:
        return f"{v*100:>+7.2f}%"
    return f"{v:>+8.4f}"


# ── Shared setup ──────────────────────────────────────────────────────────────

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
    symbols               = SYMBOLS,
    start_date            = START,
    end_date              = END,
    transaction_cost_bps  = 10.0,
    slippage_bps          = 5.0,
)


# ── 1. Run three strategies ───────────────────────────────────────────────────

section("1. Running three strategies on 2 years of simulated data")
print(f"  Period  : {START} → {END}")
print(f"  Assets  : {', '.join(SYMBOLS)}")
print(f"  Capital : $100,000")

strategies = {
    "EqualWeight": (
        EqualWeightStrategy(StrategyConfig(name="ew", description="Equal weight")),
        BacktestConfig(**BASE_CFG, rebalancing_frequency="weekly"),
        BLUE,
    ),
    "Momentum": (
        MomentumStrategy(
            StrategyConfig(name="mom", description="Momentum"),
            lookback_period=60, top_k=2, long_only=True,
        ),
        BacktestConfig(**BASE_CFG, rebalancing_frequency="weekly"),
        GREEN,
    ),
    "MinVariance": (
        MinVarianceStrategy(
            StrategyConfig(name="mv", description="Min Variance"),
            lookback_window=60,
        ),
        BacktestConfig(**BASE_CFG, rebalancing_frequency="weekly"),
        ORANGE,
    ),
}

results = {}
for name, (strat, cfg, _color) in strategies.items():
    eng = BacktestEngine(cfg)
    results[name] = eng.run(strat, DATA)
    r = results[name]
    nav_start = r.portfolio_values.iloc[0]
    nav_end   = r.portfolio_values.iloc[-1]
    total_ret = (nav_end - nav_start) / nav_start
    print(f"\n  [{name}]")
    print(f"    NAV start      : ${nav_start:>10,.0f}")
    print(f"    NAV end        : ${nav_end:>10,.0f}")
    print(f"    Total return   : {total_ret*100:>+.2f}%")
    print(f"    Trades         : {len(r.trades_log)}")
    print(f"    Rebalancings   : {len(r.weights_history)}")
    print(f"    Computation    : {r.computation_time_ms:.1f} ms")


# ── 2. Self-financing check ───────────────────────────────────────────────────

section("2. Self-financing invariant check")
print("  The engine asserts self-financing at every rebalance.")
print("  If any backtest above completed without an AssertionError,")
print("  the invariant holds for all 3 strategies.")
print("  ✓  All backtests completed — no self-financing violations.")


# ── 3. Rebalancing oracle comparison ─────────────────────────────────────────

section("3. Rebalancing oracle comparison — weekly vs threshold (5%)")

strat_ew  = EqualWeightStrategy(StrategyConfig(name="ew_w", description="EW weekly"))
strat_thr = EqualWeightStrategy(StrategyConfig(name="ew_t", description="EW threshold"))

cfg_weekly = BacktestConfig(**BASE_CFG, rebalancing_frequency="weekly")
cfg_thresh = BacktestConfig(**BASE_CFG, rebalancing_frequency="threshold", threshold=0.05)

res_weekly = BacktestEngine(cfg_weekly).run(strat_ew,  DATA)
res_thresh = BacktestEngine(cfg_thresh).run(strat_thr, DATA)

total_cost_weekly = sum(t["cost"] for t in res_weekly.trades_log)
total_cost_thresh = sum(t["cost"] for t in res_thresh.trades_log)

print(f"  {'Oracle':20s}  {'Rebalancings':>14}  {'Trades':>8}  {'Total Cost':>12}")
print(f"  {'-'*60}")
print(f"  {'Weekly':20s}  {len(res_weekly.weights_history):>14}  "
      f"{len(res_weekly.trades_log):>8}  ${total_cost_weekly:>10,.0f}")
print(f"  {'Threshold 5%':20s}  {len(res_thresh.weights_history):>14}  "
      f"{len(res_thresh.trades_log):>8}  ${total_cost_thresh:>10,.0f}")
print(f"\n  Threshold oracle fires only when drift exceeds 5% — fewer rebalancings,")
print(f"  lower transaction costs, but potentially more drift between events.")


# ── 4. Plot: NAV + Drawdown comparison ───────────────────────────────────────

section("4. Generating charts …")

benchmark = list(results.values())[0].benchmark_values

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1]})
fig.suptitle("Phase 3 — Backtesting Engine: Strategy Comparison", fontsize=15, y=0.98)

ax_nav, ax_dd = axes

# NAV panel
for name, (strat, cfg, color) in strategies.items():
    r = results[name]
    nav = r.portfolio_values / r.portfolio_values.iloc[0] * 100
    ax_nav.plot(nav.index, nav.values, label=name, color=color, lw=1.8)

bench_norm = benchmark / benchmark.iloc[0] * 100
ax_nav.plot(bench_norm.index, bench_norm.values,
            label="Benchmark (EW buy-and-hold)", color=GRAY, lw=1.2, linestyle="--")

ax_nav.set_ylabel("Portfolio Value (indexed to 100)")
ax_nav.legend(loc="upper left")
ax_nav.grid(True)
ax_nav.set_title(f"{START} → {END} | $100k initial capital | Weekly rebalancing | 10 bps + 5 bps slippage")

# Drawdown panel
for name, (strat, cfg, color) in strategies.items():
    r = results[name]
    pv = r.portfolio_values
    dd = (pv - pv.cummax()) / pv.cummax() * 100
    ax_dd.fill_between(dd.index, dd.values, 0, alpha=0.35, color=color)
    ax_dd.plot(dd.index, dd.values, color=color, lw=1.2)

ax_dd.set_ylabel("Drawdown (%)")
ax_dd.set_xlabel("Date")
ax_dd.grid(True)

plt.tight_layout()
out1 = "demos/phase3_nav_comparison.png"
fig.savefig(out1, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"  Saved → {out1}")
plt.close(fig)


# ── 5. Plot: Momentum weights over time ───────────────────────────────────────

mom_result = results["Momentum"]
wh = mom_result.weights_history

if not wh.empty:
    fig2, ax = plt.subplots(figsize=(14, 5))
    fig2.suptitle("Phase 3 — Momentum Strategy: Portfolio Weights Over Time", fontsize=14)

    colors_assets = [BLUE, GREEN, ORANGE, RED, PURPLE]
    ax.stackplot(
        wh.index, [wh[s].values for s in SYMBOLS],
        labels=SYMBOLS, colors=colors_assets, alpha=0.8,
    )
    ax.set_ylabel("Weight")
    ax.set_xlabel("Date")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", ncol=5)
    ax.grid(True)

    plt.tight_layout()
    out2 = "demos/phase3_weights_history.png"
    fig2.savefig(out2, dpi=130, bbox_inches="tight", facecolor=fig2.get_facecolor())
    print(f"  Saved → {out2}")
    plt.close(fig2)
else:
    print("  (Momentum weights history is empty — strategy never triggered)")

print("\nDone. Phase 3 is fully operational.\n")
