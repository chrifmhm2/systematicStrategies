"""
Phase 2 Demo — Strategy Framework
==================================
Run: .venv/bin/python demos/phase2_demo.py

Shows:
  1. All registered strategies (the plugin catalog)
  2. Each strategy's output on simulated price data
  3. Registry usage: create a strategy by name
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from core.strategies import StrategyRegistry, StrategyConfig
from core.strategies.allocation.equal_weight import EqualWeightStrategy
from core.strategies.allocation.min_variance import MinVarianceStrategy
from core.strategies.allocation.max_sharpe import MaxSharpeStrategy
from core.strategies.allocation.risk_parity import RiskParityStrategy
from core.strategies.signal.momentum import MomentumStrategy
from core.strategies.signal.mean_reversion import MeanReversionStrategy
from core.strategies.hedging.delta_hedge import DeltaHedgeStrategy

# ---------------------------------------------------------------------------
# Shared simulated price data
# ---------------------------------------------------------------------------

SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
N_DAYS  = 300

rng = np.random.default_rng(42)
log_ret = rng.normal(0.0003, 0.015, size=(N_DAYS, len(SYMBOLS)))
prices = 100.0 * np.exp(np.cumsum(log_ret, axis=0))
idx = pd.date_range("2023-01-02", periods=N_DAYS, freq="B")
price_df = pd.DataFrame(prices, index=idx, columns=SYMBOLS)

TODAY = price_df.index[-1].date()


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def show_weights(pw) -> None:
    print(f"  Timestamp : {pw.timestamp}")
    print(f"  Cash      : {pw.cash_weight:.4f}")
    for sym, w in sorted(pw.weights.items()):
        bar = "█" * int(w * 40)
        print(f"  {sym:6s}  {w:6.4f}  {bar}")
    print(f"  Total     : {pw.total_weight():.6f}")


# ---------------------------------------------------------------------------
# 1. Plugin catalog
# ---------------------------------------------------------------------------

section("1. Registered strategies (StrategyRegistry.list_strategies())")

for s in StrategyRegistry.list_strategies():
    print(f"  [{s['family']:10s}]  {s['name']:30s}  {s['description'][:55]}")

# ---------------------------------------------------------------------------
# 2. Equal Weight
# ---------------------------------------------------------------------------

section("2. EqualWeightStrategy — 1/N allocation")

strat = EqualWeightStrategy(StrategyConfig(name="ew", description="equal weight"))
pw = strat.compute_weights(TODAY, price_df)
show_weights(pw)

# ---------------------------------------------------------------------------
# 3. Min Variance
# ---------------------------------------------------------------------------

section("3. MinVarianceStrategy — minimise portfolio volatility")

strat = MinVarianceStrategy(StrategyConfig(name="mv", description="min var"), lookback_window=60)
pw = strat.compute_weights(TODAY, price_df)
show_weights(pw)

# ---------------------------------------------------------------------------
# 4. Max Sharpe
# ---------------------------------------------------------------------------

section("4. MaxSharpeStrategy — best return per unit of risk")

strat = MaxSharpeStrategy(
    StrategyConfig(name="ms", description="max sharpe"),
    lookback_window=60,
    risk_free_rate_override=0.05,
)
pw = strat.compute_weights(TODAY, price_df)
show_weights(pw)

# ---------------------------------------------------------------------------
# 5. Risk Parity
# ---------------------------------------------------------------------------

section("5. RiskParityStrategy — equal risk contribution per asset")

strat = RiskParityStrategy(StrategyConfig(name="rp", description="risk parity"), lookback_window=60)
pw = strat.compute_weights(TODAY, price_df)
show_weights(pw)

# Verify risk contributions
w_arr = np.array([pw.weights[s] for s in SYMBOLS])
returns = price_df.tail(60).pct_change().dropna()
cov = returns.cov().values
port_var = float(w_arr @ cov @ w_arr)
mrc = cov @ w_arr
rc = w_arr * mrc / port_var
print(f"\n  Risk contributions (should each be ~{1/len(SYMBOLS):.3f}):")
for sym, rci in zip(SYMBOLS, rc):
    print(f"  {sym:6s}  RC = {rci:.4f}")

# ---------------------------------------------------------------------------
# 6. Momentum
# ---------------------------------------------------------------------------

section("6. MomentumStrategy — buy top-3 by trailing 252-day return")

strat = MomentumStrategy(
    StrategyConfig(name="mom", description="momentum"),
    lookback_period=252,
    top_k=3,
    long_only=True,
)
pw = strat.compute_weights(TODAY, price_df)
show_weights(pw)

trailing = (price_df.iloc[-1] / price_df.iloc[0] - 1).sort_values(ascending=False)
print("\n  Full trailing-return ranking:")
for sym, r in trailing.items():
    mark = " ← selected" if pw.weights.get(sym, 0) > 0 else ""
    print(f"  {sym:6s}  {r:+.2%}{mark}")

# ---------------------------------------------------------------------------
# 7. Mean Reversion
# ---------------------------------------------------------------------------

section("7. MeanReversionStrategy — buy oversold (z < -2)")

strat = MeanReversionStrategy(
    StrategyConfig(name="mr", description="mean reversion"),
    lookback_window=20,
    z_threshold=2.0,
)
pw = strat.compute_weights(TODAY, price_df)
show_weights(pw)

# Show z-scores
window = price_df.tail(20)
ma = window.mean()
std = window.std().replace(0, np.nan)
z = (price_df.iloc[-1] - ma) / std
print("\n  Z-scores (signal if |z| > 2.0):")
for sym in SYMBOLS:
    signal = " ← BUY signal" if z[sym] < -2.0 else (" ← SELL signal" if z[sym] > 2.0 else "")
    print(f"  {sym:6s}  z = {z[sym]:+.2f}{signal}")

# ---------------------------------------------------------------------------
# 8. Delta Hedge
# ---------------------------------------------------------------------------

section("8. DeltaHedgeStrategy — hedge a basket call option")

strat = DeltaHedgeStrategy(
    StrategyConfig(name="dh", description="delta hedge"),
    strike=100.0,
    maturity_years=0.5,
    n_simulations=20_000,
    risk_free_rate=0.05,
    volatilities=[0.25, 0.20, 0.22, 0.18, 0.30],
    correlation=[
        [1.0,  0.6,  0.5,  0.4,  0.3],
        [0.6,  1.0,  0.55, 0.45, 0.25],
        [0.5,  0.55, 1.0,  0.5,  0.2],
        [0.4,  0.45, 0.5,  1.0,  0.2],
        [0.3,  0.25, 0.2,  0.2,  1.0],
    ],
)
pw = strat.compute_weights(TODAY, price_df)
show_weights(pw)
print("  (cash_weight > 0 because delta hedge never uses 100% capital)")

# ---------------------------------------------------------------------------
# 9. Registry: create by name (as the API will do it)
# ---------------------------------------------------------------------------

section("9. StrategyRegistry.create() — instantiate by string name")

strat = StrategyRegistry.create("MomentumStrategy", {
    "lookback_period": 60,
    "top_k": 2,
    "long_only": True,
})
pw = strat.compute_weights(TODAY, price_df)
print(f"  Created via registry: {type(strat).__name__}")
print(f"  lookback_period = {strat.lookback_period},  top_k = {strat.top_k}")
show_weights(pw)

print("\n\nDone. Phase 2 is fully operational.\n")
