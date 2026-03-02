from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from core.backtester.costs import TransactionCostModel
from core.risk.metrics import PerformanceMetrics
from core.backtester.rebalancing import PeriodicRebalancing, RebalancingOracle, ThresholdRebalancing
from core.data.base import IDataProvider
from core.models.portfolio import Portfolio, Position
from core.models.results import BacktestResult
from core.strategies.base import IStrategy


@dataclass
class BacktestConfig:
    """
    Configuration for a single backtest run.

    Parameters
    ----------
    initial_value : float
        Starting portfolio value in currency units (default $100,000).
    start_date : date | None
        First date of the backtest (inclusive). If None, uses the first date
        returned by the data provider.
    end_date : date | None
        Last date of the backtest (inclusive). If None, uses the last date
        returned by the data provider.
    symbols : list[str] | None
        Universe of assets to trade. Required unless the data provider infers it.
    rebalancing_frequency : str
        "daily", "weekly", "monthly", or "threshold". Controls when the strategy
        is called to recompute weights.
    transaction_cost_bps : float
        Broker commission in basis points. Passed to TransactionCostModel.
    slippage_bps : float
        Market impact / bid-ask spread in basis points.
    threshold : float
        Drift threshold for ThresholdRebalancing (only used when
        rebalancing_frequency == "threshold"). Default 5%.
    """

    initial_value: float = 100_000.0
    start_date: date | None = None
    end_date: date | None = None
    symbols: list[str] = field(default_factory=list)
    rebalancing_frequency: str = "weekly"
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    threshold: float = 0.05


class BacktestEngine:
    """
    Drives a backtest of an IStrategy over a historical price series.

    Design invariants enforced by the engine:
    - No look-ahead bias: `prices.loc[:t]` is the only data passed to compute_weights.
    - Self-financing: portfolio value before rebalancing = value after + transaction costs.
    - Required history: strategy.compute_weights is not called until at least
      `strategy.required_history_days` rows of prices are available.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def run(self, strategy: IStrategy, data_provider: IDataProvider) -> BacktestResult:
        """
        Execute the backtest.

        Parameters
        ----------
        strategy : IStrategy
            A fully constructed strategy instance.
        data_provider : IDataProvider
            Source of historical price data (simulated or real).

        Returns
        -------
        BacktestResult
            Complete backtest output: NAV series, benchmark, weights history, trades log.
        """
        start_time = time.perf_counter()

        cfg = self.config
        symbols = cfg.symbols

        # ── 1. Fetch price data ─────────────────────────────────────────────
        prices = data_provider.get_prices(symbols, cfg.start_date, cfg.end_date)
        prices = prices.dropna(how="all")  # drop dates where all prices are NaN

        risk_free_rate = data_provider.get_risk_free_rate(
            prices.index[0].date() if cfg.start_date is None else cfg.start_date
        )
        dt = 1 / 252  # one trading day in years

        # ── 2. Build oracle and cost model ─────────────────────────────────
        oracle = self._build_oracle(cfg)
        cost_model = TransactionCostModel(
            commission_bps=cfg.transaction_cost_bps,
            slippage_bps=cfg.slippage_bps,
        )

        # ── 3. Initialise portfolio ─────────────────────────────────────────
        portfolio = Portfolio(positions={}, cash=cfg.initial_value)

        # ── 4. Accumulators ────────────────────────────────────────────────
        portfolio_values: dict = {}
        weights_records: dict = {}
        trades_log: list[dict] = []

        # ── 5. Main backtest loop ───────────────────────────────────────────
        for t in prices.index:
            t_date = t.date()
            current_prices = prices.loc[t].to_dict()

            # Update position prices to current market prices
            for sym, pos in portfolio.positions.items():
                if sym in current_prices and not pd.isna(current_prices[sym]):
                    pos.price = current_prices[sym]

            # Cash accrues risk-free interest each trading day
            portfolio.cash *= 1.0 + risk_free_rate * dt

            # Mark-to-market NAV
            valid_prices = {
                sym: p for sym, p in current_prices.items() if not pd.isna(p)
            }
            V_t = portfolio.total_value(valid_prices)
            portfolio_values[t] = V_t

            # Only trade once we have enough history
            history = prices.loc[:t]
            if len(history) < strategy.required_history_days:
                continue

            # Ask the oracle whether to rebalance today
            if not oracle.should_rebalance(t_date, portfolio):
                continue

            # ── Rebalance ─────────────────────────────────────────────────
            portfolio.date = t_date
            pw = strategy.compute_weights(t_date, history, portfolio)

            # Compute new quantities and trade costs
            total_costs = 0.0
            new_positions: dict[str, Position] = {}
            step_trades: list[dict] = []

            for sym in symbols:
                price = current_prices.get(sym)
                if price is None or pd.isna(price) or price <= 0:
                    continue

                target_weight = pw.weights.get(sym, 0.0)
                new_qty = target_weight * V_t / price
                old_qty = portfolio.positions[sym].quantity if sym in portfolio.positions else 0.0
                delta_qty = new_qty - old_qty

                trade_value = delta_qty * price
                cost = cost_model.compute_cost(trade_value) if abs(delta_qty) > 1e-9 else 0.0
                total_costs += cost

                new_positions[sym] = Position(symbol=sym, quantity=new_qty, price=price)

                if abs(delta_qty) > 1e-9:
                    step_trades.append({
                        "date": str(t_date),
                        "symbol": sym,
                        "quantity": delta_qty,
                        "price": price,
                        "cost": cost,
                    })

            # Self-financing: new_cash = V_t - equity_deployed - total_costs
            equity_deployed = sum(
                pos.quantity * current_prices[sym]
                for sym, pos in new_positions.items()
                if sym in current_prices and not pd.isna(current_prices[sym])
            )
            new_cash = V_t - equity_deployed - total_costs

            # Assert self-financing invariant (within floating-point tolerance)
            reconstructed_value = equity_deployed + new_cash
            assert abs(reconstructed_value - (V_t - total_costs)) < 1e-6, (
                f"Self-financing violated at {t_date}: "
                f"reconstructed={reconstructed_value:.6f}, "
                f"expected={V_t - total_costs:.6f}"
            )

            portfolio.positions = new_positions
            portfolio.cash = new_cash

            weights_records[t] = pw.weights.copy()
            trades_log.extend(step_trades)

            # Notify threshold oracle of new target weights
            if isinstance(oracle, ThresholdRebalancing):
                oracle.update_target_weights(pw.weights)

        # ── 6. Build portfolio_values Series ───────────────────────────────
        pv_series = pd.Series(portfolio_values, name="portfolio_value")
        pv_series.index = pd.DatetimeIndex(pv_series.index)

        # ── 7. Benchmark — equal-weight buy-and-hold ───────────────────────
        benchmark_series = self._compute_benchmark(prices, symbols, cfg.initial_value)

        # ── 8. Weights history DataFrame ───────────────────────────────────
        if weights_records:
            weights_df = pd.DataFrame(weights_records).T
            weights_df.index = pd.DatetimeIndex(weights_df.index)
            weights_df = weights_df.reindex(columns=symbols, fill_value=0.0)
        else:
            weights_df = pd.DataFrame(columns=symbols)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # ── 9. Compute risk metrics [P4-16] ────────────────────────────────
        risk_metrics = PerformanceMetrics.compute_all(
            values=pv_series,
            benchmark=benchmark_series,
            weights_history=weights_df,
            risk_free_rate=risk_free_rate,
        )

        return BacktestResult(
            portfolio_values=pv_series,
            benchmark_values=benchmark_series,
            weights_history=weights_df,
            trades_log=trades_log,
            risk_metrics=risk_metrics,
            config=cfg,
            strategy_name=strategy.config.name,
            computation_time_ms=elapsed_ms,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _build_oracle(cfg: BacktestConfig) -> RebalancingOracle:
        if cfg.rebalancing_frequency == "threshold":
            return ThresholdRebalancing(threshold=cfg.threshold)
        return PeriodicRebalancing(frequency=cfg.rebalancing_frequency)

    @staticmethod
    def _compute_benchmark(
        prices: pd.DataFrame,
        symbols: list[str],
        initial_value: float,
    ) -> pd.Series:
        """Equal-weight buy-and-hold benchmark."""
        available = [s for s in symbols if s in prices.columns]
        if not available:
            return pd.Series(dtype=float)

        first_prices = prices[available].iloc[0]
        n = len(available)
        # quantity of each asset purchased on day 1
        qty = (initial_value / n) / first_prices  # Series: symbol → quantity
        benchmark = (prices[available] * qty).sum(axis=1)
        benchmark.name = "benchmark"
        return benchmark
