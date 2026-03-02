from __future__ import annotations

import math

import pandas as pd


class PerformanceMetrics:
    """Collection of static portfolio performance metric functions."""

    # ── [P4-03] Total return ────────────────────────────────────────────────

    @staticmethod
    def total_return(values: pd.Series) -> float:
        """(V_T - V_0) / V_0"""
        if len(values) < 2:
            return float("nan")
        return (values.iloc[-1] - values.iloc[0]) / values.iloc[0]

    # ── [P4-04] Annualized return ───────────────────────────────────────────

    @staticmethod
    def annualized_return(values: pd.Series, trading_days: int = 252) -> float:
        """(1 + total_return)^(252/N) - 1"""
        n = len(values)
        if n < 2:
            return float("nan")
        tr = PerformanceMetrics.total_return(values)
        return (1 + tr) ** (trading_days / n) - 1

    # ── [P4-05] Annualized volatility ───────────────────────────────────────

    @staticmethod
    def annualized_volatility(values: pd.Series, trading_days: int = 252) -> float:
        """std(daily_returns) * sqrt(252)"""
        returns = values.pct_change().dropna()
        if len(returns) < 2:
            return float("nan")
        return float(returns.std() * math.sqrt(trading_days))

    # ── [P4-06] Sharpe ratio ────────────────────────────────────────────────

    @staticmethod
    def sharpe_ratio(values: pd.Series, risk_free_rate: float = 0.05) -> float:
        """(ann_return - r_f) / ann_vol; NaN if ann_vol == 0"""
        ann_ret = PerformanceMetrics.annualized_return(values)
        ann_vol = PerformanceMetrics.annualized_volatility(values)
        if math.isnan(ann_vol) or ann_vol == 0:
            return float("nan")
        return (ann_ret - risk_free_rate) / ann_vol

    # ── [P4-07] Sortino ratio ───────────────────────────────────────────────

    @staticmethod
    def sortino_ratio(values: pd.Series, risk_free_rate: float = 0.05) -> float:
        """Like Sharpe but uses downside volatility (std of negative returns × √252)."""
        returns = values.pct_change().dropna()
        ann_ret = PerformanceMetrics.annualized_return(values)
        downside = returns[returns < 0]
        if len(downside) < 2:
            return float("nan")
        downside_vol = float(downside.std() * math.sqrt(252))
        if downside_vol == 0:
            return float("nan")
        return (ann_ret - risk_free_rate) / downside_vol

    # ── [P4-08] Max drawdown ────────────────────────────────────────────────

    @staticmethod
    def max_drawdown(values: pd.Series) -> float:
        """max((peak - trough) / peak) — returned as a negative number."""
        if len(values) < 2:
            return 0.0
        rolling_max = values.cummax()
        drawdown = (values - rolling_max) / rolling_max
        return float(drawdown.min())

    # ── [P4-09] Calmar ratio ────────────────────────────────────────────────

    @staticmethod
    def calmar_ratio(values: pd.Series) -> float:
        """annualized_return / abs(max_drawdown); NaN if max_drawdown == 0"""
        ann_ret = PerformanceMetrics.annualized_return(values)
        mdd = PerformanceMetrics.max_drawdown(values)
        if mdd == 0:
            return float("nan")
        return ann_ret / abs(mdd)

    # ── [P4-10] Win rate ────────────────────────────────────────────────────

    @staticmethod
    def win_rate(values: pd.Series) -> float:
        """Fraction of positive daily returns."""
        returns = values.pct_change().dropna()
        if len(returns) == 0:
            return float("nan")
        return float((returns > 0).mean())

    # ── [P4-11] Profit factor ───────────────────────────────────────────────

    @staticmethod
    def profit_factor(values: pd.Series) -> float:
        """sum(positive_returns) / abs(sum(negative_returns))"""
        returns = values.pct_change().dropna()
        positive = returns[returns > 0].sum()
        negative = returns[returns < 0].sum()
        if negative == 0:
            return float("nan")
        return float(positive / abs(negative))

    # ── [P4-12] Tracking error ──────────────────────────────────────────────

    @staticmethod
    def tracking_error(portfolio: pd.Series, benchmark: pd.Series) -> float:
        """std(portfolio_returns - benchmark_returns) * sqrt(252)"""
        port_ret = portfolio.pct_change().dropna()
        bench_ret = benchmark.pct_change().dropna()
        port_ret, bench_ret = port_ret.align(bench_ret, join="inner")
        active = port_ret - bench_ret
        if len(active) < 2:
            return float("nan")
        return float(active.std() * math.sqrt(252))

    # ── [P4-13] Information ratio ───────────────────────────────────────────

    @staticmethod
    def information_ratio(portfolio: pd.Series, benchmark: pd.Series) -> float:
        """(ann_portfolio_return - ann_bench_return) / tracking_error"""
        ann_port = PerformanceMetrics.annualized_return(portfolio)
        ann_bench = PerformanceMetrics.annualized_return(benchmark)
        te = PerformanceMetrics.tracking_error(portfolio, benchmark)
        if math.isnan(te) or te == 0:
            return float("nan")
        return (ann_port - ann_bench) / te

    # ── [P4-14] Turnover ────────────────────────────────────────────────────

    @staticmethod
    def turnover(weights_history: pd.DataFrame) -> float:
        """Mean of total absolute weight change at each rebalancing date."""
        if weights_history is None or len(weights_history) < 2:
            return float("nan")
        changes = weights_history.diff().dropna()
        if len(changes) == 0:
            return float("nan")
        return float(changes.abs().sum(axis=1).mean())

    # ── [P4-15] compute_all ─────────────────────────────────────────────────

    @staticmethod
    def compute_all(
        values: pd.Series,
        benchmark: pd.Series | None = None,
        weights_history: pd.DataFrame | None = None,
        risk_free_rate: float = 0.05,
    ) -> dict:
        """Call all metrics and return a single dict ready for the API response."""
        from core.risk.var import VaRCalculator  # local import — no circular dep

        returns = values.pct_change().dropna()

        result: dict = {
            "total_return": PerformanceMetrics.total_return(values),
            "annualized_return": PerformanceMetrics.annualized_return(values),
            "annualized_volatility": PerformanceMetrics.annualized_volatility(values),
            "sharpe_ratio": PerformanceMetrics.sharpe_ratio(values, risk_free_rate),
            "sortino_ratio": PerformanceMetrics.sortino_ratio(values, risk_free_rate),
            "max_drawdown": PerformanceMetrics.max_drawdown(values),
            "calmar_ratio": PerformanceMetrics.calmar_ratio(values),
            "win_rate": PerformanceMetrics.win_rate(values),
            "profit_factor": PerformanceMetrics.profit_factor(values),
            "var_95": VaRCalculator.historical_var(returns, confidence=0.95),
            "cvar_95": VaRCalculator.historical_cvar(returns, confidence=0.95),
        }

        if benchmark is not None and len(benchmark) > 1:
            result["tracking_error"] = PerformanceMetrics.tracking_error(values, benchmark)
            result["information_ratio"] = PerformanceMetrics.information_ratio(values, benchmark)

        if weights_history is not None and not weights_history.empty:
            result["turnover"] = PerformanceMetrics.turnover(weights_history)
        else:
            result["turnover"] = float("nan")

        return result
