"""
[P5-06] Backtest and compare routes.
POST /backtest         → run a single backtest
POST /backtest/compare → run multiple backtests and compare
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
from fastapi import APIRouter, HTTPException

import core.strategies  # noqa: F401 — triggers @StrategyRegistry.register decorators

from api.schemas import BacktestRequest, BacktestResponse, CompareRequest
from core.backtester.engine import BacktestConfig, BacktestEngine
from core.data import SimulatedDataProvider, YahooDataProvider
from core.data.base import IDataProvider
from core.models.results import BacktestResult
from core.strategies.registry import StrategyRegistry

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_provider(
    data_source: str,
    symbols: list[str],
    start_date: date,  # noqa: ARG001
    end_date: date,  # noqa: ARG001
) -> IDataProvider:
    if data_source == "yahoo":
        return YahooDataProvider()
    # simulated: sensible defaults so any symbol set works without internet
    n = len(symbols)
    spots = {s: 100.0 for s in symbols}
    vols = {s: 0.20 for s in symbols}
    corr = np.eye(n)
    return SimulatedDataProvider(
        spots=spots,
        volatilities=vols,
        correlation=corr,
        drift=0.07,
        risk_free_rate=0.05,
        seed=42,
    )


def _clean_float(v: object) -> object:
    """Replace NaN/Inf floats with None so they survive JSON serialisation."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _clean_metrics(d: dict) -> dict:
    return {k: _clean_float(v) for k, v in d.items()}


def _result_to_response(result: BacktestResult) -> BacktestResponse:
    pv = {str(k.date()): float(v) for k, v in result.portfolio_values.items()}

    bv = None
    if result.benchmark_values is not None and len(result.benchmark_values) > 0:
        bv = {str(k.date()): float(v) for k, v in result.benchmark_values.items()}

    if not result.weights_history.empty:
        wh = {
            str(k.date()): {sym: float(w) for sym, w in row.items()}
            for k, row in result.weights_history.iterrows()
        }
    else:
        wh = {}

    return BacktestResponse(
        portfolio_values=pv,
        benchmark_values=bv,
        weights_history=wh,
        risk_metrics=_clean_metrics(result.risk_metrics),
        trades_log=result.trades_log,
        computation_time_ms=result.computation_time_ms,
        strategy_name=result.strategy_name,
    )


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post("/backtest")
def run_backtest(req: BacktestRequest) -> BacktestResponse:
    try:
        strategy = StrategyRegistry.create(req.strategy_id, req.params)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    provider = _make_provider(req.data_source, req.symbols, req.start_date, req.end_date)

    config = BacktestConfig(
        start_date=req.start_date,
        end_date=req.end_date,
        symbols=req.symbols,
        initial_value=req.initial_value,
        rebalancing_frequency=req.params.get("rebalancing_frequency", "weekly"),
        transaction_cost_bps=float(req.params.get("transaction_cost_bps", 10.0)),
    )

    result = BacktestEngine(config).run(strategy, provider)
    return _result_to_response(result)


@router.post("/backtest/compare")
def compare_backtests(req: CompareRequest) -> list[BacktestResponse]:
    responses = []
    for spec in req.strategies:
        sub_req = BacktestRequest(
            strategy_id=spec.strategy_id,
            symbols=req.symbols,
            start_date=req.start_date,
            end_date=req.end_date,
            initial_value=req.initial_value,
            params=spec.params,
            data_source=req.data_source,
        )
        responses.append(run_backtest(sub_req))
    return responses
