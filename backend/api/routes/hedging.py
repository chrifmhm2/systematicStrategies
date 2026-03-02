"""
[P5-07] Hedging simulation route.
POST /hedging/simulate → run n_paths delta-hedge backtests
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
from fastapi import APIRouter

import core.strategies  # noqa: F401 — triggers @StrategyRegistry.register decorators

from api.schemas import HedgingRequest, HedgingResponse
from core.backtester.engine import BacktestConfig, BacktestEngine
from core.data.simulated import SimulatedDataProvider
from core.pricing.monte_carlo import MonteCarloPricer
from core.strategies.registry import StrategyRegistry

router = APIRouter()

_START_DATE = date(2024, 1, 2)  # fixed anchor for simulated paths


@router.post("/hedging/simulate")
def simulate_hedging(req: HedgingRequest) -> HedgingResponse:
    spots_arr = np.array(req.initial_spots)
    weights_arr = np.array(req.weights)
    vols_arr = np.array(req.volatilities)
    corr = np.array(req.correlation_matrix)

    spots_dict = dict(zip(req.symbols, req.initial_spots))
    vols_dict = dict(zip(req.symbols, req.volatilities))

    # ── Initial option price ────────────────────────────────────────────
    pricer = MonteCarloPricer(n_simulations=req.n_simulations, seed=42)
    pricing_result = pricer.price_basket_option(
        spots=spots_arr,
        weights=weights_arr,
        strike=req.strike,
        maturity=req.maturity_years,
        risk_free_rate=req.risk_free_rate,
        volatilities=vols_arr,
        correlation=corr,
    )

    start_date = _START_DATE
    end_date = start_date + timedelta(days=int(req.maturity_years * 365))

    # ── Run n_paths independent simulations ─────────────────────────────
    paths: list[dict] = []
    tracking_errors: list[float] = []

    for seed in range(req.n_paths):
        provider = SimulatedDataProvider(
            spots=spots_dict,
            volatilities=vols_dict,
            correlation=corr,
            drift=req.risk_free_rate,
            risk_free_rate=req.risk_free_rate,
            seed=seed,
        )

        strategy_params = {
            "rebalancing_frequency": req.rebalancing_frequency,
            "option_weights": req.weights,
            "strike": req.strike,
            "maturity_years": req.maturity_years,
            "n_simulations": req.n_simulations,
            "risk_free_rate": req.risk_free_rate,
            "volatilities": req.volatilities,
            "correlation": req.correlation_matrix,
        }
        strategy = StrategyRegistry.create("DeltaHedgeStrategy", strategy_params)

        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            symbols=req.symbols,
            initial_value=pricing_result.price,
            rebalancing_frequency=req.rebalancing_frequency,
        )

        result = BacktestEngine(config).run(strategy, provider)
        path_dict = {str(k.date()): float(v) for k, v in result.portfolio_values.items()}
        paths.append(path_dict)

        if len(result.portfolio_values) > 1:
            returns = result.portfolio_values.pct_change().dropna()
            tracking_errors.append(float(returns.std()))

    avg_te = float(np.mean(tracking_errors)) if tracking_errors else 0.0

    return HedgingResponse(
        paths=paths,
        average_tracking_error=avg_te,
        initial_option_price=pricing_result.price,
        initial_option_price_ci=list(pricing_result.confidence_interval),
    )
