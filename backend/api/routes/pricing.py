"""
[P5-10] Option pricing route.
POST /pricing/option → price a European option via Black-Scholes or Monte Carlo
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas import OptionPricingRequest, OptionPricingResponse
from core.pricing.black_scholes import BlackScholesModel
from core.pricing.monte_carlo import MonteCarloPricer

router = APIRouter()


@router.post("/pricing/option")
def price_option(req: OptionPricingRequest) -> OptionPricingResponse:
    if req.method == "bs":
        try:
            if req.option_type == "call":
                price = BlackScholesModel.call_price(req.S, req.K, req.T, req.r, req.sigma)
            else:
                price = BlackScholesModel.put_price(req.S, req.K, req.T, req.r, req.sigma)

            greeks = {
                "delta": BlackScholesModel.delta(
                    req.S, req.K, req.T, req.r, req.sigma, req.option_type
                ),
                "gamma": BlackScholesModel.gamma(req.S, req.K, req.T, req.r, req.sigma),
                "vega": BlackScholesModel.vega(req.S, req.K, req.T, req.r, req.sigma),
                "theta": BlackScholesModel.theta(
                    req.S, req.K, req.T, req.r, req.sigma, req.option_type
                ),
                "rho": BlackScholesModel.rho(
                    req.S, req.K, req.T, req.r, req.sigma, req.option_type
                ),
            }
        except (ValueError, ZeroDivisionError) as e:
            raise HTTPException(status_code=400, detail=str(e))

        return OptionPricingResponse(
            price=price,
            std_error=None,
            confidence_interval=None,
            deltas=None,
            greeks=greeks,
        )

    # ── Monte Carlo path ─────────────────────────────────────────────────────
    spots = np.array(req.spots) if req.spots else np.array([req.S])
    weights = np.array(req.weights) if req.weights else np.ones(len(spots)) / len(spots)
    vols = np.array(req.volatilities) if req.volatilities else np.array([req.sigma] * len(spots))
    n = len(spots)
    corr = np.array(req.correlation) if req.correlation else np.eye(n)

    try:
        pricer = MonteCarloPricer(n_simulations=req.n_simulations, seed=42)
        result = pricer.price_basket_option(
            spots=spots,
            weights=weights,
            strike=req.K,
            maturity=req.T,
            risk_free_rate=req.r,
            volatilities=vols,
            correlation=corr,
        )
    except (ValueError, np.linalg.LinAlgError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    deltas_list = result.deltas.tolist() if result.deltas is not None else None
    greeks = {"delta": deltas_list or []}

    return OptionPricingResponse(
        price=result.price,
        std_error=result.std_error,
        confidence_interval=list(result.confidence_interval),
        deltas=deltas_list,
        greeks=greeks,
    )
