"""
[P5-08] Risk analytics route.
POST /risk/analyze → compute performance + risk metrics for a portfolio
"""
from __future__ import annotations

import math

import pandas as pd
from fastapi import APIRouter

from api.schemas import RiskAnalyzeRequest
from core.risk.metrics import PerformanceMetrics

router = APIRouter()


def _clean_metrics(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
            for k, v in d.items()}


@router.post("/risk/analyze")
def analyze_risk(req: RiskAnalyzeRequest) -> dict:
    values = pd.Series(req.portfolio_values)

    benchmark = None
    if req.benchmark_values:
        benchmark = pd.Series(req.benchmark_values)

    metrics = PerformanceMetrics.compute_all(
        values=values,
        benchmark=benchmark,
        risk_free_rate=req.risk_free_rate,
    )
    return _clean_metrics(metrics)
