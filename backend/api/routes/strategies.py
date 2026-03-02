"""
[P5-05] Strategy listing routes.
GET /strategies          → all registered strategies
GET /strategies/{id}     → single strategy or 404
"""
from __future__ import annotations

import core.strategies  # noqa: F401 — triggers @StrategyRegistry.register decorators

from fastapi import APIRouter, HTTPException

from api.schemas import StrategyInfo
from core.strategies.registry import StrategyRegistry

router = APIRouter()


def _to_info(entry: dict) -> StrategyInfo:
    return StrategyInfo(
        id=entry["name"],
        name=entry["name"],
        family=entry["family"],
        description=entry["description"],
        params=entry["param_schema"],
    )


@router.get("/strategies")
def list_strategies() -> dict:
    entries = StrategyRegistry.list_strategies()
    return {"strategies": [_to_info(e) for e in entries]}


@router.get("/strategies/{strategy_id}")
def get_strategy(strategy_id: str) -> StrategyInfo:
    entries = StrategyRegistry.list_strategies()
    entry = next((e for e in entries if e["name"] == strategy_id), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found")
    return _to_info(entry)
