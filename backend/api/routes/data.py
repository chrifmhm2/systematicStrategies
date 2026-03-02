"""
[P5-09] Market data routes.
GET /data/assets                                  → default universe list
GET /data/prices?symbols=AAPL,MSFT&start=...&end= → fetch prices via Yahoo
"""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Query

from core.data.yahoo import DEFAULT_UNIVERSE, YahooDataProvider

router = APIRouter()


@router.get("/data/assets")
def list_assets() -> dict:
    return {"assets": [{"symbol": sym} for sym in DEFAULT_UNIVERSE]}


@router.get("/data/prices")
def get_prices(
    symbols: str = Query(..., description="Comma-separated tickers, e.g. AAPL,MSFT"),
    start: date = Query(...),
    end: date = Query(...),
) -> dict:
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    provider = YahooDataProvider()
    df = provider.get_prices(symbol_list, start, end)
    prices = {
        sym: {str(idx.date()): float(val) for idx, val in df[sym].items()}
        for sym in symbol_list
        if sym in df.columns
    }
    return {"prices": prices}
