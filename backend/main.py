"""
[P5-01] QuantForge API — FastAPI entry point.

Run:
    cd backend
    uvicorn main:app --reload --port 8000
    # API docs at http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import backtest, data, hedging, pricing, risk, strategies

app = FastAPI(title="QuantForge API", version="1.0.0")

# ── [P5-01] CORS ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── [P5-01] Mount routers ─────────────────────────────────────────────────────
app.include_router(strategies.router, prefix="/api")
app.include_router(backtest.router, prefix="/api")
app.include_router(hedging.router, prefix="/api")
app.include_router(risk.router, prefix="/api")
app.include_router(data.router, prefix="/api")
app.include_router(pricing.router, prefix="/api")


# ── [P5-11] Global exception handler ─────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )


@app.get("/")
def root() -> dict:
    return {"message": "QuantForge API", "docs": "/docs"}
