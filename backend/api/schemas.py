"""
Phase 5 — Pydantic request/response schemas.
[P5-04a] through [P5-04i]
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator, model_validator


# ── [P5-04i] StrategyInfo ─────────────────────────────────────────────────────


class StrategyInfo(BaseModel):
    id: str
    name: str
    family: str
    description: str
    params: dict


# ── [P5-04a] BacktestRequest ──────────────────────────────────────────────────


class BacktestRequest(BaseModel):
    strategy_id: str
    symbols: list[str]
    start_date: date
    end_date: date
    initial_value: float = 100_000.0
    params: dict = Field(default_factory=dict)
    data_source: str = "simulated"

    @field_validator("initial_value")
    @classmethod
    def positive_value(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("initial_value must be positive")
        return v

    @field_validator("symbols")
    @classmethod
    def non_empty_symbols(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("symbols must be non-empty")
        for s in v:
            if not s.strip():
                raise ValueError("symbol names must be non-empty strings")
        return v

    @model_validator(mode="after")
    def dates_ordered(self) -> BacktestRequest:
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        return self


# ── [P5-04b] BacktestResponse ─────────────────────────────────────────────────


class BacktestResponse(BaseModel):
    portfolio_values: dict[str, float]
    benchmark_values: dict[str, float] | None
    weights_history: dict[str, dict[str, float]]
    risk_metrics: dict
    trades_log: list[dict]
    computation_time_ms: float
    strategy_name: str


# ── [P5-04c] CompareRequest ───────────────────────────────────────────────────


class StrategySpec(BaseModel):
    strategy_id: str
    params: dict = Field(default_factory=dict)


class CompareRequest(BaseModel):
    strategies: list[StrategySpec]
    symbols: list[str]
    start_date: date
    end_date: date
    initial_value: float = 100_000.0
    data_source: str = "simulated"

    @model_validator(mode="after")
    def dates_ordered(self) -> CompareRequest:
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        return self


# ── [P5-04d] HedgingRequest ───────────────────────────────────────────────────


class HedgingRequest(BaseModel):
    option_type: str = "call"
    weights: list[float]
    symbols: list[str]
    strike: float
    maturity_years: float
    risk_free_rate: float = 0.05
    volatilities: list[float]
    correlation_matrix: list[list[float]]
    initial_spots: list[float]
    n_simulations: int = 10_000
    rebalancing_frequency: str = "weekly"
    data_source: str = "simulated"
    n_paths: int = 5


# ── [P5-04e] HedgingResponse ──────────────────────────────────────────────────


class HedgingResponse(BaseModel):
    paths: list[dict]
    average_tracking_error: float
    initial_option_price: float
    initial_option_price_ci: list[float]


# ── [P5-04f] OptionPricingRequest ─────────────────────────────────────────────


class OptionPricingRequest(BaseModel):
    option_type: str = "call"
    S: float
    K: float
    T: float
    r: float
    sigma: float
    method: str = "bs"
    n_simulations: int = 100_000
    # Optional fields for MC basket pricing
    spots: list[float] | None = None
    weights: list[float] | None = None
    volatilities: list[float] | None = None
    correlation: list[list[float]] | None = None


# ── [P5-04g] OptionPricingResponse ────────────────────────────────────────────


class OptionPricingResponse(BaseModel):
    price: float
    std_error: float | None = None
    confidence_interval: list[float] | None = None
    deltas: list[float] | None = None
    greeks: dict


# ── [P5-04h] RiskAnalyzeRequest ───────────────────────────────────────────────


class RiskAnalyzeRequest(BaseModel):
    portfolio_values: dict[str, float]
    benchmark_values: dict[str, float] | None = None
    risk_free_rate: float = 0.05
