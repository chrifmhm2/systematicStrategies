"""
Phase 5 tests — FastAPI Backend
[P5-13a] through [P5-13i]
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


# ── [P5-13a] GET /api/strategies ─────────────────────────────────────────────


def test_list_strategies_returns_at_least_six() -> None:
    """Should return HTTP 200 with a list of at least 6 registered strategies."""
    resp = client.get("/api/strategies")
    assert resp.status_code == 200
    data = resp.json()
    assert "strategies" in data
    assert len(data["strategies"]) >= 6


# ── [P5-13b] GET /api/strategies/{id} ────────────────────────────────────────


def test_get_equal_weight_strategy() -> None:
    """EqualWeightStrategy must exist with family='allocation'."""
    resp = client.get("/api/strategies/EqualWeightStrategy")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "EqualWeightStrategy"
    assert data["family"] == "allocation"


# ── [P5-13c] GET /api/strategies/nonexistent ─────────────────────────────────


def test_get_nonexistent_strategy_returns_404() -> None:
    resp = client.get("/api/strategies/nonexistent")
    assert resp.status_code == 404


# ── [P5-13d] POST /api/backtest — happy path ─────────────────────────────────


def test_backtest_equal_weight_simulated() -> None:
    """EqualWeightStrategy on 2 simulated assets returns a complete BacktestResponse."""
    payload = {
        "strategy_id": "EqualWeightStrategy",
        "symbols": ["ASSET1", "ASSET2"],
        "start_date": "2023-01-01",
        "end_date": "2023-06-30",
        "initial_value": 100_000.0,
        "params": {},
        "data_source": "simulated",
    }
    resp = client.post("/api/backtest", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "portfolio_values" in data
    assert len(data["portfolio_values"]) > 0
    assert "risk_metrics" in data
    assert len(data["risk_metrics"]) > 0
    assert "trades_log" in data


# ── [P5-13e] POST /api/backtest — unknown strategy ───────────────────────────


def test_backtest_unknown_strategy_returns_400() -> None:
    payload = {
        "strategy_id": "NonExistentStrategy",
        "symbols": ["AAPL"],
        "start_date": "2023-01-01",
        "end_date": "2023-06-30",
        "initial_value": 100_000.0,
        "params": {},
        "data_source": "simulated",
    }
    resp = client.post("/api/backtest", json=payload)
    assert resp.status_code == 400


# ── [P5-13f] POST /api/backtest/compare ──────────────────────────────────────


def test_compare_two_strategies_returns_two_results() -> None:
    payload = {
        "strategies": [
            {"strategy_id": "EqualWeightStrategy", "params": {}},
            {"strategy_id": "MomentumStrategy", "params": {"lookback_period": 20}},
        ],
        "symbols": ["ASSET1", "ASSET2"],
        "start_date": "2023-01-01",
        "end_date": "2023-06-30",
        "initial_value": 100_000.0,
        "data_source": "simulated",
    }
    resp = client.post("/api/backtest/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2


# ── [P5-13g] POST /api/pricing/option — BS ATM ───────────────────────────────


def test_bs_atm_call_price() -> None:
    """ATM call: S=100, K=100, T=1, r=0.05, sigma=0.20 → price ≈ 10.45."""
    payload = {
        "option_type": "call",
        "S": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.20,
        "method": "bs",
    }
    resp = client.post("/api/pricing/option", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "price" in data
    assert abs(data["price"] - 10.45) < 0.5
    assert "greeks" in data
    assert "delta" in data["greeks"]


# ── [P5-13h] POST /api/risk/analyze ──────────────────────────────────────────


def test_risk_analyze_returns_required_keys() -> None:
    """Risk endpoint must return sharpe_ratio, max_drawdown, and var_95."""
    # monotonically rising portfolio: 100 → 399 over 300 days
    portfolio_values = {f"2023-{(i // 31) + 1:02d}-{(i % 28) + 1:02d}": float(100 + i)
                        for i in range(300)}
    payload = {
        "portfolio_values": portfolio_values,
        "risk_free_rate": 0.05,
    }
    resp = client.post("/api/risk/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "sharpe_ratio" in data
    assert "max_drawdown" in data
    assert "var_95" in data


# ── [P5-13i] GET /api/data/assets ────────────────────────────────────────────


def test_list_assets_returns_non_empty() -> None:
    resp = client.get("/api/data/assets")
    assert resp.status_code == 200
    data = resp.json()
    assert "assets" in data
    assert len(data["assets"]) > 0
