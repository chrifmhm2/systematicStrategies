from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from core.models.results import PricingResult
from core.pricing.monte_carlo import MonteCarloPricer
from core.strategies.base import IStrategy, PortfolioWeights, StrategyConfig
from core.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class DeltaHedgeStrategy(IStrategy):
    """
    Delta-hedging strategy for a European basket call option.

    At each rebalancing date the strategy:
      1. Computes Monte Carlo deltas (∂V/∂Sᵢ) for each underlying
      2. Converts deltas into portfolio weights: wᵢ = Δᵢ · Sᵢ / V
      3. Allocates remaining capital to cash

    This replicates the option's delta exposure so that small moves in the
    underlying basket are (approximately) offset by the hedge portfolio.
    """

    DESCRIPTION = "Delta-hedges a basket call option using Monte Carlo deltas"
    FAMILY = "hedging"

    def __init__(
        self,
        config: StrategyConfig,
        option_weights: list[float] | None = None,
        strike: float = 100.0,
        maturity_years: float = 1.0,
        n_simulations: int = 50_000,
        risk_free_rate: float = 0.05,
        volatilities: list[float] | None = None,
        correlation: list[list[float]] | None = None,
    ) -> None:
        super().__init__(config)
        self.option_weights = option_weights  # basket weights ωᵢ
        self.strike = strike
        self.maturity_years = maturity_years
        self.n_simulations = n_simulations
        self.risk_free_rate = risk_free_rate
        self.volatilities = volatilities
        self.correlation = correlation

    @property
    def required_history_days(self) -> int:
        return 1

    def compute_weights(
        self,
        current_date: date,
        price_history: pd.DataFrame,
        current_portfolio: object | None = None,
    ) -> PortfolioWeights:
        symbols = list(price_history.columns)
        n = len(symbols)
        spots = price_history.iloc[-1].values.astype(float)

        # Defaults: equal basket weights, 20% vol, identity correlation
        weights_arr = np.array(self.option_weights) if self.option_weights else np.ones(n) / n
        vols = np.array(self.volatilities) if self.volatilities else np.full(n, 0.2)
        corr = np.array(self.correlation) if self.correlation else np.eye(n)

        pricer = MonteCarloPricer(n_simulations=self.n_simulations, seed=42)
        deltas = pricer.compute_deltas(
            spots=spots,
            weights=weights_arr,
            strike=self.strike,
            maturity=self.maturity_years,
            risk_free_rate=self.risk_free_rate,
            volatilities=vols,
            correlation=corr,
        )

        # Option value (approx) = weighted basket × average delta
        option_value = float(np.dot(deltas, spots))
        if option_value <= 0:
            option_value = float(np.sum(spots))  # fallback

        # Weight of asset i = (Δᵢ × Sᵢ) / V
        raw_weights = {sym: float(deltas[i] * spots[i] / option_value) for i, sym in enumerate(symbols)}
        total_asset_weight = sum(raw_weights.values())
        cash_weight = max(0.0, 1.0 - total_asset_weight)

        return PortfolioWeights(
            weights=raw_weights,
            cash_weight=cash_weight,
            timestamp=current_date,
        )

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "option_weights":       {"type": "array",   "items": {"type": "number"}, "description": "Basket weights (must sum to 1)"},
            "strike":               {"type": "number",  "default": 100.0,            "description": "Option strike price K"},
            "maturity_years":       {"type": "number",  "default": 1.0,              "description": "Time to expiry in years"},
            "n_simulations":        {"type": "integer", "default": 50000,            "description": "Monte Carlo paths"},
            "risk_free_rate":       {"type": "number",  "default": 0.05,             "description": "Annualised risk-free rate"},
            "volatilities":         {"type": "array",   "items": {"type": "number"}, "description": "Annual vol per asset"},
            "correlation":          {"type": "array",   "items": {"type": "array"},  "description": "Correlation matrix"},
            "rebalancing_frequency":{"type": "string",  "default": "weekly",         "enum": ["daily", "weekly", "monthly"]},
        }
