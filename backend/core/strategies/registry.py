from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.strategies.base import IStrategy

from core.strategies.base import StrategyConfig


class StrategyRegistry:
    """
    Global catalog of all available strategies.

    Strategies self-register at import time via the @StrategyRegistry.register decorator.
    No central list to maintain — add a new file with the decorator and it appears everywhere.
    """

    _strategies: dict[str, type[IStrategy]] = {}

    @classmethod
    def register(cls, strategy_class: type[IStrategy]) -> type[IStrategy]:
        """Decorator — adds strategy_class to the registry under its class name.

        Usage:
            @StrategyRegistry.register
            class MyStrategy(IStrategy):
                ...
        """
        cls._strategies[strategy_class.__name__] = strategy_class
        return strategy_class  # return unchanged so the class is still usable normally

    @classmethod
    def list_strategies(cls) -> list[dict]:
        """Return a description of every registered strategy.

        Each entry contains:
            name        — class name used as the registry key
            description — human-readable summary (from StrategyConfig)
            family      — "hedging" | "allocation" | "signal" (from class attribute)
            param_schema — JSON-schema-like dict for frontend form rendering
        """
        result = []
        for name, klass in cls._strategies.items():
            result.append({
                "name": name,
                "description": getattr(klass, "DESCRIPTION", ""),
                "family": getattr(klass, "FAMILY", "unknown"),
                "param_schema": klass.get_param_schema(),
            })
        return result

    @classmethod
    def create(cls, name: str, config: dict) -> IStrategy:
        """Instantiate a strategy by name with the given config dict.

        Parameters
        ----------
        name   : Registered class name, e.g. "MomentumStrategy"
        config : Dict of strategy-specific params. Must include at minimum
                 "name" and "description" (or they default to the class name).

        Returns
        -------
        IStrategy instance, ready to call compute_weights() on.

        Raises
        ------
        KeyError if the name is not registered.
        """
        if name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise KeyError(
                f"Strategy '{name}' not found in registry. "
                f"Available: {available}"
            )

        klass = cls._strategies[name]

        strategy_config = StrategyConfig(
            name=config.get("name", name),
            description=config.get("description", getattr(klass, "DESCRIPTION", "")),
            rebalancing_frequency=config.get("rebalancing_frequency", "weekly"),
            transaction_cost_bps=float(config.get("transaction_cost_bps", 10.0)),
        )

        return klass(strategy_config, **{
            k: v for k, v in config.items()
            if k not in {"name", "description", "rebalancing_frequency", "transaction_cost_bps"}
        })
