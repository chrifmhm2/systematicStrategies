from core.strategies.base import IStrategy, PortfolioWeights, StrategyConfig
from core.strategies.registry import StrategyRegistry

# Import subpackages so that @StrategyRegistry.register decorators fire at import time
import core.strategies.hedging      # noqa: F401
import core.strategies.allocation   # noqa: F401
import core.strategies.signal       # noqa: F401

__all__ = [
    "IStrategy",
    "PortfolioWeights",
    "StrategyConfig",
    "StrategyRegistry",
]
