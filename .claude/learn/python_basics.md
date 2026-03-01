# Python Basics — Learning Notes

---

## What is a dataclass?

A `@dataclass` automatically generates `__init__`, `__repr__`, and `__eq__` from the fields you declare — no boilerplate.

```python
from dataclasses import dataclass

@dataclass
class DataFeed:
    symbol: str
    date: date
    price: float

df = DataFeed(symbol="AAPL", date=date(2024,1,2), price=185.5)
print(df)  # DataFeed(symbol='AAPL', date=2024-01-02, price=185.5)
```

Without `@dataclass` you'd write `__init__` manually with `self.symbol = symbol` etc.

---

## What is a decorator?

A decorator is a function that **wraps another function or class** to add behaviour, applied with `@`:

```python
@dataclass          # decorator
class DataFeed:
    ...

# equivalent to:
DataFeed = dataclass(DataFeed)
```

---

## What are `__init__`, `__repr__`, `__eq__`?

**Dunder methods** (double-underscore = "magic methods") — Python hooks called automatically:

| Method | Triggered by | What it does |
|--------|-------------|--------------|
| `__init__` | `DataFeed(...)` | Constructor — initialises the object |
| `__repr__` | `print(obj)` / `repr(obj)` | Returns a readable string of the object |
| `__eq__` | `obj1 == obj2` | Defines equality between two objects |

`@dataclass` generates all three automatically from your field declarations.

---

## What is `field()` in a dataclass?

`field()` customises how a dataclass field behaves. Needed for **mutable defaults** (dict, list) which cannot be set directly.

```python
# WRONG — all instances share the same dict object
@dataclass
class Portfolio:
    positions: dict = {}                          # ❌

# CORRECT — each instance gets its own fresh dict
@dataclass
class Portfolio:
    positions: dict = field(default_factory=dict) # ✅
```

**Immutable** types (`int`, `float`, `str`, `bool`, `tuple`) are safe as plain defaults — they cannot be changed in-place, so sharing is not a problem.

**Mutable** types (`dict`, `list`, `set`) can be changed in-place — if shared across instances, modifying one affects all. `field(default_factory=...)` calls a fresh constructor per instance to avoid this.

---

## What is `from __future__ import annotations`?

Makes Python treat all type hints as **strings** (lazy evaluation) instead of evaluating them at class definition time.

Allows modern syntax (`X | Y`, `list[str]`) to work on older Python versions:

```python
from __future__ import annotations

# works even on Python 3.9 — without it, | None would crash
benchmark_values: dict[str, float] | None = None
```

Put it at the top of any file using modern type hint syntax.

---

## What are benchmark values?

A **benchmark** is a simple reference portfolio — the baseline to compare your strategy against.
In QuantForge it is **equal-weight buy-and-hold**: invest equally in all assets on day 1, never touch it again.

```
benchmark  → market goes up 2% → benchmark up 2%   (no skill, just market)
strategy   → market goes up 2% → strategy up 3.5%  (strategy adds 1.5% alpha)
```

If the strategy equity curve stays above the benchmark curve, the strategy adds value.

---

## Phase 2 — Python Patterns & Algorithms

---

### Abstract Base Classes (ABC)

An **abstract class** defines a contract — a set of methods every subclass MUST implement. It cannot be instantiated directly.

```python
from abc import ABC, abstractmethod

class IStrategy(ABC):
    @abstractmethod
    def compute_weights(self, ...) -> PortfolioWeights:
        ...   # no body needed — subclasses must provide one

    @property
    @abstractmethod
    def required_history_days(self) -> int:
        ...
```

If a subclass forgets to implement `compute_weights`, Python raises `TypeError` on instantiation:
```
TypeError: Can't instantiate abstract class MomentumStrategy
           with abstract method compute_weights
```

**Why use it?** Enforces a contract at the language level — every strategy is guaranteed to have `compute_weights` and `required_history_days`. The backtester can call these blindly on any strategy.

---

### Registry / Plugin Pattern

A **class-level dict** stores registered items. The `@register` decorator adds to it at import time:

```python
class StrategyRegistry:
    _strategies: dict[str, type[IStrategy]] = {}   # class-level, shared by all instances

    @classmethod
    def register(cls, strategy_class):
        cls._strategies[strategy_class.__name__] = strategy_class
        return strategy_class   # must return the class unchanged
```

Usage:
```python
@StrategyRegistry.register        # fires when the file is imported
class MomentumStrategy(IStrategy):
    ...
```

**Decorator mechanics**: `@StrategyRegistry.register` is equivalent to:
```python
MomentumStrategy = StrategyRegistry.register(MomentumStrategy)
```
The decorator receives the class, stores it in `_strategies`, and returns it unchanged.

**`@classmethod` vs `@staticmethod` vs regular method:**

| | `self` | `cls` | Description |
|---|---|---|---|
| Regular method | ✅ | ✗ | Works on one instance |
| `@classmethod` | ✗ | ✅ | Works on the class itself (shared state) |
| `@staticmethod` | ✗ | ✗ | Pure function inside the class namespace |

Registry uses `@classmethod` so `_strategies` is shared across all callers.

---

### `@property` decorator

Turns a method into an attribute — called without `()`:

```python
class DeltaHedgeStrategy(IStrategy):
    @property
    def required_history_days(self) -> int:
        return 1

# usage — no parentheses!
strat.required_history_days   # → 1, not strat.required_history_days()
```

Used for `required_history_days` because it's a read-only characteristic, not an action.

---

### Composition Pattern

Instead of inheriting, you **own** another object and delegate to it:

```python
class DeltaGammaHedgeStrategy(IStrategy):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self._delta_hedge = DeltaHedgeStrategy(config, **kwargs)   # owns it

    def compute_weights(self, ...):
        return self._delta_hedge.compute_weights(...)   # delegates
```

**Composition vs Inheritance rule of thumb:**
- Inheritance (`is-a`): `DeltaGammaHedge` IS a `IStrategy` → inherit
- Composition (`has-a`): `DeltaGammaHedge` HAS a `DeltaHedge` → compose

---

### `**kwargs` forwarding

`**kwargs` captures any extra keyword arguments as a dict, and `**` unpacks them:

```python
def __init__(self, config, **kwargs):
    # kwargs = {"lookback_period": 252, "top_k": 3}
    self._inner = SomeStrategy(config, **kwargs)
    # equivalent to: SomeStrategy(config, lookback_period=252, top_k=3)
```

Used in `StrategyRegistry.create()` to forward user-provided params to the strategy without listing them explicitly.

---

### `TYPE_CHECKING` guard

Avoids circular imports at runtime while keeping type hints for IDEs:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.strategies.base import IStrategy   # only imported by mypy/pylance, not at runtime
```

At runtime, `TYPE_CHECKING` is `False` → the import is skipped. The type checker sees it as `True` → gets full type info.

---

### `getattr(obj, name, default)`

Gets an attribute by name (as a string), with a fallback if it doesn't exist:

```python
getattr(klass, "DESCRIPTION", "")    # → klass.DESCRIPTION  or  "" if missing
getattr(klass, "FAMILY", "unknown")  # → klass.FAMILY  or  "unknown"
```

Used in `StrategyRegistry.list_strategies()` to read class-level attributes from any registered class without crashing if they're absent.

---

### `scipy.optimize.minimize` with SLSQP

General constrained optimizer. We use it for MinVariance, MaxSharpe, RiskParity:

```python
from scipy.optimize import minimize

result = minimize(
    objective,            # function to minimize: f(ω) → scalar
    x0,                   # starting weights (initial guess)
    method="SLSQP",       # Sequential Least Squares Programming
    bounds=bounds,        # per-weight bounds: [(0, 1), (0, 1), ...]
    constraints=constraints,  # list of dicts: {"type": "eq", "fun": ...}
)
weights = result.x        # optimal weights
```

**Constraints dict format:**
```python
{"type": "eq",  "fun": lambda w: w.sum() - 1}   # equality:   Σwᵢ = 1
{"type": "ineq","fun": lambda w: w.sum() - 0.5} # inequality: Σwᵢ ≥ 0.5
```

**SLSQP** (Sequential Least Squares Programming): iterative method that approximates the problem as a series of quadratic sub-problems. Handles both equality and inequality constraints. Converges in ~50–200 iterations for typical portfolio problems.

---

### pandas operations used in strategies

```python
# Rolling window of last N rows
prices.tail(lookback)                # last N rows by index

# Daily returns (percentage change day-over-day)
returns = prices.pct_change().dropna()
# pct_change(): (Sₜ - Sₜ₋₁) / Sₜ₋₁
# dropna(): removes first row which is NaN (no previous price)

# Sample covariance matrix
cov = returns.cov()                  # shape (n_assets, n_assets)

# Column-wise mean and std
ma  = window.mean()                  # rolling mean per asset
std = window.std()                   # rolling std per asset

# Current price vector (last row)
current = prices.iloc[-1]            # iloc = integer location

# Select rows up to date t (inclusive)
history = prices.loc[:t]             # loc = label location (date index)
```

---

### numpy operations used in strategies

```python
import numpy as np

# Convert pandas Series to numpy array
w = np.array([pw.weights[s] for s in symbols])

# Portfolio variance: ωᵀ Σ ω  (quadratic form)
port_var = float(w @ cov @ w)        # @ is matrix multiply

# Marginal risk contribution: Σω
mrc = cov @ w                        # matrix × vector

# Element-wise multiply: risk contribution
rc = w * mrc / port_var

# Argsort — indices that would sort the array
ranks = np.argsort(returns)          # ascending order
top_k_idx = np.argsort(returns)[-k:] # last k = top k

# Clip — bound values between min and max
std = std.replace(0, np.nan)         # avoid div by zero in pandas
```

---

### `__init__.py` as a package + trigger

An `__init__.py` serves two purposes:
1. Marks the folder as a Python package (importable)
2. Runs code at import time — used to trigger `@register` decorators

```python
# core/strategies/__init__.py
from core.strategies.base import IStrategy, PortfolioWeights, StrategyConfig
from core.strategies.registry import StrategyRegistry

import core.strategies.hedging      # noqa: F401  ← triggers all @register in hedging/
import core.strategies.allocation   # noqa: F401  ← triggers all @register in allocation/
import core.strategies.signal       # noqa: F401  ← triggers all @register in signal/
```

`# noqa: F401` tells the linter "I know this import looks unused — it's intentional."

---

### `__all__` — public API declaration

A list of names exported when someone does `from module import *`:

```python
__all__ = ["IStrategy", "PortfolioWeights", "StrategyConfig", "StrategyRegistry"]
```

Without `__all__`, `import *` would export everything including internal names. With `__all__`, you explicitly control the public surface of your module.

---

### List comprehension vs dict comprehension

```python
# List comprehension
nonzero = [s for s, w in pw.weights.items() if w > 1e-9]

# Dict comprehension
weights = {sym: 1.0 / top_k for sym in top_symbols}

# Dict comprehension with condition
filtered = {k: v for k, v in config.items() if k not in {"name", "description"}}
```

These are idiomatic Python — one-liners that replace multi-line for loops.

---

### Algorithmic patterns in strategies

**Ranking / Top-K selection:**
```python
# Sort by value, take last k (highest)
sorted_symbols = sorted(returns.items(), key=lambda x: x[1])
top_k = [sym for sym, _ in sorted_symbols[-k:]]
# Time complexity: O(N log N) — dominated by sort
```

**Z-score normalization:**
```python
z = (current_price - rolling_mean) / rolling_std
# Tells you: "how many std deviations away from the mean"
# z = 0  → exactly at mean
# z = +2 → 2 std above → potentially overbought
# z = -2 → 2 std below → potentially oversold
```

**Weight normalization (sum to 1):**
```python
raw = {sym: 1.0 / abs(z[sym]) for sym in triggered}
total = sum(raw.values())
weights = {sym: v / total for sym, v in raw.items()}
# Pattern: compute any positive scores, then divide by sum
```

**Finite difference (numerical derivative):**
```python
eps = spot * 0.01   # 1% bump
delta_i ≈ (price(spot + eps) - price(spot)) / eps
# Approximates ∂C/∂S without knowing the analytic formula
# More precise with smaller eps, but too small → numerical noise
```
