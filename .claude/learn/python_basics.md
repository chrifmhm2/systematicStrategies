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
