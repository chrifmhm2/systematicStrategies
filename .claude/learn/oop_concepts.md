# Concepts — Learning Notes

---

## What is ABC and `@abstractmethod`?

**ABC (Abstract Base Class)** is a class that cannot be instantiated directly — it defines a contract that all subclasses must follow.

```python
from abc import ABC, abstractmethod

class IDataProvider(ABC):
    @abstractmethod
    def get_prices(self, symbols, start_date, end_date):
        ...   # no body needed — declares the contract

    @abstractmethod
    def get_risk_free_rate(self, date):
        ...
```

**`@abstractmethod`** marks methods that every subclass **must** implement. Forgetting one raises `TypeError` at instantiation.

```python
IDataProvider()        # ❌ TypeError — abstract class

class BrokenProvider(IDataProvider):
    pass
BrokenProvider()       # ❌ TypeError — missing get_prices, get_risk_free_rate

class GoodProvider(IDataProvider):
    def get_prices(self, ...): return ...
    def get_risk_free_rate(self, date): return 0.05
GoodProvider()         # ✅ works
```

**Why use it?**
- Error caught at instantiation (early) not at method call (late)
- Guarantees all providers share the same interface
- Enables safe swapping: SimulatedDataProvider ↔ YahooDataProvider, zero strategy code changes

---

## How does the Cholesky matrix make prices move together?

**Problem:** numpy draws only independent randoms. We need correlated ones (e.g. AAPL & MSFT move together).

**Solution:**
```
ε₁, ε₂ ~ N(0,1) independent  →  Z = ε @ L.T  →  Z₁, Z₂ correlated
```

`L` is the Cholesky factor: `L @ L.T = correlation matrix`.
Because `Cov(Z) = L @ I @ L.T = correlation` — mathematically exact.

**Why in pricing?** Basket option payoff depends on multiple assets at expiry:
```
S_i^T = S_i * exp((r - σ²/2)*T  +  σ*√T * Z_i)
```
Using independent `Z_i` would assume zero correlation → wrong basket price.
Cholesky-correlated `Z_i` makes assets move realistically together → correct price.

**One-line intuition:** Cholesky turns independent coin flips into coin flips that tend to land the same way, by exactly the correlation you specify.
