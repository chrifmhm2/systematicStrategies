# Math & Finance — Learning Notes

---

## Cholesky Decomposition — making prices move together

**Problem:** numpy draws only independent random numbers. We need correlated ones (e.g. AAPL & MSFT tend to rise and fall on the same days).

**What Cholesky does:**
Given a correlation matrix `C`, Cholesky finds a lower-triangular matrix `L` such that:
```
L @ L.T = C
```

**How we use it to generate correlated randoms:**
```
Step 1 — draw independent normals:   ε ~ N(0, I)       shape (n_samples, n_assets)
Step 2 — correlate via Cholesky:     Z = ε @ L.T       shape (n_samples, n_assets)

Result: Cov(Z) = L @ Cov(ε) @ L.T = L @ I @ L.T = C   ✅ exact correlation
```

**Example:**
```
C = [[1.0, 0.6],    →   L = [[1.0, 0.0],
     [0.6, 1.0]]              [0.6, 0.8]]

L @ L.T = [[1.0, 0.6],   ✓
           [0.6, 1.0]]
```

**Why in option pricing?**
Basket option payoff depends on multiple correlated assets at expiry:
```
S_i^T = S_i * exp((r - σ_i²/2)*T  +  σ_i * √T * Z_i)
                                        ↑
                                  must be correlated
```
Using independent `Z_i` assumes zero correlation between assets → wrong basket price.
Cholesky-correlated `Z_i` → realistic joint behaviour → correct basket price.

**One-line intuition:**
> Cholesky turns independent coin flips into coin flips that tend to land the same way — by exactly the correlation you specify.

---

## GBM — Geometric Brownian Motion

The standard model for stock prices. A stock price `S_t` follows:

```
S_t = S_0 * exp( (μ - σ²/2)*t  +  σ*√t * Z )
              ────────────────    ────────────
              drift term          random shock
```

| Symbol | Name | Meaning |
|--------|------|---------|
| `S_0` | Initial price | Price today |
| `μ` | Drift | Average annual return (trend) |
| `σ` | Volatility | Annual standard deviation of returns |
| `t` | Time | In years (1 day = 1/252) |
| `Z` | Brownian shock | Random draw from N(0,1) |

**Why `σ²/2` is subtracted from the drift?**
Because we model log-returns (not prices) as normal. The `-σ²/2` is an Itô correction — without it, the expected value of `S_t` would be wrong. With it: `E[S_t] = S_0 * exp(μ*t)` as expected.

**Daily simulation (discrete version):**
```
S_t = S_{t-1} * exp( (μ - σ²/2)*dt  +  σ*√dt * Z_t )    where dt = 1/252
```

---

## Log-returns vs Simple returns

| | Simple return | Log return |
|--|--------------|------------|
| Formula | `(S_t - S_{t-1}) / S_{t-1}` | `ln(S_t / S_{t-1})` |
| Can go below -100%? | No | Yes (theoretically) |
| Additive over time? | No | Yes |
| Used in GBM? | No | Yes |

Log-returns are additive: `r_total = r_1 + r_2 + ... + r_n` — makes simulation easy via `cumsum`.

---

## Black-Scholes — closed-form option pricing

**Financial problem:** fairly price an option whose payoff depends on an unknown future price.
BS showed: continuously rebalance a stock+cash portfolio to be delta-neutral → randomness cancels → one fair price exists.

**Call price:**
```
C = S · N(d1)  −  K · e^(−rT) · N(d2)
    ──────────    ─────────────────────
    expected        expected discounted
    stock received  strike payment

d1 = [ ln(S/K) + (r + σ²/2)·T ] / (σ·√T)
d2 = d1 − σ·√T
```

**Put price (put-call parity):**
```
P = C - S + K * e^(-rT)
```

**What d1 and d2 mean:**

| Part | Meaning |
|------|---------|
| `ln(S/K)` | How far in/out of the money right now |
| `r·T` | Stock drifts up at risk-free rate (risk-neutral world) |
| `σ²/2·T` | Itô correction — bias from log vs normal space |
| `σ·√T` | Uncertainty band — how wide the future price distribution is |

**What N(d1) and N(d2) mean financially:**

| Term | Financial meaning |
|------|-----------------|
| `N(d2)` | Probability the call expires in-the-money (S_T > K) |
| `N(d1)` | Expected fraction of stock received, adjusted for drift |
| `K·e^(-rT)·N(d2)` | Expected discounted strike payment |
| `S·N(d1)` | Expected discounted value of stock received |

**ATM benchmark:** `S=100, K=100, T=1, r=0.05, σ=0.2` → call ≈ **10.45**

---

## Greeks — sensitivity of option price

`n(x)` = normal PDF (bell curve height at x)
`N(x)` = normal CDF (area under bell curve up to x)

| Greek | Formula | Financial intuition |
|-------|---------|-------------------|
| **Delta** Δ | `N(d1)` call, `N(d1)-1` put | $1 move in S → Δ move in option. Also: shares needed to hedge. ATM≈0.5, deep ITM≈1, deep OTM≈0 |
| **Gamma** Γ | `n(d1)/(S·σ·√T)` | How fast delta changes. High near expiry ATM. Tells you how often to re-hedge. Always positive |
| **Vega** ν | `S·n(d1)·√T` | Options are bets on vol — vega measures that bet. Always positive: more vol → more valuable |
| **Theta** Θ | (formula in code) | Daily time decay. Almost always negative. Sellers earn theta; buyers lose it every day |
| **Rho** ρ | `K·T·e^(-rT)·N(d2)` call | Sensitivity to rates. Positive for calls (higher r → stock drifts up → call worth more) |

---

## Monte Carlo Pricing

When there is no closed-form formula (e.g. basket options), we simulate many possible futures and average the payoffs.

```
1. Simulate N paths of terminal prices S_i^T (using GBM + Cholesky)
2. Compute payoff for each path:  payoff = max(Σ ω_i * S_i^T - K, 0)
3. Average payoffs and discount:  price = mean(payoffs) * e^(-r*T)
```

**Antithetic variates (variance reduction):**
For each random draw `Z`, also compute the payoff with `-Z`, then average the two before taking the global mean. This halves the variance of the estimate for the same number of simulations.

```
payoff_pair = (payoff(+Z) + payoff(-Z)) / 2
price = mean(payoff_pairs) * e^(-r*T)
```

**Standard error** measures how uncertain the MC estimate is:
```
std_error = std(payoffs) / √N
```
Larger N → smaller std_error → more reliable price.

---

## Put-Call Parity

A fundamental no-arbitrage relationship between call and put prices:

```
C - P = S - K * e^(-rT)
```

If this doesn't hold, you could make risk-free profit (arbitrage). In code, we verify this as a sanity check on our Black-Scholes implementation.

---

## Risk-Free Rate

The return on a theoretically riskless investment (e.g. US Treasury bills).

- Used to **discount** future payoffs back to today: `price = payoff * e^(-r*T)`
- Used as the **drift** in risk-neutral pricing (Black-Scholes, Monte Carlo)
- In our project: `IDataProvider.get_risk_free_rate()` — typically hardcoded at `0.05` (5%)

---

## Phase 2 — Strategy Framework Math

---

### Equal Weight (1/N)

No optimization. Split capital equally:
```
wᵢ = 1/N    for each of N assets,   cash = 0
```
Empirically competitive with complex optimizers (DeMiguel et al. 2009) because it has zero estimation error.

---

### Minimum Variance (Markowitz 1952)

**Goal:** Lowest possible portfolio volatility.

Optimization problem:
```
min   ωᵀ Σ ω
 ω

s.t.  Σωᵢ = 1,   ωᵢ ≥ 0
```

- `Σ` = rolling sample covariance of daily returns over `lookback_window` days
- Portfolio variance = `ωᵀ Σ ω`
- Sits at the leftmost point of the efficient frontier (min risk, any return)
- Solver: SLSQP (scipy.optimize.minimize)

---

### Maximum Sharpe Ratio (Sharpe 1966)

**Goal:** Maximize return per unit of risk.

Sharpe Ratio:
```
SR = (μₚ - rf) / σₚ
   = (ωᵀμ - rf) / √(ωᵀΣω)
```

We minimize the negative SR:
```
min   -(ωᵀμ - rf) / √(ωᵀΣω)
 ω

s.t.  Σωᵢ = 1,   ωᵢ ≥ 0
```

Annualisation: `μ_annual = μ_daily × 252`,  `Σ_annual = Σ_daily × 252`

The MaxSharpe portfolio is the **tangency portfolio** — where the Capital Market Line (from rf) is tangent to the efficient frontier.

---

### Risk Parity

**Goal:** Each asset contributes equally to total portfolio variance.

Risk Contribution of asset i:
```
RC_i = ωᵢ × (Σω)ᵢ / (ωᵀΣω)
```

Optimization (minimize dispersion of RCs):
```
min   Σᵢ (RC_i - 1/N)²
 ω

s.t.  Σωᵢ = 1,   ωᵢ > 0   (lower bound 1e-6 for stability)
```

Intuition: If TSLA volatility is 3× MSFT, TSLA gets ~1/3 the weight. Used by Bridgewater's All Weather fund.

---

### Momentum (Jegadeesh & Titman 1993)

Signal = trailing return over `lookback_period` days:
```
returnᵢ = Sᵢ,t / Sᵢ,t-L - 1
```

Long-only: rank assets, equal-weight top-k: `wᵢ = 1/k`
Long-short: top-k get `+1/(2k)`, bottom-k get `-1/(2k)`

Why it works: investor underreaction, herding, slow information diffusion.

---

### Mean Reversion

Signal = z-score of current price vs rolling mean:
```
zᵢ = (Sᵢ,t - MAᵢ) / σᵢ
```

- `|z| > threshold` → asset is mispriced → trade it
- `z < -threshold` → buy (oversold)
- Weights ∝ `1/|z|` (closer to fair value → larger weight → less risk)
- No signal → 100% cash

---

### Delta Hedging (Basket Option)

Delta = sensitivity of option price to spot:
```
Single asset:  Δ = ∂C/∂S = N(d₁)     (Black-Scholes closed form)
Basket:        Δᵢ ≈ [C(Sᵢ+ε) - C(Sᵢ)] / ε    (finite difference via Monte Carlo)
```

Convert deltas to portfolio weights:
```
wᵢ = (Δᵢ × Sᵢ) / V_option
cash_weight = max(0, 1 - Σᵢ wᵢ)
```

The portfolio replicates the option payoff dynamically. From Black-Scholes: if you continuously rebalance to hold Δ shares, randomness cancels and you earn the risk-free rate.

---

### Delta-Gamma Hedge (planned)

Gamma = rate of change of delta:
```
Γ = ∂²C/∂S² = ∂Δ/∂S
```

To neutralize gamma, add a second option position with gamma Γ_h:
```
ω_h = -Γ_portfolio / Γ_h
```
Then re-delta-hedge the combined portfolio. Currently: delegates to DeltaHedge.

---

### Efficient Frontier (summary diagram)

```
Expected Return
↑
|          * MaxSharpe (tangency point)
|        /
|      /   ← Capital Market Line (from rf to tangency)
|    /
rf  /
|  * MinVariance (leftmost point)
|    ← Efficient Frontier (pareto-optimal portfolios)
|_________________________________→ Volatility
```

---

## Implied Volatility

The volatility `σ` that, plugged into Black-Scholes, reproduces a given market price.

```
Market price observed → find σ such that BS(S, K, T, r, σ) = market price
```

Cannot be solved analytically — we use **Newton-Raphson** iteration:
```
σ_new = σ_old - (BS(σ_old) - market_price) / vega(σ_old)
```
Repeat until convergence. Vega is the derivative used because it measures how the BS price changes with σ.
