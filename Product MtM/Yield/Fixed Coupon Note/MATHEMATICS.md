# The Maths, Fully Decomposed

Every formula the four scripts in this folder (`Fixed Coupon Note.py`, `Greek Sensitivity.py`,
`Fixed Coupon Note (Autocallable).py`, `Greek Sensitivity (Autocallable).py`) actually compute.
This is the mechanical companion to `README.md` (which covers the "why" and product framing) -
here the goal is that nothing is left as an unexplained black box in the code itself.

Notation: `S0` = spot at inception, `S_T` = spot at maturity, `K = Strike · S0`, `r` = SOFR proxy,
`c` = issuer CDS spread, `q` = dividend yield, `T` = time to maturity in years, `σ` = volatility.

---

## 1. Principal (ZCB) leg

Continuously-compounded, discounted at the issuer's own funding rate (risk-free + credit spread),
since this leg carries the issuer's default risk - the same compounding convention as the option
leg below, so both legs of the replicating portfolio price off one consistent curve:

```
ZCB(T) = Principal · e^(-(r + c)·T)
```

```
∂ZCB/∂r = -T · ZCB(T)                        (rho)
∂ZCB/∂T = (r + c) · ZCB(T)                   (theta: value gained per year as T shrinks)
```

## 2. Put leg (plain FCN and autocallable-when-not-called)

Standard Black-Scholes-Merton, priced via QuantLib's `AnalyticEuropeanEngine` at the risk-free
rate alone (no credit spread - the option is hedged on derivative-pricing terms, not the issuer's
funding curve):

```
d1 = [ln(S0/K) + (r - q + σ²/2)·T] / (σ·√T)
d2 = d1 - σ·√T
Put = K·e^(-rT)·N(-d2) - S0·e^(-qT)·N(-d1)
```

Delta, Vega, Rho, Theta are QuantLib's own closed-form Greeks for this payoff - not re-derived
here.

## 3. Why the short put's quantity is `1/Strike`, not `1`

Below the strike, the note delivers `Principal / K` shares (physical conversion), worth
`Principal · S_T / K` at maturity - a payoff that scales linearly **through the origin**. A
quantity-1 short put's payoff is `Principal - max(K - S_T, 0) = Principal - (K - S_T)` for
`S_T < K`, which only matches `Principal · S_T / K` at the two endpoints `S_T = 0` and `S_T = K` -
everywhere strictly between them the two diverge (the quantity-1 shortfall is a *parallel shift*
of `Principal`, not a line through the origin). Scaling the put quantity to `1/Strike` (i.e.
`1/K` in absolute-price terms) makes the payoff `Principal - (1/Strike)·max(K - S_T, 0) =
Principal·S_T/K` for `S_T < K` exactly - reproducing the delivered-share value at every point, not
just the endpoints.

## 4. Full note price (plain FCN)

```
Price = ZCB(T) - (1/Strike) · Put(S0, K, T, r, σ, q)

Delta = -(1/Strike) · Put_delta
Vega  = -(1/Strike) · Put_vega
Rho   = ZCB_rho - (1/Strike) · Put_rho
Theta = ZCB_theta - (1/Strike) · Put_theta
```

All closed-form; verified against finite differences before shipping.

---

## 5. Autocallable extension

### 5.1 Why there's no closed form

The payoff now depends on whether spot is above `TRIGGER · S0` on *any* of several discrete
future observation dates - a discretely-monitored first-passage problem, not a single terminal
condition. There's no Black-Scholes-style closed form for this (and the continuous-monitoring
reflection-principle trick used for a single barrier doesn't apply to *multiple* discrete dates
either), so the live pricer (`simulate_forward_price`) uses Monte Carlo instead. The historical
backtest path itself is still 100% real data - only the *forward-looking remaining life*, at each
valuation date, is simulated.

### 5.2 Monte Carlo engine

Simulate `n_paths` risk-neutral GBM paths from the valuation date to maturity, over the grid of
remaining observation dates `t_1 < t_2 < ... < t_n = T`:

```
log S_{t_i} = log S_{t_{i-1}} + (r - q - σ²/2)·Δt_i + σ·√Δt_i · Z_i,   Z_i ~ N(0,1) i.i.d.
```

At each path, find the first `t_i` (if any) where `S_{t_i} ≥ TRIGGER · S0`; call that
`settle_time` (or `T` if never triggered). Then, **per path**:

```
principal_pv = Principal · e^(-(r + c)·settle_time)                       [always paid]
put_pv       = -(1/Strike)·max(K - S_T, 0) · e^(-r·T)   if never triggered, else 0
price        = mean(principal_pv + put_pv)  over all paths
```

The principal leg is discounted continuously at the issuer's funding rate, paid at whichever date
the note actually settles; the put leg only exists (and is discounted continuously at `r` alone)
on paths that reach maturity without triggering - exactly mirroring the plain FCN's two-leg split,
generalized to a random settlement date.

### 5.3 Closed-form verification: the autocall feature as a strip of digital calls

A single date's "called by then" probability, in isolation, is an ordinary cash-or-nothing digital
call: `c = Q · e^(-rT) · N(d2)`. But observation dates are **not independent** - they're driven by
the same Brownian path - so summing independent digital-call probabilities across dates
double-counts paths that could have triggered on more than one date. The correct, joint
first-passage version:

Let `X_i = W_{t_i} / √t_i` (the standardized Brownian value at date `i`). Marginally `X_i ~
N(0,1)`, and since `Cov(W_{t_i}, W_{t_j}) = min(t_i, t_j)`:

```
corr(X_i, X_j) = √(min(t_i,t_j) / max(t_i,t_j))
```

"Called at `t_i`" means `S_{t_i} ≥ K`, i.e. `X_i > -d2_i` (using each date's own Black-Scholes
`d2`). Let `k_i = -d2_i`. Then:

```
P(called for the FIRST time exactly at t_i)
    = P(X_1≤k_1, ..., X_(i-1)≤k_(i-1))     [not called on any earlier date]
    - P(X_1≤k_1, ..., X_i≤k_i)             [not called by t_i either]
```

both terms evaluated via the joint CDF of a multivariate normal with the correlation matrix above
(`scipy.stats.multivariate_normal`). Each date's exact first-passage probability is then weighted
against that date's fixed payoff `Q = TRIGGER · S0` (redemption at that level), discounted, and
summed - the strip total is `autocall_digital_strip_price`'s closed-form check on the Monte
Carlo's own `P(called before maturity)`.

### 5.4 Numerical verification (printed at runtime)

`verify_against_closed_form`: with the autocall trigger pushed unreachable, `simulate_forward_
price`'s payoff must reduce to the plain FCN's exact closed-form price (`ZCB - (1/Strike)·Put`).
The relative difference between the Monte Carlo estimate and that closed form is printed as a
PASS/FAIL check - this confirms the MC engine's wiring before trusting it for the actual
(reachable-trigger) autocall price. A second printed check compares the closed-form digital-strip
`P(called before maturity)` against the Monte Carlo's own empirical frequency, as an independent
cross-check on the trigger logic.
