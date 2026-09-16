# The Maths, Fully Decomposed

The mechanical companion to `README.md`.

Notation: `S0` = spot at inception, `Cap` = the short call's strike, `q` = dividend yield,
`T` = time to maturity, `σ` = volatility, `r` = risk-free rate.

---

## 1. LEPO (Low Exercise Price Option)

A call struck at ~0 is always exercised, so its value is the risk-neutral PV of receiving one
share at maturity - the "prepaid forward" price:

```
LEPO(S, T) = S · e^(-qT)
Delta = e^(-qT)
Theta = q · S · e^(-qT)
Vega = Rho = 0
```

Equals spot exactly only when `q = 0` (no dividends) - a real dividend payer's LEPO is worth less
than spot today, since the holder only receives the share at maturity and forgoes every dividend
paid before then. **Independent of `r` entirely**: the PV of a future share delivery carries no
rate exposure, unlike a strike-bearing option (a well-known identity, not an approximation).
Priced directly via this closed form rather than through the general Black-Scholes call engine,
since `K = 0` is numerically awkward for one (`d1`/`d2` blow up as `K → 0`).

## 2. Full certificate price

```
Price = LEPO(S, T) - Call_BlackScholes(S, Cap, T, r, σ, q)
Payoff at maturity = S_T - max(S_T - Cap, 0) = min(S_T, Cap)
```

Greeks are just the leg-by-leg difference (both closed-form, no finite differences needed).

## 3. Solving for `Cap` given a target discount

`DISCOUNT` (the hand-set dial) is not a strike - it's the resulting fair-value-vs-par gap that
`Cap` (plus vol/rate/tenor) produces. Since Price is **monotonically increasing in `Cap`** (a
higher, more out-of-the-money cap sells a cheaper call, funding a smaller discount - i.e. a fair
value closer to par), there is exactly one `Cap` in `(S0, 10·S0]` solving:

```
LEPO(S0, T) - Call_BlackScholes(S0, Cap, T, r, σ, q) = S0 · (1 - DISCOUNT)
```

There's no closed-form inverse for a Black-Scholes call strike given a target price, so this is a
genuine root-find (`scipy.optimize.brentq`, Brent's method), not an algebraic rearrangement.
