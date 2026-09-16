# The Maths, Fully Decomposed

The mechanical companion to `README.md`. Builds on `../MATHEMATICS.md` (single-name Reverse
Convertible - actually `../../Fixed Coupon Note/MATHEMATICS.md`, since the RC's math is identical
to the FCN's) for the conversion-ratio derivation; this file covers what's new for a worst-of
basket.

Notation: `n` names, each tracked as **relative performance** `x_i(t) = S_i(t)/S0_i` (1.0 at
inception, so `Principal = 1.0` "par units" throughout - there's no single dollar `S0` to anchor
multiple names to). `ρ` = the `n×n` correlation matrix from trailing real returns.

---

## 1. Why relative performance, not raw price

Comparing a $50 stock's price to a $500 stock's directly is meaningless. Each name is instead
tracked relative to its own entry level. A continuous-dividend GBM's relative-performance process
obeys the identical SDE (same `σ`, same `q`) regardless of the dollar level it's expressed in, so
this rescaling changes nothing about the dynamics - it just makes "worst performer" comparable
across names.

## 2. Payoff: worst-of conversion

```
worst_of(t) = min_i x_i(t)
Payoff = Principal                              if worst_of(T) ≥ Strike
       = Principal · worst_of(T) / Strike        if worst_of(T) < Strike   (convert into WHICHEVER
                                                   name is currently worst)
```

The direct N-asset generalization of the single-name RC's own `1/Strike` conversion-ratio logic
(`../../Fixed Coupon Note/MATHEMATICS.md` section 3) - same derivation, `S_T` replaced by
`worst_of(T)`.

## 3. Pricing: QuantLib's own basket-option machinery (not hand-rolled)

Unlike the autocallable products in this repo, a **terminal-only** worst-of put has no discrete
monitoring dates to simulate jointly over time, so QuantLib's native basket machinery applies
directly - no hand-rolled Monte Carlo needed here:

- Each name's `BlackScholesMertonProcess`, spot set to its relative performance.
- `ql.StochasticProcessArray` bundles the per-name processes with the full correlation matrix `ρ`
  - QuantLib performs the Cholesky-style correlation transform internally.
- `ql.MinBasketPayoff` wraps a plain vanilla put payoff so it's applied to `min_i x_i(T)`.
- `ql.MCEuropeanBasketEngine` prices it (no closed-form engine exists for baskets of more than 2
  names).

```
Price = ZCB(1.0) - (1/Strike) · worst_of_put_price(relative_spots, Strike, T, σ, q, ρ)
```

## 4. Verification: two independent checks

1. **N=1 degenerate case**: a "basket" of one name must reduce to an ordinary vanilla put
   (QuantLib's `AnalyticEuropeanEngine`) - `MinBasketPayoff` of a single asset is trivially that
   asset.
2. **N=2 convergence**: `MCEuropeanBasketEngine` (the general live pricer) must converge onto
   `ql.StulzEngine`'s exact closed-form price (Stulz 1982 bivariate model) as path count grows,
   for the standard 2-asset worst-of case. Confirms `StochasticProcessArray`'s correlation
   handling and `MinBasketPayoff` are wired correctly before trusting the engine for the full
   N-name basket.

Both print a relative difference and PASS/FAIL at runtime.

## 5. Why dispersion makes the downside worse

With `n` largely uncorrelated names, there are `n` independent chances for *some* name to breach
the strike, versus one chance for a single-name RC - the note's printed "for comparison" block
computes each name's own single-name RC price and confirms the worst-of price is lower than every
one of them, a direct consequence of this effect rather than an assumed one.
