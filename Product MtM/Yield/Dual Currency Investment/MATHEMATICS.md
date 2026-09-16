# The Maths, Fully Decomposed

The mechanical companion to `README.md`. This product is mechanically identical to the Reverse
Convertible with ALT playing "the stock" and DEPOSIT playing "cash" - see
`../Fixed Coupon Note/MATHEMATICS.md` (the RC's math is itself the FCN's) for the ZCB leg and the
`1/Strike` conversion-ratio derivation, both unchanged here. This file covers only what's
FX-specific.

Notation: `S` = DEPOSIT units per 1 ALT unit (see README's quoting-convention section), `r` =
`DEPOSIT_RATE`, `q` = `ALT_RATE`, `T` = time to maturity, `σ` = realized FX volatility.

---

## 1. Garman-Kohlhagen is Black-Scholes-Merton with q = the foreign rate

Holding a unit of foreign currency continuously earns that currency's own risk-free rate - the
exact same mathematical role a continuous dividend yield plays for a stock. QuantLib's
`BlackScholesMertonProcess`, given `q = ALT_RATE`, **is** Garman-Kohlhagen already; no separate FX
option engine is needed:

```
d1 = [ln(S/K) + (r - q + σ²/2)·T] / (σ·√T)
d2 = d1 - σ·√T
Put = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)
```

## 2. Full DCI price

```
Price = ZCB(Principal, T, r) - (1/Strike) · Put(S, Strike·S0, T, r, σ, q)
```

Identical structure to the Reverse Convertible's `ZCB - (1/Strike)·Put` (see
`../Fixed Coupon Note/MATHEMATICS.md` sections 1, 3-4), with the deposit rate standing in for SOFR
and the ALT short rate standing in for a dividend yield.

## 3. Terminal payoff, in the DEPOSIT-per-ALT convention

```
S_T ≥ Strike·S0:  full Principal back, in DEPOSIT
S_T < Strike·S0:  Principal/Strike units of ALT, worth Principal·(S_T/Strike) in DEPOSIT terms
```

See README's worked numeric example for why this direction can look inverted for a pair quoted
the opposite way round (e.g. USDJPY).
