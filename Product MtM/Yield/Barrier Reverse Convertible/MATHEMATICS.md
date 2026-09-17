# The Maths, Fully Decomposed

The mechanical companion to `README.md`. The ZCB leg and the conversion-ratio (`1/Strike`)
derivation are identical to the plain Reverse Convertible's - see
`../Fixed Coupon Note/MATHEMATICS.md` sections 1 and 3. This file covers what's specific to the
**down-and-in barrier** put.

Notation: `S0` = spot at inception, `K = Strike · S0`, `H = Barrier · S0` (`H < K`), `r` = the 1Y
Treasury rate (FRED `DGS1` as of `ENTRY_DATE`), `c` = issuer CDS spread, `q` = dividend yield,
`T` = time to maturity, `σ` = volatility, `N` = CRR lattice step count.

---

## 1. Why a CRR lattice, not the closed-form barrier formula

The textbook closed form for a down-and-in put (Reiner-Rubinstein / reflection principle) prices
under **true continuous monitoring** - the probability of touching the barrier at *any instant*
over `[0, T]`. But the note is only ever actually observed once per trading day (via daily
close), and continuous monitoring systematically overstates the touch probability relative to
discrete daily monitoring. Pricing the option one way while checking history a different way
would be an internal inconsistency, not just a simplification - so this script prices the put on
a **Cox-Ross-Rubinstein (CRR) binomial lattice** (QuantLib's `BinomialCRRBarrierEngine`) instead,
with the barrier checked once per lattice step and the step count set to the actual number of
remaining trading days, matching the real observation frequency by construction.

## 2. Lattice mechanics

Over `N` steps of size `Δt = T/N`, with up/down factors and risk-neutral probability:

```
u = e^(σ·√Δt),   d = 1/u,   p = (e^(r·Δt) - d) / (u - d)
```

QuantLib carries two value arrays per lattice node - one assuming the barrier has not yet been
touched on the path arriving at that node, one assuming it has (an ordinary vanilla put value
from that node onward) - and backward-induces both from the terminal payoff to today. A node at
or below the barrier collapses its "not yet touched" value onto the "already touched" value -
exactly the down-and-in knock-in mechanic, resolved once per step instead of via a closed-form
probability.

## 3. In-out parity

A path either touches the barrier or it never does - these two mutually exclusive, exhaustive
outcomes must reconstruct an ordinary vanilla put:

```
Put_vanilla(K) = Put_down-and-in(K, H) + Put_down-and-out(K, H)
```

Checked numerically (both legs priced on the same CRR lattice, same step count), not relied on as
a derivation shortcut - the lattice computes each side independently.

## 4. Once the barrier has been touched

The down-and-in put has permanently "knocked in" and is priced from then on as an **ordinary
vanilla put** (plain Black-Scholes, no lattice needed) - no longer barrier-contingent at all. This
is the mirror image of a down-and-out put (which goes to exactly 0 forever once knocked out):
breaching the barrier here makes the note's future value identical to the plain Reverse
Convertible's, not to a pure bond's.

## 5. Verification (printed at runtime)

Three independent checks:

1. **In-out parity itself** - `down_and_in + down_and_out` (both CRR, same step count) must equal
   the vanilla put to numerical precision.
2. **Convergence to the continuous-monitoring closed form** - as the CRR step count is pushed far
   higher (2,000) than the realistic daily count actually used, the down-and-in price must
   converge onto QuantLib's `AnalyticBarrierEngine` (the same Reiner-Rubinstein closed form) -
   confirms the lattice is wired correctly, even though the live pricer deliberately never uses
   that continuous limit as the actual price.
3. **Barrier pushed unreachable (`H → 0`)** - the down-and-in put can never activate, so it must
   be worth exactly 0, and the note's price must collapse to a **pure zero-coupon bond** (no put
   subtracted at all) - a different limit than the down-and-out case (which would collapse to the
   plain Reverse Convertible instead).

## 6. Greeks: finite difference, wider bumps than the closed-form products

Barrier-option Greeks are messy near the barrier (Delta can be discontinuous), so Greeks are
computed by **central finite differences on the full note price**, not by differentiating the
barrier formula further - same approach as the Bonus-Outperformance certificate's own embedded
barrier put. The CRR step count is fixed once (from the un-bumped `T`) and reused for every
bumped evaluation, so a tiny bump can't round to a different integer step count and inject
lattice-discreteness noise.

`bump_S` and `bump_σ` are **2% of spot / 2 vol points** - much wider than the 0.1%-of-spot
convention used for the closed-form products elsewhere in this repo. A CRR lattice is rebuilt
from scratch on every call, and its node grid scales multiplicatively with both `S` and `σ` - as
those vary continuously, which nodes fall above/below the *fixed* strike/barrier changes in
discrete "sawtooth" jumps, not smoothly. A small bump can land entirely inside one of those jumps
and return a Greek off by an order of magnitude (see the Bullish Sharkfin product's README for
the bump-size scan that surfaced this). Rho keeps a small bump since `r` doesn't enter the
lattice's node spacing at all, only drift/discounting.
