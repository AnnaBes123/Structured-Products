# How This Repo Actually Uses QuantLib

Every number in `Product MtM/` ultimately comes out of QuantLib. This doc explains, using
QuantLib's own source documentation rather than paraphrase, what QuantLib is actually being asked
to compute, which of its classes/engines do the work, and — the specific question this doc exists
to answer precisely — what "the barrier is checked once per trading day, not continuously" really
means mechanically. Product-specific formulas live in each product's own README/`MATHEMATICS.md`;
this doc is the one level up: the QuantLib plumbing every product wires up the same way.

## The shared plumbing: `_common.py::_quantlib_process`

Every product builds its pricing environment through one function, now consolidated in
`Product MtM/_common.py`:

```python
def _quantlib_process(S, r, q, sigma, today):
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    rf_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count, ql.Continuous, ql.Annual))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count, ql.Continuous, ql.Annual))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, day_count))
    return ql.BlackScholesMertonProcess(spot, div_ts, rf_ts, vol_ts)
```

Every piece here is a real QuantLib abstraction, not a shortcut this repo invented:

- **`Quote` / `Handle`.** `SimpleQuote` holds one observable number (spot); `QuoteHandle` and
  `YieldTermStructureHandle`/`BlackVolTermStructureHandle` are QuantLib's indirection layer
  (Handle wraps an Observable so downstream objects can be told "this input changed" without being
  rebuilt). This repo doesn't lean on that live-update machinery — a fresh `Handle` is built on
  every single pricing call — which is *why* Greeks here are finite-difference bump-and-reprice
  rather than QuantLib's own analytic Greek accessors propagating a live quote bump: nothing is
  kept alive between calls to bump.
- **`FlatForward`** is QuantLib's simplest `YieldTermStructure` implementation — one constant,
  continuously-compounded rate for every maturity, no bootstrapping from market instruments. It's
  used for both the risk-free curve and the dividend curve. QuantLib supports far richer curves
  (bootstrapped from deposits/futures/swaps); using `FlatForward` here is the repo's documented
  "flat, hand-set rate" simplification (see `Product MtM/README.md`'s Scope and limitations), made
  concrete: it is literally the flattest term structure QuantLib ships.
- **`BlackConstantVol`** is the equivalent flat simplification for the vol surface — one number,
  no smile/skew, no term structure beyond the single tenor interpolated externally (see each
  product's vol-proxy discussion) before ever reaching QuantLib.
- **`BlackScholesMertonProcess`** is the actual stochastic process every option in this repo is
  priced under. Per QuantLib's own header documentation, it represents

  ```
  d ln S(t) = (r(t) - q(t) - σ(t,S)²/2) dt + σ dW(t)
  ```

  — geometric Brownian motion in the log-price, with a continuous dividend yield `q` subtracted
  from the drift. This is the one process every engine below (analytic, binomial, or Monte Carlo)
  ultimately discretizes or integrates; the engine changes the pricing *method*, never the
  underlying dynamics assumed.
- **`NullCalendar()` + a synthetic anchor date.** QuantLib normally drives day counts and cash-flow
  schedules off a real trading calendar (holidays, weekends) and a real evaluation date. This repo
  does the opposite: every product computes `T` itself, in Python, from two real historical
  dates (`(maturity_date - date).days / 365.25`), and only then hands QuantLib a synthetic
  `ql.Date(1, 1, 2000)` anchor plus a `ql.Period(days, ql.Days)` built from that already-computed
  `T`. `NullCalendar` (every day is a business day) is what makes that period arithmetic land
  exactly on the intended `T` instead of QuantLib silently rolling it to the next "business day."
  In other words: QuantLib is deliberately used here as a pricing-formula calculator (give it
  `S, K, T, r, q, σ`, read back a price), not as a schedule/calendar engine — the real calendar
  work (matching actual trading days) is already done in pandas before QuantLib is ever called.

## Instrument + Engine: the same option, priced four different ways

QuantLib's central design pattern — an `Instrument` (what's being priced: a payoff + an exercise
style) is separate from the `PricingEngine` (how it's priced) — is not just a paraphrase here; you
can see it directly in QuantLib's own barrier-engine registry
(`quantlib.org/reference/group__barrierengines.html`), which lists sixteen different engines for
the *same* `BarrierOption` instrument: analytic closed-form engines (`AnalyticBarrierEngine`,
`AnalyticDoubleBarrierEngine`, ...), binomial-tree engines (`BinomialBarrierEngine` and its
Cox-Ross-Rubinstein specialization), finite-difference engines (`FdBlackScholesBarrierEngine`,
`FdHestonBarrierEngine`), and Monte Carlo engines (`MCBarrierEngine`) — all constructible against
the identical option object, via `option.setPricingEngine(...)`. Swapping the engine changes
*only* the numerical method; the payoff and exercise stay the same object. This repo uses exactly
four of QuantLib's engines, each for a specific reason:

| Engine | Used for | Method |
|---|---|---|
| `AnalyticEuropeanEngine` | Every plain vanilla call/put leg (warrants, LEPO-adjacent ATM calls, vanilla puts once a barrier has knocked in) | Exact closed-form Black-Scholes-Merton |
| `BinomialCRRBarrierEngine` | Every single-name barrier leg (Sharkfins, Bonus/Bonus-Outperformance/Twin-Win Certificates, Barrier Reverse Convertible) | Cox-Ross-Rubinstein binomial lattice, barrier checked once per lattice step |
| `AnalyticBarrierEngine` | Barrier Reverse Convertible only, as a **convergence benchmark**, never the live price (`down_and_in_put_continuous_benchmark`) | Exact closed-form (Reiner-Rubinstein), assumes **continuous** monitoring |
| `MCEuropeanBasketEngine` + `StochasticProcessArray` + `MinBasketPayoff` | Multi-RC / Multi-FCN worst-of baskets | Correlated multi-asset Monte Carlo (QuantLib's own basket-Monte-Carlo machinery, not a hand-rolled simulator) |
| `StulzEngine` | Multi-RC's N=2 verification check only, never the live 3-name pricer | Exact closed-form (Stulz, 1982, *Journal of Financial Economics*) for a 2-asset min/max basket |

The last row is the same pattern one level up: QuantLib has no exact closed-form engine for a
3-name worst-of basket (none exists in closed form for N>2), so the live pricer has to be Monte
Carlo — but QuantLib *does* have an exact engine for N=2, so that's used purely to confirm the
general N-asset Monte Carlo code converges onto a known-correct answer at N=2, before trusting it
at N=3.

## The important caveat: "checked once per trading day," not continuously

**Yes — that's an accurate description of the deliberate design, with one nuance worth being
precise about: it means two different things depending on whether you're looking backward or
forward from a given date on the chart.**

**Backward (the realized part of the path):** whether the barrier has already been touched is
read directly off real daily closing prices — `path.cummin() <= barrier_level`. There's no
QuantLib involved in that judgment at all; it's a plain comparison against the actual historical
close series. This part is unambiguous: the barrier genuinely is monitored on close, because
that's the only data that exists.

**Forward (the remaining, not-yet-realized life at each date):** there is no future path to
sample, so the not-yet-elapsed life is priced with a CRR binomial lattice whose step count is set
to the actual number of remaining trading days (`steps_remaining = n_dates - 1 - i`). Each lattice
time-slice functions as one barrier check, so "one step per remaining trading day" is the model's
way of making the forward-looking discretization match the real observation frequency, without
needing to simulate an actual future daily path. This is the standard, correct way to price a
discretely-monitored barrier option — but it is a matched discretization, not a literal replay of
future daily closes (which don't exist yet).

**Why not just use QuantLib's closed-form barrier formula and skip the lattice entirely?** Because
that formula (`AnalyticBarrierEngine`, implementing Haug's / Reiner-Rubinstein's formulas) prices
under **true continuous monitoring** — the probability of touching the barrier at *any instant*
over `[0,T]`, not just at each day's close. Continuous monitoring systematically overstates the
touch probability relative to discrete daily monitoring (this is standard, well-known option-pricing
theory — see Broadie, Glasserman & Kou (1997), *A Continuity Correction for Discrete Barrier
Options*, for the classic quantification of the gap). Pricing the option under a continuous-time
assumption while judging its actual history on daily closes would be internally inconsistent, so
every barrier product here uses the discrete lattice as the live pricer and treats the continuous
closed form only as a **wiring check** (see below) — never as the number that gets charted.

**A subtlety QuantLib itself flags, and how this repo avoids it:** `BinomialCRRBarrierEngine`'s own
class documentation notes that "timesteps for Cox-Ross-Rubinstein trees are adjusted using [the]
Boyle and Lau algorithm" (*Journal of Derivatives*, 1/1994, "Bumping up against the barrier with
the binomial method") — QuantLib can silently *increase* the step count internally, beyond what a
caller requests, to align a tree layer exactly with the barrier level and reduce convergence
oscillation. If that adjustment fired here, "steps = remaining trading days" would stop being
exactly true — the lattice would use more time-slices than there are trading days, decoupling the
model's monitoring frequency from the one it's supposed to match. It's disabled by construction:
`BinomialCRRBarrierEngine`'s constructor takes `(process, timeSteps, maxTimeSteps)`, and the
adjustment only ever fires when `maxTimeSteps > timeSteps`; every barrier product in this repo
calls it as `ql.BinomialCRRBarrierEngine(process, n, n)` — identical values — which QuantLib
documents as disabling Boyle-Lau outright. So the step count really is exactly `n`, one barrier
check per remaining trading day, with no silent adjustment underneath it.

**What the "closed-form verification" checks actually prove here, precisely:** Barrier Reverse
Convertible's `verify_against_closed_form` pushes the CRR step count far higher (2,000, vastly
finer than any realistic daily count) and confirms *that* number converges onto
`AnalyticBarrierEngine`'s continuous closed form. That check validates that the lattice engine
itself is correctly wired — it says nothing about, and is never used to claim, that the discrete
daily-monitored price (the one actually charted, at the real trading-day step count) equals the
continuous one. Those two numbers are expected to differ, by exactly the amount the discreteness
correction literature describes — that gap is real economics the discrete lattice is capturing on
purpose, not noise to be reconciled away.

## What this doesn't prove

Everything above is about QuantLib computing the *mechanics* correctly (the right formula, the
right lattice, the right Monte Carlo basket machinery) given the inputs it's handed. It says
nothing about whether those inputs — a flat rate, a flat proxy vol, a single illustrative credit
spread, `NullCalendar`'s "every day is a business day" — are realistic. See `Product MtM/README.md`
Scope and limitations for that half of the picture; this doc and that section are meant to be read
together.

## Sources consulted

- QuantLib barrier-engine registry: `quantlib.org/reference/group__barrierengines.html`
- `BinomialBarrierEngine` class documentation and Boyle-Lau step-adjustment logic:
  `github.com/lballabio/QuantLib/blob/master/ql/pricingengines/barrier/binomialbarrierengine.hpp`
- `BlackScholesMertonProcess` SDE and constructor documentation:
  `github.com/lballabio/QuantLib/blob/master/ql/processes/blackscholesprocess.hpp`
- `AnalyticBarrierEngine` (Haug's closed-form barrier formulas):
  QuantLib reference manual, `class_quant_lib_1_1_analytic_barrier_engine`
- `StulzEngine` (Stulz, 1982, *Journal of Financial Economics* 10, 161-185, "Options on the
  Minimum or the Maximum of Two Risky Assets"): `ql/pricingengines/basket/stulzengine.hpp`
- This repo's own `Yield/Barrier Reverse Convertible/MATHEMATICS.md` (sections 1, 2, and 5) and
  `Barrier Reverse Convertible.py` (`_price_barrier_put`, `verify_against_closed_form`), which is
  where the Boyle-Lau-disabling call convention and the continuous-benchmark check are actually
  implemented and documented at the product level.
