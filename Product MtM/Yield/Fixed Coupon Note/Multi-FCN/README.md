# Multi-FCN (Autocallable) Historical Approximation

Traces the historical mark-to-market of a **Multi-FCN (Autocallable)** along one real historical price path: the autocall
extension of `Multi-RC` (`Product MtM/Yield/Reverse Convertible/Multi-RC/`) — same worst-of
basket, same short put on the worst-relative-performer, but now with a quarterly autocall
feature layered on top, the same feature the single-name `Fixed Coupon Note (Autocallable)`
(one folder up) already has. Full principal back early if **every name in the basket**
simultaneously closes at or above a trigger on a quarterly observation date; otherwise the
worst-of put logic from `Multi-RC` applies at maturity. Defaults to the same three-name basket
as `Multi-RC` (`TICKERS = ["AAPL", "JPM", "XOM"]`) for direct comparability between the two.

This is a **deterministic historical approximation** — real historical prices, no simulation for the path itself
(the *pricing* does use Monte Carlo — see "Pricing" below).

> **Scope: model value of the redemption component — excludes coupons.** Everything this script
> computes and charts as "Model Value" is the value of the principal repayment plus the embedded
> worst-of downside/autocall feature - the part of the structure that's actually modeled here. It
> does **not** include the value of the periodic coupon cash flows a real note like this also
> pays; coupon valuation is out of scope for this project (see `Product MtM/README.md`). Don't
> read the printed/charted value as total investor return or as a full note fair value.

## Dispersion cuts both ways for a worst-of note

`Multi-RC`'s README makes the case that dispersion (low correlation across the basket) makes the
**downside worse** for a worst-of note than any single-name equivalent — more independent chances
for one name to breach the strike. This file is the other half of that story: dispersion also
makes the **upside (the autocall) harder to reach**. The autocall condition is `worst_of >=
TRIGGER`, which literally means *every single name* has to be simultaneously at or above the
trigger on the same observation date — much harder to achieve with three loosely-correlated names
than with one. Both effects have the same root cause (dispersion) and the same sign for the
investor (worse than a single-name equivalent), and both show up in this repo as real, computed
numbers rather than assertions — see the "For comparison" block `Multi-FCN (Autocallable).py`
prints, which checks that the autocallable price is *higher* than the no-autocall (`Multi-RC`)
price at the same strike (autocall optionality can only help), and `Multi-RC`'s own README shows
the mirror-image check on the downside.

## Replication

```
Multi-FCN (Autocallable) = Long Zero-Coupon Bond (Principal)
                          - Short Put on min_i(S_i/S0_i) (struck at STRIKE, qty 1/STRIKE)
                          - a worst-of autocall feature at each quarterly observation
                            date: if EVERY name closes at/above TRIGGER (equivalently,
                            worst_of >= TRIGGER), the note redeems immediately at
                            Principal and the worst-of put is knocked out
```

Everything below the autocall feature is identical to `Multi-RC` — same worst-of put, same
`1/Strike` conversion-ratio logic, same relative-performance normalization (each name divided by
its own entry level). See that product's README for the full derivation of the worst-of put
mechanics. The only new piece here is the autocall feature itself, generalized directly from the
single-name `Fixed Coupon Note (Autocallable)`'s own quarterly-observation, discretely-monitored
up-and-out-style structure — except the "up-and-out" trigger is now checked against the **worst**
name in the basket, not a single underlying.

## Pricing: a genuinely hand-rolled correlated multi-step Monte Carlo, and why

Unlike `Multi-RC` (which prices its terminal-only worst-of put entirely through QuantLib's own
`MCEuropeanBasketEngine`), this file's live pricer (`simulate_forward_price`) is **hand-rolled**
— there is no QuantLib basket-autocallable engine, and no closed-form or scipy-multivariate-normal
shortcut generalizes cleanly here either:

- The single-name `Fixed Coupon Note (Autocallable)`'s own closed-form sanity check
  (`autocall_digital_strip_price`) relies on a joint multivariate normal over **time** for one
  asset (the standardized Brownian value at each observation date, correlated across dates via
  `corr(X_i, X_j) = sqrt(min(t_i,t_j)/max(t_i,t_j))`). Extending that same trick to a worst-of
  basket would require a joint multivariate normal over **both time and assets at once** — a
  distribution of dimension `N_assets x N_obs_dates` (12-dimensional for this file's default 3
  names x 4 observation dates). `scipy.stats.multivariate_normal.cdf` handles that badly: its
  default algorithm past a handful of dimensions falls back to its own internal Monte Carlo
  estimate, which is slower and noisier than just simulating the actual GBM paths directly. So
  this file doesn't attempt it.
- QuantLib itself has no basket-option engine with early-exercise/autocall logic (`BasketOption`
  only supports European and American exercise on the terminal payoff, not a discrete-date
  knock-out feature).

So `simulate_forward_price` generalizes the single-name Autocallable FCN's own hand-rolled MC
engine directly: an added **asset axis**, correlated at each time step via a Cholesky
decomposition of the correlation matrix (`np.linalg.cholesky`) applied to that step's
innovations, with a `min` reduction across assets before the existing first-passage-over-time
logic (first-triggered-date detection via `np.argmax` on a boolean trigger array) runs unchanged.
QuantLib is still used, just narrower in scope: for the process/vanilla-option verification
benchmarks below, not as the live pricer.

## Verification: two nested reduction checks

`verify_against_closed_form` in `Multi-FCN (Autocallable).py` — both checks push the autocall
trigger to an unreachable level (`10.0`, i.e. 1000% of entry, never happens) and confirm the
hand-rolled engine collapses onto an independently-verified benchmark:

1. **N=1 reduction**: a synthetic single-name basket, autocall unreachable, should match that
   name's plain closed-form FCN price (`ZCB - put_quantity * vanilla put`, the vanilla put priced
   via QuantLib's `AnalyticEuropeanEngine`) — confirms the time-stepping/discounting machinery on
   its own, with the worst-of-across-assets reduction trivial (one asset).
2. **N=3 reduction**: the actual 3-name basket, autocall unreachable, should match `Multi-RC`'s
   own terminal-only worst-of price (computed the same way `Multi-RC.py` computes it — via
   QuantLib's `MCEuropeanBasketEngine`, included in this file as `worst_of_put_price`) —
   specifically validates that the worst-of-across-assets reduction, now happening *inside* a
   multi-step time grid rather than at a single terminal date, is still wired correctly.

Both print a PASS/FAIL and a relative difference every run; both passed at ~0.004% and ~0.09%
respectively in testing.

## Vol, correlation, and the basket

Identical approach to `Multi-RC`: realized annualized vol per name and the full correlation
matrix are estimated from a `CORRELATION_LOOKBACK_YEARS`-year (default 2) **trailing** window of
real daily closes ending at `ENTRY_DATE` — no look-ahead into the historical window, no implied vol
(no free per-name options data source exists), same basket-diversity reasoning (three different
sectors, deliberately only loosely correlated) — see `Multi-RC`'s README for the full discussion,
which applies unchanged here.

## Discounting

Same split as every ZCB-minus-put product in this repo: the **ZCB/principal leg** at
`SOFR + Goldman Sachs 5y CDS` (issuer default risk, paid whenever the note actually settles -
early call or maturity); the **worst-of put leg** at SOFR alone. `Principal = 1.0` ("par" units)
throughout, same convention as `Multi-RC`.

## Greeks: a vector, not a scalar - and a note on Rho's noise

Same reasoning as `Multi-RC`: Delta and Vega are reported **per name** (bump-and-reprice, central
difference, common random numbers via a shared `MC_SEED` across every bumped evaluation). Price,
Delta and Vega all come out smooth across the spot ladder. **Rho is visibly noisier**, especially
above the trigger — this is expected, not a bug: the autocall boundary makes `settle_time` (and
therefore the discounting) a genuinely discontinuous function of the simulated path, so a small
rate bump can flip a nontrivial number of near-the-boundary paths' realized `settle_time` even
holding the random draws fixed, injecting more Monte Carlo noise into that particular finite
difference than into Delta/Vega (which perturb the diffusion directly, not just the discounting
of an already-decided outcome). A real trading desk pricing an actual autocallable sees the same
effect and manages it the same way - more paths, or accepting the noise as a known limitation of
Monte Carlo Greeks near a discrete barrier - rather than something a "correct" implementation
would eliminate entirely.

## Product terms

| Term | Default | Meaning |
|---|---|---|
| `STRIKE` | 90% of each name's own entry level | Worst-of put strike |
| `TRIGGER` | 100% of each name's own entry level | Autocall trigger - ALL names must be at/above it |
| `OBS_PER_YEAR` | 4 | Quarterly autocall observation dates |
| `TICKERS` | `["AAPL", "JPM", "XOM"]` | The basket - same default as `Multi-RC`, any list of Yahoo Finance tickers works |
| `CORRELATION_LOOKBACK_YEARS` | 2 | Trailing window (ending at `ENTRY_DATE`) for real vol/correlation - no look-ahead |
| `RISK_FREE_RATE` | 4% (flat) | SOFR proxy |
| `GS_CDS_SPREAD` | 53.08 bps | Goldman Sachs 5y CDS - issuer credit spread, ZCB leg only |
| `N_MC_PATHS` | 50,000 | Monte Carlo path count |
| `ENTRY_DATE` / `TENOR` | 2025-01-02 / 1 year | The historical window |

Edit the constants at the top of `Multi-FCN (Autocallable).py` to reprice the note, change the
basket, or change the window.

## Output

Running `Multi-FCN (Autocallable).py` prints the resolved basket, the trailing-window vol and
correlation matrix, each name's dividend yield, the quarterly autocall schedule with the worst-of
relative performance at each observation date, whether (and when) the note actually autocalled in
this historical path, both reduction-check verifications, a per-name return summary, the note's
model value of the redemption component at inception (excluding coupons) with per-name Delta/Vega
and shared Rho/Theta, the empirical probability-of-call from the Monte Carlo, and — as a direct
sanity check — what the same basket would be worth with no autocall feature at all (`Multi-RC`, in
effect), confirming the autocallable price is always higher. It then saves a chart with each
basket member's own return path, the worst-of Redemption Payoff (Relative to Par), quarterly
observation-date markers, the actual autocall date (if any), and the model value on its own
right-hand axis.

Running `Greek Sensitivity (Autocallable).py` prints and charts the Greeks ladder: strike,
trigger, funding curve, and each name's (trailing-window) vol/correlation/dividend yield held
fixed, with a **common multiplicative shock applied to every name in the basket at once**, all
observation dates evaluated as of entry (day 1) — same x-axis convention as `Multi-RC`'s own
ladder.

## Reading the charts

### `Multi-FCN (Autocallable).png` (the historical approximation chart)

- **Thin colored lines** — each basket member's own return from entry (%).
- **Dashed indianred line** — the "Redemption Payoff (Relative to Par)": the terminal payoff formula applied to
  each day's worst-of relative performance, flat at 0% once the note has actually autocalled in
  this historical path (see the darkgreen "Autocalled" marker). This is an illustration of the
  payoff formula, **not** what you'd actually receive if the note were sold or unwound today - see
  the model-value line for the estimate that accounts for remaining time value.
- **Mediumseagreen dotted vertical lines** — every quarterly observation date, whether or not it
  triggered the autocall; a darkgreen dashed line marks the one that actually did (if any).
- **Dotted blue lines** — the strike and the (higher) autocall trigger, both in relative-
  performance terms applied identically to every name.
- **Right axis, solid darkred line** — the model value of the redemption component (% of par,
  excludes coupons, Monte Carlo). **This line ends at the actual autocall date** if the note
  called in this historical path - once settled, there is no live note left to mark.
- **Right axis, gray dotted line ("Cash Proceeds After Autocall")** — appears only if the note
  actually autocalled: the par amount received at settlement, held flat with no reinvestment
  assumed, through the end of the charted window. This is cash-in-hand accounting, not a
  continuing mark-to-market of the (now-settled) note, and it also excludes coupons.

### `Greek Sensitivity (Autocallable).png` (the Greeks ladder)

Four panels, x-axis = **a common spot shock applied to every name in the basket at once**
(60%–140%), all observation dates still in the future (evaluated as of entry). Strike, trigger,
funding curve, and each name's own vol/correlation/dividend yield are pinned at their (real,
trailing-window) entry values.

- **Price (% of Par)**: smoothly rising, visibly flattening as the shock moves well above the
  trigger (the note is increasingly certain to autocall at the first observation date regardless
  of what happens after).
- **Delta (per name)**: one line per basket member, smoothly falling as spot rises - once deep
  above the trigger, Delta falls toward 0 (the note is about to be called at a fixed par amount
  regardless of further upside, so further moves in any name stop mattering).
- **Vega (per name, per 1% vol)**: one line per basket member, negative near the strike (more vol
  raises breach risk) but turning positive well above the trigger (more vol there raises the
  *autocall* probability sooner, which is good for the investor - the same sign-flipping-with-
  spot behavior as the single-name Autocallable FCN's own Vega).
- **Rho (per 1% change in SOFR)**: a single shared line, negative and visibly noisier past the
  trigger - see "Greeks: a vector, not a scalar" above for why that's expected Monte Carlo
  behavior near a discrete autocall boundary, not a bug.

## Usage

```
pip install -r ../../../../requirements.txt
python3 "Multi-FCN (Autocallable).py"
python3 "Greek Sensitivity (Autocallable).py"
```

Data comes from `yfinance` (Yahoo Finance) for every name in the basket - no FRED fallback for
individual equities, same as `Multi-RC`.

## Scope and limitations

See `Product MtM/README.md` for the shared assumptions (vol proxy, flat rates, credit spread,
dividend treatment, numerical limitations) and the project's AI-assisted learning-project
disclosure. This product in particular excludes coupon valuation entirely - see the scope note at
the top of this file.

## The maths

`MATHEMATICS.md` in this folder has the full worked-out correlated multi-asset Monte Carlo
construction and the two nested closed-form verification checks, building on the single-name
Autocallable FCN's own `../MATHEMATICS.md` and Multi-RC's `../../Reverse
Convertible/Multi-RC/MATHEMATICS.md`.
