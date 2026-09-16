# Multi-RC (Worst-of Reverse Convertible) Historical Approximation

Traces the historical mark-to-market of a **Multi-RC** along one real historical price path: the worst-of extension of the
single-name `Reverse Convertible` (one folder up). Same "ZCB minus a short put" idea, except the
put is now written on the **worst-of relative performance across a basket** of names, not a
single underlying. Full principal back at maturity unless the **worst-performing name in the
basket** (relative to its own entry level) has fallen below a strike, in which case the note
converts into shares of whichever name that is. Defaults to a three-name basket from three
different sectors (`TICKERS = ["AAPL", "JPM", "XOM"]`) — see "Choosing the basket" below.

No autocall feature - that's what distinguishes a (Multi-)Reverse Convertible from a
(Multi-)Fixed Coupon Note in this repo's naming convention (see `Multi-FCN (Autocallable)`,
`Product MtM/Yield/Fixed Coupon Note/Multi-FCN/`, for the worst-of-plus-autocall combination).

This is a **deterministic historical approximation** — real historical prices, no simulation for the path itself
(the *pricing*, unlike the path, does use Monte Carlo — see "Pricing" below, same distinction the
`Fixed Coupon Note (Autocallable)` product already draws).

> **Scope: model value of the redemption component — excludes coupons.** Everything this script
> computes and charts as "Model Value" is the value of the principal repayment plus the embedded
> worst-of downside feature - the part of the structure that's actually modeled here. It does
> **not** include the value of the periodic coupon cash flows a real Reverse Convertible also
> pays; coupon valuation is out of scope for this project (see `Product MtM/README.md`). Don't
> read the printed/charted value as total investor return or as a full note fair value.

## Why worst-of needs correlation, and why that's unavoidable

A single-name RC's payoff depends only on that one name's own path. A worst-of note's payoff
depends on the **joint** behavior of every name in the basket — specifically, on how often the
names move together versus independently. Low correlation (dispersion) makes a worst-of note
**worse** for the investor than any single-name equivalent at the same strike: with three
largely-independent names, there are three separate chances for *one of them* to breach the
strike, not just one. This is a real, unavoidable feature of the payoff, not a modeling choice —
correlation has to enter the pricing somehow. See the "sanity check" printed by `Multi-RC.py`:
the worst-of note's model value is directly compared against what each name would be worth as a
plain single-name RC in isolation, and the worst-of figure is always lower, by construction.

## Replication

```
Multi-RC = Long Zero-Coupon Bond (pays Principal at maturity)
          - Short Put on min_i(S_i / S0_i) across the basket (struck at STRIKE,
            quantity 1/STRIKE) - the direct N-asset generalization of the
            single-name RC's own short put
```

"Worst-of relative performance" at any date is `min_i(S_i / S0_i)` — each name normalized by its
**own** entry level, never compared in raw dollar terms (a $50 stock and a $500 stock aren't
comparable that way). Below `STRIKE`, the note converts into `Principal/Strike` units of
**whichever name is currently the worst performer** — the standard real-world convention for
worst-of reverse-convertible-style notes, and the direct generalization of the single-name RC's
own `1/Strike` conversion-ratio logic (see that product's README for the full derivation of why
the ratio has to be `1/Strike` and not `1`).

**Terminal payoff**: `Principal * min(worst_of(T) / Strike, 1)`, where `worst_of(T) =
min_i(S_i(T)/S0_i)` — full principal if the worst-of relative performance finishes at or above
`Strike`; below it, principal is reduced in proportion to how far the worst name fell, exactly
like the single-name RC's `Principal*(S/Strike)` but with `S/S0` replaced by whichever name is
currently worst.

## Pricing: QuantLib's own basket-option engines, not a hand-rolled correlated simulation

This is priced entirely through QuantLib's native multi-asset machinery — the same "reach for the
industry-standard engine" principle used everywhere else in this repo, just extended to baskets:

- **`ql.BlackScholesMertonProcess`** — one per name, same building block as every other product
  here, except the spot is normalized to **relative performance** (`1.0` at inception, or
  `S_i(t)/S0_i` at a later mark-to-market date) rather than the name's real dollar price. This
  works because a continuous-dividend GBM's relative-performance process obeys the *exact same*
  SDE (same vol, same dividend yield `q`) regardless of what dollar level it's expressed in — a
  standard, exact rescaling, not an approximation.
- **`ql.StochasticProcessArray`** — bundles the per-name processes together with the full
  correlation matrix. This is where correlation actually enters the model; QuantLib handles the
  matrix decomposition internally, nothing hand-rolled.
- **`ql.MinBasketPayoff`** wrapping a plain `ql.PlainVanillaPayoff(ql.Option.Put, Strike)` — applies
  the put payoff to the *minimum* of the (relative-performance) basket rather than any single name
  in isolation. This one line is what makes it "worst-of."
- **`ql.MCEuropeanBasketEngine`** — the live pricer for any basket size. QuantLib has no
  closed-form engine for baskets of more than two names, so Monte Carlo is the general-purpose
  choice here, same as the autocallable FCN's own MC engine for its path-dependent feature.

**Verification** (`verify_against_closed_form` in `Multi-RC.py`) — two independent checks that
the basket wiring is correct, run every time the script runs:

1. **N=1 degenerate case**: a "basket" of exactly one name must reduce to an ordinary vanilla put
   (`ql.AnalyticEuropeanEngine`) — `MinBasketPayoff` of a single asset is trivially just that
   asset.
2. **N=2 convergence to a closed form**: QuantLib *does* have an exact closed-form engine for
   exactly two names — `ql.StulzEngine` (the Stulz 1982 bivariate worst-of/best-of model). The
   general MC engine, run on a synthetic 2-name case, should converge onto `StulzEngine`'s price
   as the path count grows. This is used **only as a convergence benchmark**, never as the live
   pricer for the actual (3-name) basket — the same "closed form as a sanity check, not the live
   pricer" pattern as the `Barrier Reverse Convertible` product's CRR-vs-`AnalyticBarrierEngine`
   check.

Both checks print a PASS/FAIL and a relative difference every run.

## Choosing the basket: why three different sectors

`TICKERS = ["AAPL", "JPM", "XOM"]` by default — tech, financials, and energy, deliberately chosen
to be only loosely correlated with each other. This is the point: dispersion (low correlation)
is what makes a worst-of note economically distinct from (and worse-valued than) a single-name
note, and same-sector, highly-correlated names would understate that effect. Any list of tickers
works — set `TICKERS` to whatever basket you want to price.

## Vol and correlation: real trailing data, no look-ahead, no fabricated inputs

Both **realized annualized vol per name** and the **full correlation matrix** are estimated from
real historical daily closes — `estimate_vols_and_correlation` computes them directly from log
returns, the same real-data standard as every other number in this repo. Two things distinguish
this from the single-name products' approach:

- **No implied vol term structure is used here, same as the single-name RC for a single-name
  ticker.** There's no free per-name options data source wired up anywhere in this repo, so
  neither file uses implied vol for a stock (only for an index, via VIX/VIX3M/VIX6M - see the
  single-name RC's README). The one difference is the lookback convention: the single-name RC
  computes one flat realized vol from a fixed 2-year trailing window; this file's
  `CORRELATION_LOOKBACK_YEARS` does the same per name, plus the full correlation matrix in one
  pass, since a worst-of basket needs the co-movement between names, not just each one's own vol.
- **`CORRELATION_LOOKBACK_YEARS = 2` is a TRAILING window ending at `ENTRY_DATE`** — vol and
  correlation are estimated from data available *before* the note starts, not from the historical
  window itself. Estimating correlation from the same window you're pricing against the outcome
  of would leak forward-looking information into a day-1 price (look-ahead bias) — the trailing
  window avoids that, at the cost of assuming the trailing 2 years' statistical relationship
  between the names holds going forward, a real and disclosed simplification rather than a hidden
  one.

## Discounting

Same split as the single-name RC: the **ZCB leg** is discounted at `SOFR + Goldman Sachs 5y CDS`
(issuer default risk); the **worst-of put leg** is priced at SOFR alone (a bank prices and hedges
the option on standard derivative terms, not its own funding curve). `Principal = 1.0` ("par"
units) throughout the code, since there's no single dollar `S0` to anchor to with more than one
underlying — every other product in this repo converts to "% of par" only for display, this one
just carries that as the primary unit from the start.

## Greeks: a vector, not a scalar

A multi-asset product's risk genuinely has one Delta and one Vega **per name** — an issuer
hedging this note needs to know its exposure to each underlying separately, not one blended
number. `finite_difference_greeks` bumps each name's relative spot (and separately, each name's
own vol) one at a time, central difference, common random numbers (same `MC_SEED` reused across
every bumped evaluation) so the differences aren't swamped by independent Monte Carlo noise. A
direct bump-size scan (0.01% to 5% of relative spot, holding everything else fixed) showed Delta
stable to within ~2% of its own value across that entire range — this is a **smooth** Monte Carlo
price (not a lattice with discrete "sawtooth" artifacts the way the CRR-priced barrier products
are), so the modest 1%-of-spot / 0.5-vol-point bumps used here are comfortably inside the stable
region, tighter than the CRR products' 2%/2-vol-point convention because there's no equivalent
noise floor to guard against. Rho and Theta stay single, shared numbers (one funding rate, one
clock), same as every other product here.

## Product terms

| Term | Default | Meaning |
|---|---|---|
| `STRIKE` | 90% of each name's own entry level | Worst-of put strike |
| `TICKERS` | `["AAPL", "JPM", "XOM"]` | The basket - any list of Yahoo Finance tickers |
| `CORRELATION_LOOKBACK_YEARS` | 2 | Trailing window (ending at `ENTRY_DATE`) used to estimate real vol/correlation - no look-ahead |
| `RISK_FREE_RATE` | 4% (flat) | SOFR proxy - used for both legs |
| `GS_CDS_SPREAD` | 53.08 bps | Goldman Sachs 5y CDS - issuer credit spread, ZCB leg only |
| `N_MC_PATHS` | 50,000 | Monte Carlo path count for `MCEuropeanBasketEngine` |
| `ENTRY_DATE` / `TENOR` | 2025-01-02 / 1 year | The historical window |

Edit the constants at the top of `Multi-RC.py` to reprice the note, change the basket, or change
the window.

## Output

Running `Multi-RC.py` prints the resolved basket names, the trailing-window realized vol and
correlation matrix, each name's dividend yield, the two QuantLib basket-engine verification
checks, a per-name return summary and which name actually finished worst in this historical path,
the note's model value of the redemption component at inception (as % of par, excluding coupons)
with per-name Delta/Vega plus a shared Rho/Theta, and — as a direct sanity check — what each name
would be worth as a **plain single-name RC** at the same strike, confirming the worst-of figure is
always lower. It then saves a chart with each basket member's own return path (thin lines), the
worst-of Redemption Payoff (Relative to Par) (dashed), and the model value on its own right-hand
axis.

Running `Greek Sensitivity.py` prints and charts the Greeks ladder: strike, funding curve, and
each name's (trailing-window-estimated) vol/correlation/dividend yield held fixed, with a single
**common multiplicative shock applied to every name in the basket at once** as the x-axis (e.g.
"80%" means every name is simultaneously 20% below its own entry level) — the natural stress
scenario for a multi-asset product, and the one that composes into a single chart instead of
needing a separate ladder per name.

## Reading the charts

### `Multi-RC.png` (the historical approximation chart)

- **Thin colored lines** — each basket member's own return from entry (%), one color per name
  (see the legend).
- **Dashed indianred line** — the "Redemption Payoff (Relative to Par)": the terminal payoff formula applied to
  each day's worst-of relative performance. Flat at 0% while the worst-of level stays at or above
  the strike, tracking the worst performer 1:1 below it. This is an illustration of the payoff
  formula, **not** what you'd actually receive if the note were sold or unwound today - it ignores
  all remaining time value in the still-live worst-of put. See the right-axis model-value line for
  the estimate that accounts for that.
- **Dotted blue line** — the strike (in relative-performance terms, applied identically to every
  name).
- **Right axis, solid darkred line** — the model value of the redemption component (% of par,
  excludes coupons, Monte Carlo), converging onto the tracker line exactly at maturity.

### `Greek Sensitivity.png` (the Greeks ladder)

Four panels, x-axis = **a common spot shock applied to every name in the basket at once**
(60%–140%). Strike, funding curve, and each name's own vol/correlation/dividend yield are pinned
at their (real, trailing-window) entry values - only the shared shock moves.

- **Price (% of Par)**: model value of the redemption component under that common shock - smoothly rising as the shock moves
  from a broad selloff toward a broad rally, flattening toward par well above the strike.
- **Delta (per name)**: one line per basket member. All three names start the ladder with similar
  Delta magnitudes (similar vols in the default basket), diverging slightly as their individual
  vols and dividend yields differ.
- **Vega (per name, per 1% vol)**: one line per basket member, always negative (short volatility
  on the worst-of put), largest in magnitude near the strike.
- **Rho (per 1% change in SOFR)**: a single shared line - negative throughout, the same
  ZCB-bond-duration-dominates-the-put's-own-rho story as every other ZCB-minus-put product
  in this repo.

## Usage

```
pip install -r ../../../../requirements.txt
python3 "Multi-RC.py"
python3 "Greek Sensitivity.py"
```

Data comes from `yfinance` (Yahoo Finance) for every name in the basket - no FRED fallback for
individual equities (FRED doesn't carry single-name price series), unlike the index-tracking
products elsewhere in this repo.

## Scope and limitations

See `Product MtM/README.md` for the shared assumptions (vol proxy, flat rates, credit spread,
dividend treatment, numerical limitations) and the project's AI-assisted learning-project
disclosure. This product in particular excludes coupon valuation entirely - see the scope note at
the top of this file.

## The maths

`MATHEMATICS.md` in this folder has the worst-of conversion formula, the QuantLib basket-option
construction, and the N=1/N=2 verification checks, worked out in full.
