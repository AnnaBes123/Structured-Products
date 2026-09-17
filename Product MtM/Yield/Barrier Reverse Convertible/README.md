# Barrier Reverse Convertible Historical Approximation

Traces the historical mark-to-market of a **Barrier Reverse Convertible** along one real historical price path: a
reverse convertible (bond-like note, enhanced coupon financed by selling a put) whose embedded
put is a **down-and-in put** rather than a vanilla one — the note has full capital protection
regardless of where the index ends up, UNLESS the index trades down to a barrier at some point
in the note's life (checked once per trading day — the barrier can only ever actually be observed
as often as the underlying itself is priced, not literally continuously), in which case the put
"knocks in" permanently and the note starts behaving exactly like a plain reverse convertible for
the rest of its life. Defaults to the S&P 500 (`TICKER = "^GSPC"`), but any single-name stock or
index ticker works - see "Underlying selection" below.

This is a **deterministic historical approximation** — real historical prices, no simulation or Monte Carlo
(the same style as the other products in `Product MtM/`).

> **Scope: model value of the redemption component — excludes coupons.** Everything this script
> computes and charts as "Model Value" is the value of the principal repayment plus the embedded
> down-and-in put - the part of the structure that's actually modeled here. It does **not**
> include the value of the periodic coupon cash flows a real Barrier Reverse Convertible also
> pays; coupon valuation is out of scope for this project (see `Product MtM/README.md`). Don't
> read the printed/charted value as total investor return or as a full note fair value.

## Replication

```
Barrier Reverse Convertible = Long Zero-Coupon Bond (pays Principal at maturity)
                             - Short Down-and-In Put (struck at STRIKE, e.g. 90% of S0,
                               barrier H, e.g. 70% of S0, checked once per trading day)
```

Same short-put financing story as the plain `Reverse Convertible` in the sibling folder — being
short a put is what funds the enhanced coupon — except the put here only exists at all if the
index has touched the barrier at some point. Since a down-and-in put is worth *less* than an
equivalent vanilla put (it can only pay off along a subset of paths — those that actually touch
the barrier — whereas a vanilla put pays off on every path that finishes below strike), the note
gives away a cheaper option than the plain Reverse Convertible, and so finances a **smaller**
coupon for the same strike — verified numerically below.

**Terminal payoff:**
- **Barrier never touched:** full principal back, **regardless of where S_T ends up** — even if
  S_T finishes below the strike. The down-and-in put never activates, so there is no shortfall
  despite being "in the money" in vanilla-put terms.
- **Barrier touched at some point, S_T ≥ Strike:** full principal back (the put has knocked in,
  but finishes out-of-the-money at maturity anyway).
- **Barrier touched at some point, S_T < Strike:** investor is delivered `Principal / Strike`
  shares — the classic reverse-convertible conversion, identical to the plain Reverse
  Convertible from this point on. See that product's README for why the conversion ratio has to
  be `1/Strike` and not `1`.

This payoff depends on the **path** the index took (did it ever touch the barrier?), not just
where it ends up — unlike the plain Reverse Convertible's payoff, which is a simple European put
on the terminal price alone. This is the standard structure sold under the "Barrier Reverse
Convertible" name by most issuers: protection holds unless the barrier is breached, at which
point the note is exposed like an ordinary reverse convertible.

## Pricing the embedded put

Priced on a **Cox-Ross-Rubinstein (CRR) binomial lattice** (QuantLib's `BinomialCRRBarrierEngine`),
not the textbook continuous-monitoring closed form - the note can only ever actually be observed
once per trading day, and continuous monitoring systematically overstates the touch probability
relative to that. The lattice's barrier-check frequency (step count) is set to the actual number
of remaining trading days, so the model's monitoring frequency matches reality by construction.
Once the barrier has been touched, the put is priced from then on as an ordinary vanilla put - no
longer barrier-contingent at all, mirroring the plain Reverse Convertible from that point.

Since a down-and-in put is worth *less* than an equivalent vanilla put (it only pays off on the
subset of paths that touch the barrier), the note gives away a cheaper option than the plain
Reverse Convertible, financing a **smaller** coupon for the same strike - verified numerically at
runtime (three independent checks: in-out parity, convergence to the closed form, and collapse to
a pure ZCB as the barrier is pushed unreachable). Greeks are finite-difference (barrier-option
Greeks are messy near the barrier), with wider bump sizes than the closed-form products - a CRR
lattice rebuilt on every call has discrete "sawtooth" artifacts a small bump can land inside of.

**See `MATHEMATICS.md` in this folder for the full lattice mechanics, the in-out parity identity,
and all three verification checks worked out in formulas.**

## Discounting: two different rates for two different risks

- **ZCB leg** — discounted at `Principal * e^(-(1Y Treasury rate + issuer credit spread) * T)`
  (continuous compounding, matching QuantLib's own `FlatForward` term structures). This is where
  the investor is exposed to the **issuer's own default risk** (a barrier reverse convertible is
  unsecured debt of whichever bank issues it), so the discount rate includes that bank's credit
  spread on top of the risk-free rate.
- **Put leg** — priced via **QuantLib** (CRR lattice for the barrier feature, closed-form
  Black-Scholes-Merton once breached) at **the 1Y Treasury rate alone**, no credit spread. A bank
  prices and hedges the option on standard derivative-pricing terms, not its own funding curve —
  the credit-risk premium belongs entirely to the bond leg, not the option leg.

## Underlying selection: ticker, dividends, and labels

`TICKER` (default `"^GSPC"`) can be set to any Yahoo Finance ticker - an index or a single-name
stock. Two things automatically follow from whatever `TICKER` is set to, no other constants need
touching:

- **Dividend yield** (`fetch_dividend_yield`): a flat, continuous yield `q`, fed into QuantLib's
  process for every leg of the put (the CRR lattice, the closed-form breached case, and the
  continuous-monitoring convergence benchmark) - the same simplification level as the flat
  `RISK_FREE_RATE`. Indices (any `"^"`-prefixed ticker) are treated as paying `q=0`. The fetch
  prefers yfinance's `trailingAnnualDividendYield` field (already a plain fraction) over the
  differently-scaled `dividendYield` field, which Yahoo has at various times returned as a
  **percentage** rather than a fraction (e.g. `0.33` meaning 0.33%, not 33%) - blindly using that
  field would silently overstate the yield by ~100x. Falls back to `dividendRate / price` if even
  that field is missing, and to `q=0` (with a printed note) if the fetch fails entirely.
- **Display name** (`fetch_underlying_name`): pulls the company/index name from yfinance metadata
  (`shortName`/`longName`) for every chart title, axis label, and print statement that used to
  hardcode "S&P 500" - falls back to the raw ticker symbol if the metadata fetch fails.

## Product terms

| Term | Default | Meaning |
|---|---|---|
| `STRIKE` | 90% of entry level | Short put strike / conversion level |
| `BARRIER` | 70% of entry level | Down-and-in barrier (H) for the embedded put, checked once per trading day (CRR lattice) |
| `RISK_FREE_RATE` | 1Y Treasury CMT (FRED `DGS1`) as of `ENTRY_DATE`, held flat | Used for both legs |
| `GS_CDS_SPREAD` | 26.75 bps | Goldman Sachs 1y CDS (Investing.com) - issuer credit spread, ZCB leg only, tenor-matched to `TENOR=1` |
| `TICKER` | `^GSPC` (S&P 500) | Any Yahoo Finance ticker - drives the dividend yield and display name automatically |
| `ENTRY_DATE` / `TENOR` | 2025-01-02 / 1 year | The historical window |

Edit the constants at the top of `Barrier Reverse Convertible.py` to reprice the note, change
the underlying, or change the window.

## How the daily MTM tracks the barrier — conducted on close

The barrier is **conducted on close**: both the historical breach determination and the
forward-looking CRR pricing lattice use the same daily **closing** price, once per trading day —
consistent with each other and with how the note's price is actually ever observed. An earlier
version of this script checked the historical breach against the daily intraday **low** instead
(a proxy for continuous monitoring), while the pricing model priced under a once-per-close
lattice — that mismatch meant a day that dipped through the barrier intraday and closed back
above it would flip the note straight into "breached" pricing at a spot level well above the
barrier, producing a real but somewhat artificial jump in MTM on the observation date. Conducting
on close removes
that mismatch: a day that dips through the barrier intraday and closes back above it does **not**
knock the put in, matching the common real-world term-sheet convention and the resolution the
model prices at going forward.

`fetch_index_path` returns a single daily-close series (no intraday range needed anymore).
Both `barrier_reverse_convertible_mtm_price_series` and `barrier_reverse_convertible_running_return`
track `path.cummin() <= barrier_level` day by day: once the running minimum of the close drops to
or through the barrier, the put leg is priced as an ordinary vanilla put for every subsequent date
(a knock-in, once triggered, stays triggered — "sticky") — the same "sticky barrier" bookkeeping
used for the embedded barrier put in `Bonus-Outperfomance` (which also checks `Close` there).

`barrier_reverse_convertible_mtm_price_series` also passes the actual number of remaining trading
days in the historical path as the CRR lattice's step count, so the forward model checks the
barrier exactly once per future trading day — the finest resolution consistent with how the
note's price will ever actually be observed going forward, deliberately not an (unachievable)
continuous-monitoring assumption.

## Volatility

Same approach as the other products in this repo: for an index, SPX implied vol interpolated
from the VIX / VIX3M / VIX6M term structure for each day's actual remaining time-to-maturity (not
a flat number, and not the raw 30-day VIX applied to a much longer holding period); for a
single-name stock, that ticker's own trailing 2-year realized volatility (held flat, since no
free historical implied-vol source exists for an arbitrary stock the way VIX serves the index).
See the `Product MtM/Participation/Outperformance/` folder's README for the full rationale on
both, and the known limitation beyond 182 days
(VIX6M held flat, since there's no free historical source for a longer-dated implied vol).

## Output

Running `Barrier Reverse Convertible.py` prints the resolved underlying name and dividend yield,
entry/maturity levels, realized returns, whether the barrier was actually touched in this
historical path, the in-out-parity, closed-form-convergence, and unreachable-limit verification
checks, the note's model value of the redemption component at inception (as % of par, excluding
coupons), and its Greeks — then saves a chart with the underlying's price and note payoff on the
left axis and the model value on its own right-hand axis, with dotted lines marking the strike
(blue) and barrier (green), and — if the barrier was touched in this path — a vertical dashed
green line at the knock-in date. Every label (legend, axis, chart title) uses the resolved
underlying name, not a hardcoded "S&P 500".

Running `Greek Sensitivity.py` prints and charts the Greeks ladder: strike, barrier, tenor, vol
and the funding curve held fixed, only spot varies, so each Greek's value at a given spot level
tells you directly how much MTM moves for a 1-unit change in that variable, right now, at that
spot.

## Reading the charts

### `Barrier Reverse Convertible.png` (the historical approximation chart)

- **Left axis, firebrick line** — the real underlying's return from entry (%).
- **Left axis, dashed indianred line** — the "Payoff If Settled Today (Relative to Par)": the terminal payoff formula
  applied to today's level, flat at 0% unless the barrier has *already* been touched in this path
  AND today's level is below the strike (see the down-and-in payoff logic above) - it can stay
  flat even well below the strike, right up until the moment the barrier is actually touched.
  This is an illustration of the payoff formula, **not** what you'd actually receive if the note
  were sold or unwound today - see the model-value line for the estimate that accounts for
  remaining time value.
- **Left axis, dotted lines** — blue marks the strike, green marks the barrier; a dashed green
  vertical line (with a "Barrier Knocked In" label) marks the date the barrier was actually
  touched, if it was, in this historical path.
- **Right axis, solid darkred line** — the model value of the redemption component (% of par,
  excludes coupons), converging onto the tracker line exactly at maturity.

### `Greek Sensitivity.png` (the Greeks ladder)

Four panels, x-axis = **spot as % of S0** (60%–140%). Strike, barrier, tenor, vol and the funding
curve are pinned at day-1 values - only spot moves. Greeks are **finite-difference** (bump and
reprice on the full note price), since barrier-option Greeks have no simple closed form and are
genuinely discontinuous right at the barrier - you'll see a visible kink in Delta/Vega in the one
or two spot scenarios immediately around the 70% barrier line; that's the real discontinuity
being sampled by the bump, not a plotting bug.

- **Price (% of Par)**: model value of the redemption component at that spot level, assuming the
  barrier hasn't been touched yet (except at/below the barrier itself, where it necessarily has
  been).
- **Delta**: points of note value per 1-point index move, right now, at that spot.
- **Vega (per 1% change in vol)**: percentage points of par per 1-percentage-point vol move.
  **Not a fixed number** - recomputed at every spot level; note it can be small or even flip
  sign near the barrier, since a vol increase here has two competing effects - it raises the
  chance of ever touching the barrier (bad, since that activates the put) but also changes the
  value of the put once it's live - which effect wins depends on exactly where spot sits.
- **Rho (per 1% change in the 1Y Treasury rate)**: percentage points of par per 1-percentage-point
  rate move (the risk-free rate here is the 1Y Treasury CMT, FRED `DGS1`, as of `ENTRY_DATE` -
  real and historical, held flat for the note's life, rather than a hand-set number). Worked
  example: a reading of **≈ -0.0081 at spot=100%** means "if the 1Y Treasury rate rose by 1
  percentage point (e.g. 4% to 5%) right now, index unchanged, barrier not yet touched, the note's
  model value would fall by about 0.81 percentage points of par" - roughly $8.10 on a $1,000-par
  note. Negative for the same
  reason as the plain Reverse Convertible: the ZCB leg's bond-duration effect dominates the
  smaller offsetting rho from the short (barrier-contingent) put.

## Usage

```
pip install -r ../../../requirements.txt
python3 "Barrier Reverse Convertible.py"
python3 "Greek Sensitivity.py"
```

Data comes from `yfinance` (Yahoo Finance), falling back to FRED for the S&P 500 and VIX series
if Yahoo is unavailable.

## Scope and limitations

See `Product MtM/README.md` for the shared assumptions (vol proxy, flat rates, credit spread,
dividend treatment, monitoring approximation, numerical limitations) and the project's
AI-assisted learning-project disclosure. This product in particular excludes coupon valuation
entirely - see the scope note at the top of this file.
