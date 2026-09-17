# Bearish Sharkfin Historical Approximation

Traces the historical mark-to-market of a **Bearish Sharkfin** along one real historical price path: a capital-protected
note that gives full 1:1 *downside* participation below a strike, as long as the index never
trades down to a barrier. Touch the barrier once and the participation is knocked out for good -
the note then simply returns principal at maturity. Defaults to the S&P 500
(`TICKER = "^GSPC"`), but any single-name stock or index ticker works - see "Underlying
selection" below.

This is a **deterministic historical approximation** — real historical prices, no simulation or Monte Carlo
(the same style as the other products in `Product MtM/`).

> **"Model Value"** means this script's own computed value of the product's payoff, as a
> percentage of par, under the assumptions listed below (flat rates, a vol proxy, etc.) — read it
> as "what this simplified model says the payoff is worth today," not a market quote or a claim
> that the product could actually be bought or sold at that level. See `Product MtM/README.md`
> for the full explanation.

This is the mirror image of the `Bullish Sharkfin` product in this same folder: same ZCB +
barrier-option architecture, flipped onto the downside - a long put instead of a long call, a
barrier below the strike instead of above it. Read that product's README first if this is your
first time in `Product MtM/Capital Protection/`; most of the mechanics (CRR lattice, "conducted
on close", the wider Greeks bump sizes) are described there in full and only summarized here.

## Replication

```
Bearish Sharkfin = Long Zero-Coupon Bond (pays Principal at maturity)
                  + Long Down-and-Out Put (struck at STRIKE=100% of S0,
                    barrier H, e.g. 80% of S0, checked once per trading day)
```

The **ZCB leg** makes this capital-protected: it pays Principal at maturity regardless of the
option leg, so "if the underlying finishes above strike, capital is returned at 100%" falls
straight out of the put being worthless there.

The **down-and-out put** gives full, uncapped 1:1 participation in the **downside** below the
strike (the note gains value as the index falls) as long as the barrier is never touched. Touch
it once and the put is knocked out for good - the note then just pays back principal at
maturity, plus whatever `REBATE` was already received at the moment of the knock-out (not part
of the note's forward MTM from then on, since it has already been paid).

**Terminal payoff:**
- **Barrier never touched:** `Principal + max(Strike - S_T, 0)` — full uncapped downside
  participation.
- **Barrier touched at some point:** `Principal` (+ any rebate already received) — capital
  returned at 100%, no further participation, regardless of where the index ends up afterward.

## Pricing the embedded put

Priced on a **Cox-Ross-Rubinstein binomial lattice** (QuantLib `BinomialCRRBarrierEngine`),
barrier checked once per lattice step, step count = remaining trading days - see the `Barrier
Reverse Convertible` product's README (`Product MtM/Yield/`) for the full continuous-vs-discrete-
monitoring rationale, and the `Bullish Sharkfin` product's README (this folder) for the specific
finite-difference bump-size scan that motivates the wider (2%-of-spot / 2-vol-point) Delta/Vega
bumps used here - the same CRR "sawtooth" lattice-noise issue applies to any barrier product
priced this way, not just the up-barrier case.

**Sanity check** (`verify_against_closed_form`): with the barrier pushed unreachable and the
rebate forced to 0, the down-and-out put can never knock out, so its CRR price should converge
onto the plain vanilla put struck at `STRIKE` - confirms the barrier engine collapses to the
ordinary closed form in the no-barrier limit.

See `Product MtM/Yield/Barrier Reverse Convertible/MATHEMATICS.md` for the full CRR lattice
mechanics (same engine, just a down-and-out put here instead of down-and-in).

## Underlying selection: stocks (with dividends) vs. indices (no dividends)

`TICKER` (default `"^GSPC"`) can be set to any Yahoo Finance ticker - the **same mechanism**
used throughout this repo, no separate flag needed:

- **Set `TICKER` to a Yahoo index ticker** (anything starting with `"^"`, e.g. `"^GSPC"`,
  `"^FTSE"`) to model an index underlying: `fetch_dividend_yield` short-circuits to `q=0`
  immediately, no network call needed for the yield itself (Yahoo doesn't expose a meaningful
  per-ticker dividend field for indices anyway).
- **Set `TICKER` to a real stock ticker** (e.g. `"AAPL"`, `"MCD"`) to model a dividend-paying
  single name: `fetch_dividend_yield` fetches a real trailing dividend yield from yfinance and
  feeds it into QuantLib's process as a flat, continuous `q` - the same simplification level as
  the flat `RISK_FREE_RATE`. It prefers yfinance's `trailingAnnualDividendYield` field (already a
  plain fraction) over the differently-scaled `dividendYield` field, which Yahoo has at various
  times returned as a **percentage** rather than a fraction (e.g. `0.33` meaning 0.33%, not 33%)
  - blindly using that field would silently overstate the yield by ~100x. Falls back to
  `dividendRate / price` if even that field is missing, and to `q=0` (with a printed note) if the
  fetch fails entirely.

`fetch_underlying_name` similarly pulls the company/index display name from yfinance metadata for
every chart title, axis label, and print statement, regardless of which kind of ticker is used -
falls back to the raw ticker symbol if the metadata fetch fails.

## Discounting

- **ZCB leg** — discounted at `Principal * e^(-(risk-free rate + issuer credit spread) * T)`
  (continuous compounding, matching QuantLib's own `FlatForward` term structures), using the
  Goldman Sachs 1y CDS spread (26.75 bps, Investing.com) as the issuer credit spread, same convention as every other
  principal-at-risk-of-issuer-default product in this repo.
- **Put leg** — priced with risk-neutral pricing at the **risk-free rate alone**, no credit
  spread.

## Product terms

| Term | Default | Meaning |
|---|---|---|
| `STRIKE` | 100% of entry level | Long put strike (at the money) |
| `BARRIER` | 80% of entry level | Down-and-out barrier (H) for the embedded put, checked once per trading day (CRR lattice) |
| `REBATE` | 0.0 | Cash paid immediately if the barrier is touched, in underlying currency (not a fraction of S0) - set by hand, not economically important for MtM purposes |
| `RISK_FREE_RATE` | `fetch_risk_free_rate(ENTRY_DATE)` | 1Y Treasury CMT (FRED `DGS1`) as of `ENTRY_DATE`, held flat for the note's life - real and historical, but still a single point on the curve, not a bootstrapped term structure. Used for both legs. |
| `GS_CDS_SPREAD` | 26.75 bps | Goldman Sachs 1y CDS (Investing.com) - issuer credit spread, ZCB leg only, tenor-matched to `TENOR=1` |
| `TICKER` | `^GSPC` (S&P 500) | Any Yahoo Finance ticker - index (no dividend) or stock (real dividend yield fetched), drives the display name too |
| `ENTRY_DATE` / `TENOR` | 2025-01-02 / 1 year | The historical window |

Edit the constants at the top of `Bearish Sharkfin.py` to reprice the note, change the
underlying, or change the window.

## How the daily MTM tracks the barrier

`bearish_sharkfin_mtm_price_series` and `bearish_sharkfin_running_return` track
`path.cummin() <= barrier_level` day by day — a running **minimum** compared against a **lower**
barrier (the mirror image of the Bullish Sharkfin, which tracks a running maximum against an
upper barrier). Conducted on close: once the running minimum of the close falls to or through the
barrier, the put leg is priced at exactly 0 for every subsequent date (a knock-out, once
triggered, stays triggered — "sticky"). `bearish_sharkfin_mtm_price_series` also passes the
actual number of remaining trading days as the CRR lattice's step count for each date.

## Output

Running `Bearish Sharkfin.py` prints the resolved underlying name and dividend yield,
entry/maturity levels, realized returns, whether the barrier was actually touched in this
historical path, the closed-form verification check, the note's model value at inception (as % of
par), and its Greeks — then saves a chart with the underlying's price and note payoff on the left
axis and the note's model value on its own right-hand axis, with dotted lines marking the strike
(blue) and barrier (green), and — if the barrier was touched in this path — a vertical dashed
green line at the knock-out date.

Running `Greek Sensitivity.py` prints and charts the Greeks ladder: strike, barrier, tenor, vol
and the funding curve held fixed, only spot varies.

## Reading the charts

### `Bearish Sharkfin.png` (the historical approximation chart)

- **Left axis, firebrick line** — the real underlying's return from entry (%).
- **Left axis, dashed indianred line** — the "Payoff If Settled Today (Relative to Par)": the terminal payoff formula
  applied to each day's spot (flat at 0% above the strike, rising 1:1 as the index falls below it,
  as long as never breached). This is **not** what you'd actually receive if the note were sold
  or unwound on that date - it ignores all remaining time value in the still-live put. It's the
  closest thing to a running "how much downside has this path locked in so far" readout, not a
  settlement value - see the right-axis MTM line for the model-value estimate.
- **Left axis, dotted lines** — blue marks the strike, green marks the barrier; a dashed green
  vertical line (with a "Barrier Knocked Out" label) marks the date the barrier was actually
  touched, if it was, in this historical path.
- **Right axis, solid darkred line** — the note's model value (% of par), converging onto the
  tracker line exactly at maturity (an option's time value is 0 at expiry, so the two are
  identical by then).

### `Greek Sensitivity.png` (the Greeks ladder)

Four panels, x-axis = **spot as % of S0** (60%–140%). Strike, barrier, tenor, vol and the funding
curve are pinned at day-1 values - only spot moves. Greeks are **finite-difference** (bump and
reprice, with the wider Delta/Vega bumps described above), since barrier-option Greeks have no
simple closed form and are genuinely discontinuous right at the barrier.

- **Price (% of Par)**: the mirror-image "shark fin" shape - flat at the pure-ZCB value at and
  below the barrier (already breached there), rises as spot moves away from the barrier, peaks
  somewhere near the strike, then declines gently as spot rises further above the strike (less
  and less downside protection value left to price).
- **Delta**: 0 at and below the barrier (pure ZCB, no equity sensitivity left). Positive just
  above the barrier (behaves like a put - falls in the index raise the note's value), but turns
  slightly **negative** as spot approaches and passes the strike - the same genuine, well-
  documented feature of knock-out barrier options as the Bullish Sharkfin exhibits (see that
  product's README): being closer to a barrier raises near-term knock-out risk enough to
  sometimes outweigh the change in moneyness.
- **Vega (per 1% change in vol)**: similarly shaped, 0 beyond the barrier.
- **Rho (per 1% change in rates)**: small and negative throughout (the ZCB leg's bond-duration
  effect), barely moving with spot.

## Usage

```
pip install -r ../../../requirements.txt
python3 "Bearish Sharkfin.py"
python3 "Greek Sensitivity.py"
```

Data comes from `yfinance` (Yahoo Finance), falling back to FRED for the S&P 500 and VIX series
if Yahoo is unavailable.
## Scope and limitations

See `Product MtM/README.md` for the shared assumptions (vol proxy, flat rates, credit spread,
dividend treatment, monitoring approximation, numerical limitations) and the project's
AI-assisted learning-project disclosure.
