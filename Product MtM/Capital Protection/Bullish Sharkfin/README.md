# Bullish Sharkfin Historical Approximation

Traces the historical mark-to-market of a **Bullish Sharkfin** along one real historical price path: a capital-protected
note that gives full 1:1 upside participation above a strike, as long as the index never trades
up to a barrier. Touch the barrier once and the participation is knocked out for good - the
note then simply returns principal at maturity. Defaults to the S&P 500 (`TICKER = "^GSPC"`),
but any single-name stock or index ticker works - see "Underlying selection" below.

This is a **deterministic historical approximation** — real historical prices, no simulation or Monte Carlo
(the same style as the other products in `Product MtM/`).

> **"Model Value"** means this script's own computed value of the product's payoff, as a
> percentage of par, under the assumptions listed below (flat rates, a vol proxy, etc.) — read it
> as "what this simplified model says the payoff is worth today," not a market quote or a claim
> that the product could actually be bought or sold at that level. See `Product MtM/README.md`
> for the full explanation.

This is the first product in `Product MtM/Capital Protection/` — distinct from the `Yield/`
products (Reverse Convertible, Fixed Coupon Note, Barrier Reverse Convertible, Discount
Certificate), which are all short an option leg to finance an enhanced coupon at the cost of
some or all capital protection, and from the `Participation/` products (Outperformance,
Bonus-Outperfomance, Bonus Certificate), which give up full principal protection in exchange for
richer participation. A Bullish Sharkfin keeps the principal protection unconditional and pays
for its upside optionality with a barrier cutoff instead.

## Replication

```
Bullish Sharkfin = Long Zero-Coupon Bond (pays Principal at maturity)
                  + Long Up-and-Out Call (struck at STRIKE=100% of S0,
                    barrier H, e.g. 120% of S0, checked once per trading day)
```

The **ZCB leg** is what makes this capital-protected: it pays Principal at maturity regardless
of anything the option leg does. "If the underlying finishes below strike, capital is returned
at 100%" falls straight out of the call being worthless there — no separate logic is needed for
that case, it's automatic.

The **up-and-out call** is what gives the "shark fin" its shape: as long as the index never
trades up to the barrier, the note gets full, uncapped 1:1 participation in the upside above the
strike (unlike the capped products in `Participation/`, there is no cap on the call itself, only
the barrier's all-or-nothing cutoff). Touch the barrier once and the call is knocked out for
good - the note then just pays back principal at maturity, plus whatever `REBATE` was already
received at the moment of the knock-out (not part of the note's forward MTM from that point on,
since it has already been paid out).

This is the **opposite financing story** to every other barrier product in this repo (Barrier
Reverse Convertible, Bonus Certificate, Bonus-Outperformance): there the option leg is **short**
and subtracted, financing an enhanced coupon at the cost of capital protection; here the option
leg is **long** and added on top of a fully protected bond, financed instead by accepting a
barrier cutoff on how far the participation can run.

**Terminal payoff:**
- **Barrier never touched:** `Principal + max(S_T - Strike, 0)` — full uncapped upside
  participation.
- **Barrier touched at some point:** `Principal` (+ whatever rebate was paid at the touch, already
  received) — capital returned at 100%, no further participation, regardless of where the index
  ends up afterward.

## Pricing the embedded call

Priced on a **Cox-Ross-Rubinstein binomial lattice** (QuantLib `BinomialCRRBarrierEngine`),
barrier checked once per lattice step, step count = remaining trading days — not the textbook
continuous-monitoring closed form, which overstates the touch probability relative to what daily
data can ever confirm. See the `Barrier Reverse Convertible` product's README
(`Product MtM/Yield/Barrier Reverse Convertible/`) for the full rationale and lattice mechanics;
the same reasoning applies here unchanged, just mirrored onto an upper barrier (`ql.Barrier.UpOut`)
instead of a lower one.

**Sanity check** (`verify_against_closed_form` in `Bullish Sharkfin.py`): with the barrier pushed
unreachable and the rebate forced to 0, the up-and-out call can never knock out, so its CRR price
should converge onto the plain vanilla call struck at `STRIKE` - confirms the barrier engine
collapses to the ordinary closed form in the no-barrier limit.

### A numerical note on the Greeks: lattice "sawtooth" noise

A CRR lattice is rebuilt from scratch on every pricing call here, and its node grid scales
*multiplicatively* with both spot and volatility (`u = e^(sigma*sqrt(dt))`). As those inputs vary
continuously, which lattice nodes fall above or below the *fixed* strike/barrier changes in
discrete jumps rather than smoothly - a well-documented artifact of naive binomial barrier
pricing (see e.g. Boyle & Lau, 1994). Concretely, during development a finite-difference Delta
computed with a 0.1%-of-spot bump (the convention used for the closed-form, non-barrier products
in this repo) came out to **-0.17** at inception; scanning across bump sizes showed the real
(bump-size-independent) value is closer to **-0.02** — a small bump had landed inside one of
these jumps. Vega was worse: it ranged from **-517 to -2508** depending on bump size before
settling near **-700**. Rho was unaffected (the risk-free rate doesn't enter the lattice's node
spacing at all, only drift and discounting).

The fix used here: `finite_difference_greeks_at` bumps spot by **2% of S0** and vol by **2
percentage points** (both far wider than the 0.1%/0.01-point bumps used elsewhere in this repo)
- wide enough to average across several lattice "teeth" instead of landing inside one. This
trades a small amount of truncation error for a much larger reduction in lattice noise, the
standard practitioner tradeoff for hedging barrier options priced on a binomial tree. Rho keeps
a small (0.0001) bump since it isn't affected. The resulting Greeks ladder is visibly smooth (see
`Greek Sensitivity.png`) with no discontinuous jumps between adjacent spot scenarios.

This same underlying issue (small bumps on a rebuilt-from-scratch CRR lattice) likely affects the
Greeks in the other CRR-barrier products in this repo (`Barrier Reverse Convertible`,
`Bonus-Outperfomance`, `Bonus Certificate`, all of which use smaller bump sizes) to some degree -
it was only caught here because of the fine-grained spot scan done while building this product.

## Underlying selection: ticker, dividends, and labels

`TICKER` (default `"^GSPC"`) can be set to any Yahoo Finance ticker - an index or a single-name
stock. Two things automatically follow from whatever `TICKER` is set to, no other constants need
touching:

- **Dividend yield** (`fetch_dividend_yield`): a flat, continuous yield `q`, fed into QuantLib's
  process for the up-and-out call - the same simplification level as the flat `RISK_FREE_RATE`.
  Indices (any `"^"`-prefixed ticker) are treated as paying `q=0`. The fetch prefers yfinance's
  `trailingAnnualDividendYield` field (already a plain fraction) over the differently-scaled
  `dividendYield` field, which Yahoo has at various times returned as a **percentage** rather
  than a fraction (e.g. `0.33` meaning 0.33%, not 33%) - blindly using that field would silently
  overstate the yield by ~100x. Falls back to `dividendRate / price` if even that field is
  missing, and to `q=0` (with a printed note) if the fetch fails entirely.
- **Display name** (`fetch_underlying_name`): pulls the company/index name from yfinance metadata
  (`shortName`/`longName`) for every chart title, axis label, and print statement - falls back to
  the raw ticker symbol if the metadata fetch fails.

## Discounting: two different rates for two different risks

- **ZCB leg** — discounted at `Principal / (1 + SOFR + issuer credit spread)^T`. This is where
  the investor is exposed to the **issuer's own default risk** (a Bullish Sharkfin is unsecured
  debt of whichever bank issues it), so the discount rate includes that bank's credit spread on
  top of the risk-free rate. Uses the Goldman Sachs 5y CDS spread, same as every other
  principal-at-risk-of-issuer-default product in this repo.
- **Call leg** — priced with risk-neutral pricing at **SOFR alone**, no credit spread. A bank
  prices and hedges the option on standard derivative-pricing terms, not its own funding curve.

## Product terms

| Term | Default | Meaning |
|---|---|---|
| `STRIKE` | 100% of entry level | Long call strike (at the money) |
| `BARRIER` | 120% of entry level | Up-and-out barrier (H) for the embedded call, checked once per trading day (CRR lattice) |
| `REBATE` | 0.0 | Cash paid immediately if the barrier is touched, in underlying currency (not a fraction of S0) - set by hand, not economically important for MtM purposes |
| `RISK_FREE_RATE` | 4% (flat) | SOFR proxy - used for both legs |
| `GS_CDS_SPREAD` | 53.08 bps | Goldman Sachs 5y CDS - issuer credit spread, ZCB leg only |
| `TICKER` | `^GSPC` (S&P 500) | Any Yahoo Finance ticker - drives the dividend yield and display name automatically |
| `ENTRY_DATE` / `TENOR` | 2025-01-02 / 1 year | The historical window |

Edit the constants at the top of `Bullish Sharkfin.py` to reprice the note, change the
underlying, or change the window.

## How the daily MTM tracks the barrier

`bullish_sharkfin_mtm_price_series` and `bullish_sharkfin_running_return` track
`path.cummax() >= barrier_level` day by day — note this is a running **maximum** compared against
an **upper** barrier, the mirror image of the down-barrier products elsewhere in this repo, which
track a running minimum against a lower barrier. Conducted on close (matching the CRR lattice's
own once-per-trading-day observation frequency): once the running maximum of the close rises to
or through the barrier, the call leg is priced at exactly 0 for every subsequent date (a
knock-out, once triggered, stays triggered — "sticky"). `bullish_sharkfin_mtm_price_series` also
passes the actual number of remaining trading days as the CRR lattice's step count for each date.

## Output

Running `Bullish Sharkfin.py` prints the resolved underlying name and dividend yield,
entry/maturity levels, realized returns, whether the barrier was actually touched in this
historical path, the closed-form verification check, the note's model value at inception (as % of
par), and its Greeks — then saves a chart with the underlying's price and note payoff on the left
axis and the note's model value on its own right-hand axis, with dotted lines marking the strike
(blue) and barrier (green), and — if the barrier was touched in this path — a vertical dashed
green line at the knock-out date.

Running `Greek Sensitivity.py` prints and charts the Greeks ladder: strike, barrier, tenor, vol
and the funding curve held fixed, only spot varies, so each Greek's value at a given spot level
tells you directly how much MTM moves for a 1-unit change in that variable, right now, at that
spot.

## Reading the charts

### `Bullish Sharkfin.png` (the historical approximation chart)

- **Left axis, firebrick line** — the real underlying's return from entry (%).
- **Left axis, dashed indianred line** — the "Redemption Payoff (Relative to Par)": the terminal payoff formula
  applied to each day's spot (flat at 0% below the strike, tracking the index 1:1 above it, as
  long as never breached). This is **not** what you'd actually receive if the note were sold or
  unwound on that date - it ignores all remaining time value in the still-live call. It's the
  closest thing to a running "how much upside has this path locked in so far" readout, not a
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
reprice on the full note price, with wider bumps than usual for spot and vol - see the numerical
note above), since barrier-option Greeks have no simple closed form and are genuinely
discontinuous right at the barrier.

- **Price (% of Par)**: the classic "shark fin" shape - rises with spot, peaks somewhere below
  the barrier (the point where increasing knock-out risk starts to outweigh increasing
  moneyness), then declines toward the barrier and flattens exactly at the pure-ZCB value for
  every spot at or beyond it.
- **Delta**: positive at low spot (behaves like a call), but turns **negative** as spot
  approaches the barrier - a genuine, well-documented feature of up-and-out options, not a bug:
  moving closer to the barrier raises the near-term knock-out probability enough to outweigh the
  extra intrinsic value from being more in the money. Exactly 0 at and beyond the barrier (the
  note is pure ZCB there, no equity sensitivity left at all).
- **Vega (per 1% change in vol)**: similarly hump-shaped then negative approaching the barrier -
  higher vol raises the value of the call while spot is far from the barrier, but raises the
  knock-out probability enough to hurt once spot is close to it. Exactly 0 beyond the barrier.
- **Rho (per 1% change in rates)**: small and negative throughout (the ZCB leg's bond-duration
  effect), barely moving with spot since the call leg's own rho is small relative to the bond
  leg's.

## Usage

```
pip install -r ../../../requirements.txt
python3 "Bullish Sharkfin.py"
python3 "Greek Sensitivity.py"
```

Data comes from `yfinance` (Yahoo Finance), falling back to FRED for the S&P 500 and VIX series
if Yahoo is unavailable.
## Scope and limitations

See `Product MtM/README.md` for the shared assumptions (vol proxy, flat rates, credit spread,
dividend treatment, monitoring approximation, numerical limitations) and the project's
AI-assisted learning-project disclosure.
