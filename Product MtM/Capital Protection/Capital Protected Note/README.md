# Capital Protected Note (with Participation) Historical Approximation

Traces the historical mark-to-market of a **Capital Protected Note** along one real historical price path: a bond-like note
that guarantees 100% of principal back at maturity, plus a share of the underlying's upside
above a strike. Defaults to the S&P 500 (`TICKER = "^GSPC"`), but any single-name stock or index
ticker works - see "Underlying selection" below.

This is a **deterministic historical approximation** — real historical prices, no simulation or Monte Carlo
(the same style as the other products in `Product MtM/`).

> **"Model Value"** means this script's own computed value of the product's payoff, as a
> percentage of par, under the assumptions listed below (flat rates, a vol proxy, etc.) — read it
> as "what this simplified model says the payoff is worth today," not a market quote or a claim
> that the product could actually be bought or sold at that level. See `Product MtM/README.md`
> for the full explanation.

The simplest product in `Product MtM/Capital Protection/` — no barrier at all, unlike the
`Bullish Sharkfin`/`Bearish Sharkfin` products in the sibling folders. Read this one first if
you're new to this category; the barrier products add a knock-out feature on top of essentially
the same "ZCB + long option" idea.

## Replication

```
Capital Protected Note = Long Zero-Coupon Bond (pays Principal at maturity)
                        + PARTICIPATION_RATE x Long ATM Call (struck at STRIKE=100% of S0)
```

The **ZCB leg** is what makes this capital-protected: it pays Principal at maturity regardless of
the call, so "if the underlying finishes at or below strike, capital is returned at 100%" falls
straight out of the call being worthless there — no separate logic needed.

The **call quantity is `PARTICIPATION_RATE`, not 1x** — this is the whole point of the product
name. A bank funds the ZCB leg first (it consumes most of the note's par value just to guarantee
getting Principal back), and whatever premium budget is left over buys only a **fraction** of a
full ATM call at realistic market rates/vol/tenor — hence participation is usually **below**
100% (`PARTICIPATION_RATE = 0.80` by default here). It can exceed 100% too — "leveraged" capital
protection — when the ZCB is cheap to fund and/or the call is cheap (low vol, short tenor, high
rates all help). Set by hand here, same as every other manually-set term in this repo (`STRIKE`,
`BARRIER` in the barrier products, etc.) — it isn't derived from anything, it's whatever the
note's actual economics support.

**Terminal payoff**: `Principal + PARTICIPATION_RATE * max(S_T - Strike, 0)`. Unlike the Sharkfin
products, this only depends on where the index **ends up**, not the path it took to get there —
there's no barrier to knock anything out.

## Pricing

Priced entirely via **QuantLib's `AnalyticEuropeanEngine`** (closed-form Black-Scholes-Merton,
dividend yield `q` a native input) — no barrier feature means no CRR lattice is needed anywhere
in this product, unlike the Sharkfin products (see those READMEs, sibling folders, for the CRR
lattice mechanics and the "sawtooth" numerical noise that comes with pricing a barrier that way).
Both legs' Greeks come directly from QuantLib and the ZCB's own closed form — no finite-difference
bump-and-reprice needed anywhere in this file, since there's no barrier discontinuity to work
around.

## Underlying selection: stocks (with dividends) vs. indices (no dividends)

`TICKER` (default `"^GSPC"`) can be set to any Yahoo Finance ticker - the same mechanism used
throughout this repo, no separate flag needed:

- **Set `TICKER` to a Yahoo index ticker** (anything starting with `"^"`, e.g. `"^GSPC"`,
  `"^FTSE"`) to model an index underlying: `fetch_dividend_yield` short-circuits to `q=0`
  immediately.
- **Set `TICKER` to a real stock ticker** (e.g. `"AAPL"`, `"MCD"`) to model a dividend-paying
  single name: `fetch_dividend_yield` fetches a real trailing dividend yield from yfinance and
  feeds it into QuantLib's process as a flat, continuous `q`. It prefers yfinance's
  `trailingAnnualDividendYield` field (already a plain fraction) over the differently-scaled
  `dividendYield` field, which Yahoo has at various times returned as a **percentage** rather
  than a fraction (e.g. `0.33` meaning 0.33%, not 33%) - blindly using that field would silently
  overstate the yield by ~100x. Falls back to `dividendRate / price` if even that field is
  missing, and to `q=0` (with a printed note) if the fetch fails entirely.

`fetch_underlying_name` similarly pulls the company/index display name from yfinance metadata for
every chart title, axis label, and print statement, regardless of which kind of ticker is used.

## Discounting

- **ZCB leg** — discounted at `Principal / (1 + SOFR + issuer credit spread)^T`, using the
  Goldman Sachs 5y CDS spread, same convention as every other principal-at-risk-of-issuer-default
  product in this repo.
- **Call leg** — priced with risk-neutral pricing at **SOFR alone**, no credit spread.

## Product terms

| Term | Default | Meaning |
|---|---|---|
| `STRIKE` | 100% of entry level | Long call strike (at the money) |
| `PARTICIPATION_RATE` | 80% | Quantity of the ATM call - set by hand, not usually 1:1 |
| `RISK_FREE_RATE` | 4% (flat) | SOFR proxy - used for both legs |
| `GS_CDS_SPREAD` | 53.08 bps | Goldman Sachs 5y CDS - issuer credit spread, ZCB leg only |
| `TICKER` | `^GSPC` (S&P 500) | Any Yahoo Finance ticker - index (no dividend) or stock (real dividend yield fetched), drives the display name too |
| `ENTRY_DATE` / `TENOR` | 2025-01-02 / 1 year | The historical window |

Edit the constants at the top of `Capital Protected Note.py` to reprice the note, change the
participation rate, change the underlying, or change the window.

## Output

Running `Capital Protected Note.py` prints the resolved underlying name and dividend yield,
entry/maturity levels, realized returns, the note's model value at inception (as % of par), and
its Greeks — then saves a chart with the underlying's price and the note's Redemption Payoff (Relative to Par)
on the left axis and the note's model value on its own right-hand axis, with a dotted blue line
marking the strike.

Running `Greek Sensitivity.py` prints and charts the Greeks ladder: strike, participation, tenor,
vol and the funding curve held fixed, only spot varies.

## Reading the charts

### `Capital Protected Note.png` (the historical approximation chart)

- **Left axis, firebrick line** — the real underlying's return from entry (%).
- **Left axis, dashed indianred line** — the "Redemption Payoff (Relative to Par)": the terminal payoff formula
  applied to each day's spot, `PARTICIPATION_RATE * max(S - Strike, 0) / S0` (flat at 0% below the
  strike, rising at `PARTICIPATION_RATE`x the index's own pace above it). This is **not** what
  you'd actually receive if the note were sold or unwound on that date - it ignores all remaining
  time value in the still-live call. See the right-axis model-value line for the estimate that accounts for that.
- **Left axis, dotted blue line** — the strike.
- **Right axis, solid darkred line** — the note's model value (% of par), converging onto the
  tracker line exactly at maturity (an option's time value is 0 at expiry, so the two are
  identical by then). Note this line starts **above** par at inception whenever
  `PARTICIPATION_RATE` and the prevailing rate/vol combination make the call worth more than the
  ZCB's own discount - that's the real, unfunded cost of the participation rate at issuance, not
  a bug.

### `Greek Sensitivity.png` (the Greeks ladder)

Four panels, x-axis = **spot as % of S0** (60%–140%). Strike, participation, tenor, vol and the
funding curve are pinned at day-1 values - only spot moves. All four Greeks are **exact
closed-form** here (no finite differences needed, no barrier discontinuity to work around).

- **Price (% of Par)**: smoothly rising, convex above the strike (call optionality), flattening
  toward the ZCB's own discounted value well below it.
- **Delta**: rises smoothly from near 0 (deep out of the money) toward `PARTICIPATION_RATE`
  itself (deep in the money) - the call's own delta approaches 1 there, scaled by the
  participation quantity.
- **Vega (per 1% change in vol)**: always positive (long optionality), peaking near the strike
  where the call has the most time value at risk, fading toward both extremes.
- **Rho (per 1% change in rates)**: negative at low spot (the ZCB leg's bond-duration effect
  dominates when the call is worth little), turning less negative / potentially positive at high
  spot as the call's own positive rho (higher rates raise a call's forward-looking value) starts
  to offset the bond leg.

## Usage

```
pip install -r ../../../requirements.txt
python3 "Capital Protected Note.py"
python3 "Greek Sensitivity.py"
```

Data comes from `yfinance` (Yahoo Finance), falling back to FRED for the S&P 500 and VIX series
if Yahoo is unavailable.
## Scope and limitations

See `Product MtM/README.md` for the shared assumptions (vol proxy, flat rates, credit spread,
dividend treatment, monitoring approximation, numerical limitations) and the project's
AI-assisted learning-project disclosure.
