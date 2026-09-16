# Call Warrant Historical Approximation

Traces the historical mark-to-market of a **Call Warrant** along one real historical price path: a long position in a
standard European vanilla call, nothing more. Defaults to the S&P 500 (`TICKER = "^GSPC"`), but
any single-name stock or index ticker works - see "Underlying selection" below.

This is a **deterministic historical approximation** — real historical prices, no simulation or Monte Carlo
(the same style as the other products in `Product MtM/`).

> **"Model Value"** means this script's own computed value of the product's payoff, as a
> percentage of par, under the assumptions listed below (flat rates, a vol proxy, etc.) — read it
> as "what this simplified model says the payoff is worth today," not a market quote or a claim
> that the product could actually be bought or sold at that level. See `Product MtM/README.md`
> for the full explanation.

The simplest product in this repo. Every other product here is built around a zero-coupon bond
or LEPO leg — some kind of principal/par being returned at maturity. A warrant has none of that:
it's **not a debt instrument**, prepackaged or otherwise. You pay a premium, you own a call, and
if it finishes out of the money you lose the whole premium — there is no bond leg cushioning
that outcome. This is the key structural difference from every other folder in `Product MtM/`,
and it's why this product's pricing/Greeks/chart are presented differently — see "Why this
product isn't expressed as % of par" below.

## Replication

```
Call Warrant = Long European Vanilla Call (struck at STRIKE, typically 100% of S0)
```

That's the entire product. No ZCB, no LEPO, no barrier, no participation multiple, no cap.

**Terminal payoff:** `max(S_T - Strike, 0)`.

## Pricing

Priced entirely via **QuantLib's `AnalyticEuropeanEngine`** (closed-form Black-Scholes-Merton,
dividend yield `q` a native input) — European exercise only, as specified. No barrier feature
means no CRR lattice anywhere in this product, and all Greeks come directly from QuantLib —
no finite-difference bump-and-reprice needed anywhere in this file.

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

There is no ZCB leg and no issuer credit spread anywhere in this product — there's no principal
being lent to an issuer for credit risk to attach to. The call itself is priced with risk-neutral
pricing at `RISK_FREE_RATE` alone, same as the option legs in every other product in this repo.

## Why this product isn't expressed as "% of par"

Every note/certificate product elsewhere in `Product MtM/` reports price and Greeks as a
percentage of par, because those products are built on a bond/LEPO leg with a real principal
amount to normalize against. A warrant has no such leg — the "investment" is the premium paid for
the option, not a par amount, and that premium is usually a small fraction of the underlying's
own price. Forcing a "% of par" framing onto a warrant would be meaningless (% of what?), so this
product instead reports price and Greeks in the units a standard option pricing screen would use:

- **Price** — in the underlying's own price units (index points, or $ for a single stock).
- **Delta** — dimensionless, 0 to 1 for a call.
- **Vega / Rho** — per 1 percentage point change in vol / rates.
- **Theta** — per year (and per day, in the printed summary).

The chart's left axis still shows the underlying's own return in % (matching every other product
in this repo), but the right axis (the warrant's intrinsic value and model value) is in raw
price units on its **own independent scale** — it is deliberately *not* forced to share y-limits
with the left axis, since the warrant's value and the underlying's own price live on genuinely
different scales and unifying them would flatten the warrant line into invisibility.

## Product terms

| Term | Default | Meaning |
|---|---|---|
| `STRIKE` | 100% of entry level | Call strike (at the money) |
| `RISK_FREE_RATE` | 4% (flat) | Used for pricing the call |
| `TICKER` | `^GSPC` (S&P 500) | Any Yahoo Finance ticker - index (no dividend) or stock (real dividend yield fetched), drives the display name too |
| `ENTRY_DATE` / `TENOR` | 2025-01-02 / 1 year | The historical window |

Edit the constants at the top of `Call Warrant.py` to reprice the warrant, change the underlying,
or change the window.

## Output

Running `Call Warrant.py` prints the resolved underlying name and dividend yield, entry/maturity
levels, realized returns, the warrant's premium and Greeks at inception, and its effective
gearing (how much the warrant's value moves, in % of premium paid, for a 1% move in the
underlying) — then saves a chart with the underlying's price return on the left axis and the
warrant's intrinsic value / model value in raw price units on its own right-hand axis, with a
dotted blue line marking the strike.

Running `Greek Sensitivity.py` prints and charts the Greeks ladder: strike, tenor, vol and rate
held fixed, only spot varies.

## Reading the charts

### `Call Warrant.png` (the historical approximation chart)

- **Left axis, firebrick line** — the real underlying's return from entry (%).
- **Left axis, dotted blue line** — the strike, in % terms.
- **Right axis, dashed indianred line** — the warrant's intrinsic value if exercised today,
  `max(S - Strike, 0)`, in raw underlying price units. This is **not** what you'd actually
  receive if the warrant were sold today - it ignores all remaining time value. See the MTM line
  for the model-value estimate.
- **Right axis, solid darkred line** — the warrant's model value (raw price units), converging
  onto the intrinsic value line exactly at maturity.

### `Greek Sensitivity.png` (the Greeks ladder)

Four panels, x-axis = **spot as % of S0** (60%–140%). Strike, tenor, vol and rate are pinned at
day-1 values - only spot moves. All Greeks are exact closed-form (no finite differences needed,
no barrier discontinuity to work around).

- **Price**: smoothly rising and convex above the strike, flattening toward 0 well below it.
- **Delta**: rises smoothly from near 0 (deep out of the money) toward 1 (deep in the money).
- **Vega (per 1% change in vol)**: always positive (long optionality), peaking near the strike.
- **Rho (per 1% change in rates)**: positive throughout (higher rates raise a call's
  forward-looking value), most pronounced deep in the money.

## Usage

```
pip install -r ../../../requirements.txt
python3 "Call Warrant.py"
python3 "Greek Sensitivity.py"
```

Data comes from `yfinance` (Yahoo Finance), falling back to FRED for the S&P 500 and VIX series
if Yahoo is unavailable.
## Scope and limitations

See `Product MtM/README.md` for the shared assumptions (vol proxy, flat rates, credit spread,
dividend treatment, monitoring approximation, numerical limitations) and the project's
AI-assisted learning-project disclosure.
