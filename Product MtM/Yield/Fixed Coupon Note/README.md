# Fixed Coupon Note (FCN) Historical Approximation

Traces the historical mark-to-market of a **Fixed Coupon Note** along one real historical price path: a bond-like
structure that gives back full principal at maturity unless the index has fallen below a
strike level, in which case the shortfall is deducted point-for-point. Defaults to the S&P 500
(`TICKER = "^GSPC"`), but any single-name stock or index ticker works - see "Underlying
selection" below.

This is a **deterministic historical approximation** — real historical prices, no simulation or Monte Carlo
(the same style as the other products in `Product MtM/`).

## Replication

```
Fixed Coupon Note = Long Zero-Coupon Bond (pays Principal at maturity)
                   - Short Put (struck at STRIKE, e.g. 90% of S0)
```

The short put's premium is what finances the note's enhanced yield ("coupon") over a plain
bond — that's the whole economic story of an FCN. Selling the put means the investor keeps the
premium but is on the hook if the index finishes below the strike.

**Terminal payoff**: `Principal - max(Strike - S_T, 0)` — full principal back if the index
finishes at or above the strike; below it, principal is reduced dollar-for-dollar with the
shortfall, exactly like a short put. This only depends on where the index **ends up**, not the
path it took to get there (European put, no barrier) — the historical path shows this clearly: the
2025 window has a -18.9% max drawdown that breaches the -10% strike intraperiod, but since the
index recovers and finishes +16.65%, the terminal note return is 0% (full principal), not a
loss.

## Discounting: two different rates for two different risks

- **ZCB leg** — discounted at `Principal / (1 + SOFR + issuer credit spread)^T`. This is where
  the investor is exposed to the **issuer's own default risk** (an FCN is unsecured debt of
  whichever bank issues it), so the discount rate includes that bank's credit spread on top of
  the risk-free rate.
- **Put leg** — priced via **QuantLib** (`AnalyticEuropeanEngine`, closed-form
  Black-Scholes-Merton) at **SOFR alone**, no credit spread. A bank prices and hedges the option
  on standard derivative-pricing terms, not its own funding curve — the credit-risk premium
  belongs entirely to the bond leg, not the option leg.

## Underlying selection: ticker, dividends, and labels

`TICKER` (default `"^GSPC"`) can be set to any Yahoo Finance ticker - an index or a single-name
stock - across all four scripts in this folder. Two things automatically follow from whatever
`TICKER` is set to, no other constants need touching:

- **Dividend yield** (`fetch_dividend_yield`): a flat, continuous yield `q`, fetched from
  `yfinance` and fed into every pricing engine in this folder as a native input - QuantLib's
  process for the two closed-form FCN scripts, and the risk-neutral GBM drift (`r-q`) for the
  autocallable's Monte Carlo engine and its digital-call closed-form check. Indices (any
  `"^"`-prefixed ticker) are treated as paying `q=0`. The fetch prefers yfinance's
  `trailingAnnualDividendYield` field (already a plain fraction) over the differently-scaled
  `dividendYield` field, which Yahoo has at various times returned as a **percentage** rather
  than a fraction (e.g. `0.33` meaning 0.33%, not 33%) - blindly using that field would silently
  overstate the yield by ~100x. Falls back to `dividendRate / price` if even that field is
  missing, and to `q=0` (with a printed note) if the fetch fails entirely.
- **Display name** (`fetch_underlying_name`): pulls the company/index name from yfinance metadata
  (`shortName`/`longName`) for every chart title, axis label, and print statement that used to
  hardcode "S&P 500" - falls back to the raw ticker symbol if the metadata fetch fails.

## Product terms

| Term | Default | Meaning |
|---|---|---|
| `STRIKE` | 90% of entry level | Short put strike |
| `RISK_FREE_RATE` | 4% (flat) | SOFR proxy - used for both legs |
| `GS_CDS_SPREAD` | 53.08 bps | Goldman Sachs 5y CDS - issuer credit spread, ZCB leg only |
| `TICKER` | `^GSPC` (S&P 500) | Any Yahoo Finance ticker - drives the dividend yield and display name automatically |
| `ENTRY_DATE` / `TENOR` | 2025-01-02 / 1 year | The historical window |

Edit the constants at the top of `Fixed Coupon Note.py` to reprice the note, change the
underlying, or change the window.

## Greeks

All four are closed-form (verified against finite differences before shipping):

- **Delta** comes entirely from the short put (`-put_delta`) - the ZCB has no equity
  sensitivity at all. Being short a put means positive delta (long-like exposure).
- **Vega** is `-put_vega` - short volatility, since the ZCB has no vega either.
- **Rho** combines the ZCB's bond-duration sensitivity to SOFR with the put's own rho.
- **Theta** is positive by construction: the bond "pulls to par" as time passes, and the short
  put decays in the position's favor - both effects push value up over time, which is the whole
  point of an income-generating note like this.

## Volatility

Same approach as the other products in this repo: SPX implied vol interpolated from the
VIX / VIX3M / VIX6M term structure for each day's actual remaining time-to-maturity (not a
flat number, and not the raw 30-day VIX applied to a much longer holding period). See the
`Product MtM/Participation/Outperformance/` folder's README for the full rationale and the known limitation beyond 182
days (VIX6M held flat, since there's no free historical source for a longer-dated implied vol).

## Output

Running the script prints the resolved underlying name and dividend yield, entry/maturity
levels, realized returns, the note's fair value at inception (as % of par - note this is
typically **below** par, since the ZCB alone doesn't earn back its own discount without the put
premium topping it up), and its Greeks - then saves a chart with the underlying's price and note
payoff on the left axis and the note's fair value on its own right-hand axis, with a dotted line
marking the strike. Every label (legend, axis, chart title) uses the resolved underlying name,
not a hardcoded "S&P 500".

## Reading the charts

This folder has two products (plain and autocallable) with two charts each.

### `Fixed Coupon Note.png` (plain FCN historical approximation)

- **Left axis, firebrick line** — the real underlying's return from entry (%).
- **Left axis, dashed indianred line** — the "Participation Tracker": the terminal payoff formula
  applied to each day's spot, flat at 0% above the strike, falling 1:1 with the index below it
  (dotted blue line marks the strike). This is **not** what you'd actually receive if the note
  were sold or unwound on that date - it ignores all remaining time value in the still-live put.
  See the right-axis MTM line for the actual fair-value estimate.
- **Right axis, solid darkred line** — the note's actual fair value (% of par), converging onto
  the tracker line exactly at maturity.

### `Greek Sensitivity.png` (plain FCN Greeks ladder)

Four panels, x-axis = **spot as % of S0** (60%–140%). Tenor, strike, funding curve and vol are
pinned at day-1 values throughout - only spot varies; Theta is excluded since tenor never moves.
All four Greeks are exact closed-form here (this note is just ZCB minus a plain vanilla put).

- **Price (% of Par)**: fair value at that spot level.
- **Delta**: points of note value per 1-point index move, right now, at that spot.
- **Vega (per 1% change in vol)**: percentage points of par per 1-percentage-point vol move.
  **Not a fixed number** - recomputed at every spot, always negative (short volatility), largest
  near the strike.
- **Rho (per 1% change in SOFR)**: percentage points of par per 1-percentage-point rate move.
  Worked example: **≈ -0.0069 at spot=100%** means a SOFR move from 4% to 5%, index unchanged,
  costs the note about 0.69 percentage points of par (~$6.90 on $1,000 par) - the ZCB leg's bond
  duration dominates the smaller, opposite-signed rho from the short put.

### `Fixed Coupon Note (Autocallable).png` (autocallable historical approximation)

- **Left axis, firebrick line** — the real underlying's return from entry (%).
- **Left axis, dashed indianred line** — the "Participation Tracker": the terminal payoff formula
  applied to today's level, flat at 0% once the note has actually autocalled in this historical
  path (see the darkgreen "Autocalled" marker).
- **Left axis, mediumseagreen dotted vertical lines** — every quarterly observation date,
  whether or not it triggered the autocall; a darkgreen dashed line marks the one that actually
  did (if any).
- **Right axis, solid darkred line** — the note's Monte Carlo fair value (% of par); this is
  simulated, not closed-form, since the autocall payoff depends on multiple discrete future
  dates jointly (see the file's own comments on why no simple Black-Scholes formula applies).

### `Greek Sensitivity (Autocallable).png` (autocallable Greeks ladder)

Same 60%–140% spot ladder, but all four Greeks are **Monte Carlo finite differences** (bump and
reprice with common random numbers, since there's no closed form for the autocall feature) -
Theta is excluded here too, for the same fixed-tenor reason.

- **Price / Delta / Vega / Rho**: read the same way as the plain note's ladder above.
- Worked example: **≈ -0.0024 at spot=100%** for Rho - smaller in magnitude than the plain
  note's -0.0069, because the autocall feature typically shortens the note's *expected* life
  (it can redeem early at the 100% trigger), which shrinks the ZCB leg's effective duration and
  therefore its rate sensitivity.

## Usage

```
pip install -r ../../../requirements.txt
python3 "Fixed Coupon Note.py"
python3 "Greek Sensitivity.py"
python3 "Fixed Coupon Note (Autocallable).py"
python3 "Greek Sensitivity (Autocallable).py"
```

Data comes from `yfinance` (Yahoo Finance), falling back to FRED for the S&P 500 and VIX
series if Yahoo is unavailable.
