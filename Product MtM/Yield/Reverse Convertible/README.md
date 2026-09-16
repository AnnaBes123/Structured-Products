# Reverse Convertible Historical Approximation

Traces the historical mark-to-market of a **Reverse Convertible** along one real historical price path: a bond-like note
that pays an enhanced coupon in exchange for taking on the risk of physical delivery of the
underlying if it finishes below a strike level at maturity. Defaults to the S&P 500
(`TICKER = "^GSPC"`), but any single-name stock or index ticker works - see "Underlying
selection" below.

This is a **deterministic historical approximation** — real historical prices, no simulation or Monte Carlo
(the same style as the other products in `Product MtM/`).

Same replication and terms as the `Fixed Coupon Note` product in the sibling folder — a reverse
convertible and a fixed coupon note are the same underlying structure in practice, priced here
under identical assumptions. This version has no autocall feature.

> **Scope: model value of the redemption component — excludes coupons.** Everything this script
> computes and charts as "Model Value" is the value of the principal repayment plus the embedded
> downside feature (the short put / physical-delivery risk) - the part of the structure that's
> actually modeled here. It does **not** include the value of the periodic coupon cash flows a
> real Reverse Convertible also pays; coupon valuation is out of scope for this project (see
> `Product MtM/README.md`). Don't read the printed/charted value as total investor return or as a
> full note fair value.

## Replication

```
Reverse Convertible = Long Zero-Coupon Bond (pays Principal at maturity)
                     - Short Put (struck at STRIKE, e.g. 90% of S0)
```

The short put's premium is what finances the note's enhanced coupon over a plain bond — that's
the whole economic story of a reverse convertible. Selling the put means the investor keeps the
premium but is on the hook for physical delivery of the underlying if the index finishes below
the strike.

**Terminal payoff**: full principal back if the index finishes at or above the strike; below it,
the investor is delivered `Principal / Strike` shares of the underlying instead of cash — the
defining feature of a reverse convertible versus a plain FCN, which settles the shortfall in
cash. In value terms this is `Principal * min(S_T / Strike, 1)`, which is why the short put is
sized at `1 / Strike` units rather than 1x: a 1x put's payoff (`Principal - max(Strike - S_T,
0)`) only matches the value of `Principal/Strike` delivered shares (`Principal * S_T / Strike`)
at the two endpoints `S_T = 0` and `S_T = Strike` — everywhere strictly between them the two
diverge, and only the `1/Strike` conversion ratio reproduces the delivered-share value exactly
at every point below the strike. This only depends on where the index **ends up**, not the path
it took to get there (European put, no barrier) — the historical path shows this clearly: the 2025
window has a -18.9% max drawdown that breaches the -10% strike intraperiod, but since the index
recovers and finishes +16.65%, the terminal note return is 0% (full principal), not a loss.

## Discounting: two different rates for two different risks

- **ZCB leg** — discounted at `Principal / (1 + SOFR + issuer credit spread)^T`. This is where
  the investor is exposed to the **issuer's own default risk** (a reverse convertible is
  unsecured debt of whichever bank issues it), so the discount rate includes that bank's credit
  spread on top of the risk-free rate.
- **Put leg** — priced via **QuantLib** (`AnalyticEuropeanEngine`, closed-form
  Black-Scholes-Merton) at **SOFR alone**, no credit spread. A bank prices and hedges the option
  on standard derivative-pricing terms, not its own funding curve — the credit-risk premium
  belongs entirely to the bond leg, not the option leg.

## Underlying selection: ticker, dividends, and labels

`TICKER` (default `"^GSPC"`) can be set to any Yahoo Finance ticker - an index or a single-name
stock. Two things automatically follow from whatever `TICKER` is set to, no other constants need
touching:

- **Dividend yield** (`fetch_dividend_yield`): a flat, continuous yield `q`, fetched from
  `yfinance` and fed into QuantLib's process as a native input - the same simplification level as
  the flat `RISK_FREE_RATE`. Indices (any `"^"`-prefixed ticker, e.g. `^GSPC`) are treated as
  paying `q=0`. For a real stock, ignoring dividends would misprice the put leg by roughly `q*T`
  and make the MTM theoretically inconsistent with how the underlying actually trades (a
  dividend-paying stock's forward price is below its spot by the dividend drag, which both the
  put's value and its Greeks need to reflect). The fetch prefers yfinance's
  `trailingAnnualDividendYield` field (already a plain fraction) over the differently-scaled
  `dividendYield` field, which Yahoo has at various times returned as a **percentage** rather than
  a fraction (e.g. `0.33` meaning 0.33%, not 33%) - blindly using that field would silently
  overstate the yield by ~100x. Falls back to `dividendRate / price` if even that field is
  missing, and to `q=0` (with a printed note) if the fetch fails entirely.
- **Display name** (`fetch_underlying_name`): pulls the company/index name from yfinance metadata
  (`shortName`/`longName`) for every chart title, axis label, and print statement that used to
  hardcode "S&P 500" - falls back to the raw ticker symbol if the metadata fetch fails.

## Product terms

| Term | Default | Meaning |
|---|---|---|
| `STRIKE` | 90% of entry level | Short put strike / conversion level |
| `RISK_FREE_RATE` | 4% (flat) | SOFR proxy - used for both legs |
| `GS_CDS_SPREAD` | 53.08 bps | Goldman Sachs 5y CDS - issuer credit spread, ZCB leg only |
| `TICKER` | `^GSPC` (S&P 500) | Any Yahoo Finance ticker - drives the dividend yield and display name automatically |
| `ENTRY_DATE` / `TENOR` | 2025-01-02 / 1 year | The historical window |

Edit the constants at the top of `Reverse Convertible.py` to reprice the note, change the
underlying, or change the window.

## Greeks

All four are closed-form (verified against finite differences before shipping):

- **Delta** comes entirely from the short put (`-(1/Strike) * put_delta`) - the ZCB has no
  equity sensitivity at all. Being short a put means positive delta (long-like exposure).
- **Vega** is `-(1/Strike) * put_vega` - short volatility, since the ZCB has no vega either.
- **Rho** combines the ZCB's bond-duration sensitivity to SOFR with the put's own rho.
- **Theta** is positive by construction: the bond "pulls to par" as time passes, and the short
  put decays in the position's favor - both effects push value up over time, which is the whole
  point of an income-generating note like this.

## Volatility

Same approach as the other products in this repo: for an index, SPX implied vol interpolated
from the VIX / VIX3M / VIX6M term structure for each day's actual remaining time-to-maturity
(not a flat number, and not the raw 30-day VIX applied to a much longer holding period); for a
single-name stock, that ticker's own trailing 2-year realized volatility (held flat, since no
free historical implied-vol source exists for an arbitrary stock the way VIX serves the index).
See the `Product MtM/Participation/Outperformance/` folder's README for the full rationale on
both, and the known limitation beyond 182
days (VIX6M held flat, since there's no free historical source for a longer-dated implied vol).

## Output

Running the script prints the resolved underlying name and dividend yield, entry/maturity
levels, realized returns, the note's model value of the redemption component at inception (as %
of par, excluding coupons - note this is typically **below** par, since the ZCB alone doesn't earn
back its own discount without the put premium topping it up, and there is no coupon leg here to
top it up further), and its Greeks - then saves a chart with the underlying's price and note
payoff on the left axis and the model value on its own right-hand axis, with a dotted line marking
the strike. Every label (legend, axis, chart title) uses the resolved underlying name, not a
hardcoded "S&P 500".

## Reading the charts

### `Reverse Convertible.png` (the historical approximation chart)

- **Left axis, firebrick line** — the real underlying's return from entry (%).
- **Left axis, dashed indianred line** — the "Redemption Payoff (Relative to Par)": the terminal payoff formula
  applied to today's level (ignoring time value): flat at 0% above the strike, then falling 1:1
  with the index below it. This is **not** what you'd actually receive if the note were sold or
  unwound today - it ignores all remaining time value in the still-live put. The dotted blue line
  marks the strike.
- **Right axis, solid darkred line** — the model value of the redemption component (% of par,
  excludes coupons, Black-Scholes) each day, including time value. It converges onto the tracker
  line exactly at maturity.
- The Greeks box (top right) is the day-1 (inception) snapshot only — for how Greeks change with
  spot, see the ladder chart below.

### `Greek Sensitivity.png` (the Greeks ladder)

Four panels sharing one x-axis: **spot, as % of S0**, from 60% to 140%. Tenor, strike, the
funding curve (SOFR + CDS spread) and vol are all pinned at their day-1 values throughout - only
spot moves. Theta is excluded here on purpose: since tenor never varies across the ladder, a
"time passing" Greek has nothing meaningful to show against a fixed T.

- **Price (% of Par)**: the model value of the redemption component at that spot level.
- **Delta**: points of note value gained/lost per 1-point move in the index, right now, at that
  spot. Always between 0 and ~1 here, since being short a put gives long-like (positive) delta
  that grows as spot falls toward the strike.
- **Vega (per 1% change in vol)**: percentage points of par the note's value moves for a
  1-percentage-point move in implied vol (e.g. 18% → 19%). This is **not a fixed number** - it's
  recomputed at every spot level, and it's always negative here (short volatility: selling the
  put means a vol *increase* hurts you), largest in magnitude near the strike where the put has
  the most optionality, and fading toward zero far from it.
- **Rho (per 1% change in SOFR)**: percentage points of par the note's value moves for a
  1-percentage-point move in the risk-free rate. Worked example: a reading of **≈ -0.0069 at
  spot=100%** means "if SOFR rose from 4% to 5% right now, index unchanged, the note's model value
  would fall by about 0.69 percentage points of par" - roughly $6.90 on a $1,000-par note. It's
  small and negative because the ZCB leg's bond-duration effect (higher rates → lower present
  value of the principal) outweighs the smaller, opposite-signed rho contributed by the short
  put.

## Usage

```
pip install -r ../../../requirements.txt
python3 "Reverse Convertible.py"
```

For the Greeks ladder (spot varies, tenor/strike/funding curve/vol held fixed - see the
`Product MtM/Participation/Outperformance/` folder's README for what this shows and why Theta is excluded from it):

```
python3 "Greek Sensitivity.py"
```

Data comes from `yfinance` (Yahoo Finance), falling back to FRED for the S&P 500 and VIX
series if Yahoo is unavailable.

## Scope and limitations

See `Product MtM/README.md` for the shared assumptions (vol proxy, flat rates, credit spread,
dividend treatment, numerical limitations) and the project's AI-assisted learning-project
disclosure. This product in particular excludes coupon valuation entirely - see the scope note at
the top of this file.

## The maths

This note's replication and pricing math is identical to the Fixed Coupon Note's (sibling
folder) - see `../Fixed Coupon Note/MATHEMATICS.md` for every formula worked out, including the
conversion-ratio derivation.
