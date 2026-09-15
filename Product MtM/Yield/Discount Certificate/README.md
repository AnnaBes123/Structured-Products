# Discount Certificate Historical Approximation

Traces the historical mark-to-market of a **Discount Certificate** along one real historical price path: you get full
downside participation in the underlying but capped upside, in exchange for buying it today at
a discount to spot. Defaults to the S&P 500 (`TICKER = "^GSPC"`), but any single-name stock or
index ticker works - see "Underlying selection" below.

This is a **deterministic historical approximation** — real historical prices, no simulation or Monte Carlo
(the same style as the other products in `Product MtM/`).

## Replication

```
Discount Certificate = LEPO (Low Exercise Price Option, struck at 0)
                      - Short Call (struck at CAP, solved from DISCOUNT)
```

`CAP` is the only real strike in this product - `DISCOUNT` is not a strike, it's the resulting
fair-value-vs-par gap that `CAP` (plus vol/rate/tenor) produces. In practice an issuer works
backward from a target discount to whichever `CAP` funds it, so that's the direction this script
solves in too: `DISCOUNT` (e.g. `5%`) is the hand-set dial, and `solve_cap_for_discount` finds
the `CAP` that makes the certificate's day-1 fair value equal exactly `S0*(1-DISCOUNT)`, via
root-finding (`scipy.optimize.brentq` - fair value is monotonically increasing in `CAP`, so
there's exactly one solution). This mirrors real issuance (you're quoted a discount, not a cap)
rather than the "set `CAP`, observe whatever discount falls out" flow used before.

A **LEPO** is a call struck at ~0 — always exercised, so its price is the risk-neutral PV of
receiving one share at maturity, `S*e^(-qT)` (see `lepo_price_and_greeks`). This equals spot
exactly only when the underlying pays **no dividend** (`q=0`, the default for an index like
`^GSPC`) - for a real dividend-paying stock, owning the LEPO is economically identical to owning
the underlying via a forward purchase, not outright, since the option holder only receives the
share at maturity and so forgoes every dividend paid before then. Priced directly via its exact
closed form rather than through the general option engine, since `K=0` is a numerically awkward
input for one (`d1`/`d2` blow up as `K→0`).

The **short call** is priced via QuantLib (`AnalyticEuropeanEngine`) and subtracted. Selling it
caps the upside at `CAP`, and the premium received is exactly what funds buying the LEPO leg at
a discount to today's spot - that's the whole economic story of a "discount certificate." If the
underlying pays dividends, part of the "discount to spot" now reflects the dividend drag on the
LEPO leg as well as the short-call premium - both are real, theoretically-consistent components
of why the certificate is cheaper than spot, not just the option premium alone.

**Payoff at maturity**: `S_T - max(S_T - Cap, 0) = min(S_T, Cap)` — full downside exposure
(no floor, unlike the Bonus-Outperformance or Fixed Coupon Note products), capped upside above
the cap level.

## Underlying selection: ticker, dividends, and labels

`TICKER` (default `"^GSPC"`) can be set to any Yahoo Finance ticker - an index or a single-name
stock. Two things automatically follow from whatever `TICKER` is set to, no other constants need
touching:

- **Dividend yield** (`fetch_dividend_yield`): a flat, continuous yield `q`, fed into both the
  LEPO leg's closed form and QuantLib's process for the short call - the same simplification
  level as the flat `RISK_FREE_RATE`. Indices (any `"^"`-prefixed ticker) are treated as paying
  `q=0`. The fetch prefers yfinance's `trailingAnnualDividendYield` field (already a plain
  fraction) over the differently-scaled `dividendYield` field, which Yahoo has at various times
  returned as a **percentage** rather than a fraction (e.g. `0.33` meaning 0.33%, not 33%) -
  blindly using that field would silently overstate the yield by ~100x. Falls back to
  `dividendRate / price` if even that field is missing, and to `q=0` (with a printed note) if the
  fetch fails entirely.
- **Display name** (`fetch_underlying_name`): pulls the company/index name from yfinance metadata
  (`shortName`/`longName`) for every chart title, axis label, and print statement that used to
  hardcode "S&P 500" - falls back to the raw ticker symbol if the metadata fetch fails.

## Product terms

| Term | Default | Meaning |
|---|---|---|
| `DISCOUNT` | 5% | Target discount to spot at inception - the hand-set dial; `CAP` is solved from this |
| `CAP` | *(resolved)* | Short call strike / upside cap - derived at runtime by `solve_cap_for_discount`, not set by hand |
| `RISK_FREE_RATE` | 4% (flat) | Used for the option valuation |
| `TICKER` | `^GSPC` (S&P 500) | Any Yahoo Finance ticker - drives the dividend yield and display name automatically |
| `ENTRY_DATE` / `TENOR` | 2025-01-02 / 1 year | The historical window |

Edit `DISCOUNT` (not `CAP`) at the top of `Discount Certificate.py` to reprice the certificate -
`CAP` is recalculated from it every run. Change `TICKER`/`ENTRY_DATE`/`TENOR` to change the
underlying or window.

## Greeks

Both legs are plain vanilla calls (LEPO priced via its exact closed form, short call via
QuantLib), so all four Greeks are **closed-form** (no finite differences needed, same as the
plain Outperformance certificate):

- **Delta** = `LEPO_delta - ShortCall_delta` - `LEPO_delta` is `e^(-qT)` (exactly 1.0 only when
  q=0), and the short call's delta eats into it further as spot rises toward and past the cap.
- **Vega** = `-ShortCall_vega` - always negative (short volatility), since the LEPO has no
  vega at all.
- **Theta** = `LEPO_theta - ShortCall_theta` - the short call decays in the position's favor as
  time passes (same mechanism as the Fixed Coupon Note's short put), and for a dividend payer the
  LEPO leg's own theta is also positive (`q*S*e^(-qT)`, since less dividend drag remains to be
  missed as maturity approaches) - both push value up over time.

## Volatility

Same approach as the other products in this repo: SPX implied vol interpolated from the
VIX / VIX3M / VIX6M term structure for each day's actual remaining time-to-maturity. See the
`Product MtM/Participation/Outperformance/` folder's README for the full rationale.

## Output

Running the script fetches the historical path and vol surface, **solves for `CAP`** from the
`DISCOUNT` target (printed as its own step), then prints the resolved underlying name and
dividend yield, entry/maturity levels, realized returns, the certificate's fair value at
inception (should land within solver tolerance of `100% - DISCOUNT`), and its Greeks - then saves
a chart with the underlying's price and certificate payoff on the left axis (note the payoff line
visibly flattens once spot crosses the cap) and the certificate's fair value on its own
right-hand axis. Every label (legend, axis, chart title) uses the resolved underlying name, not a
hardcoded "S&P 500".

## Reading the charts

### `Discount Certificate.png` (the historical approximation chart)

- **Left axis, firebrick line** — the real underlying's return from entry (%).
- **Left axis, dashed indianred line** — the "Participation Tracker": the terminal payoff formula
  applied to today's level - tracks the index 1:1, then visibly flattens once spot crosses the
  cap (the short call capping the upside).
- **Right axis, solid darkred line** — the certificate's actual Black-Scholes fair value (% of
  par) each day, converging onto the dashed line exactly at maturity.
- The Greeks box (top right) is the day-1 snapshot only - see the ladder chart below for how
  Greeks move with spot.

### `Greek Sensitivity.png` (the Greeks ladder)

Four panels, one shared x-axis: **spot, as % of S0**, from 60% to 140%. `CAP` is resolved once
from `DISCOUNT` at day-1 vol (same as the main script) and then held fixed, along with tenor, vol
and rate - only spot varies. Theta is excluded since tenor never moves across the ladder, so
there's nothing meaningful for a "time passing" Greek to show. Both legs here are plain vanilla
calls, so every Greek is exact closed-form, not finite difference.

- **Price (% of Par)**: the certificate's fair value at that spot level - note it can exceed
  100% below the cap (you're holding the index outright there) and flattens above it.
- **Delta**: points of certificate value gained/lost per 1-point index move, right now, at that
  spot. Starts near 1.0 at low spot (behaves like the index itself) and falls toward 0 well
  above the cap (the short call has essentially cancelled the LEPO's own delta by then).
- **Vega (per 1% change in vol)**: percentage points of par moved per 1-percentage-point move in
  implied vol. **Not a fixed number** - recomputed at every spot, always negative here (short
  the call = short volatility), largest in magnitude near the cap and fading away from it.
- **Rho (per 1% change in rates)**: percentage points of par moved per 1-percentage-point move
  in the risk-free rate. Worked example: a reading of **≈ -0.0037 at spot=100%** means "if rates
  rose from 4% to 5% right now, index unchanged, the certificate's fair value would fall by about
  0.37 percentage points of par" - roughly $3.70 on a $1,000-par certificate. Small because the
  LEPO leg (long call struck at 0, rho ≈ 0 since it's always exercised - no time-value optionality
  left to be rate-sensitive) and the short call leg's rho largely offset each other.

## Usage

```
pip install -r ../../../requirements.txt
python3 "Discount Certificate.py"
```

For the Greeks ladder (spot varies, tenor/cap/vol/rate held fixed - see the `Outperformance`
folder's README for what this shows and why Theta is excluded from it):

```
python3 "Greek Sensitivity.py"
```

Data comes from `yfinance` (Yahoo Finance), falling back to FRED for the S&P 500 and VIX
series if Yahoo is unavailable.
