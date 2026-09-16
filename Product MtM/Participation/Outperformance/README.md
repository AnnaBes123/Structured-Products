# Outperformance Certificate Historical Approximation

Traces the historical mark-to-market of an **outperformance certificate** along one real historical price path: an
investor holds the underlying below a strike, and gets a leveraged (>100%) participation in
gains above that strike. Defaults to the S&P 500 (`TICKER = "^GSPC"`), but any single-name stock
or index ticker works - see "Underlying selection" below. It compares two views of the
certificate's value over the year:

1. **Redemption Payoff (Relative to Par)** — the certificate's payoff formula applied to each
   day's index level, ignoring any time value (as if the certificate matured on that day).
2. **Model Value** — the certificate's actual value each day, priced as an option portfolio via
   QuantLib, including time value.

This is a **deterministic historical approximation** — real historical prices, no simulation or Monte Carlo.

> **"Model Value"** means this script's own computed value of the product's payoff, as a
> percentage of par, under the assumptions listed below (flat rates, a vol proxy, etc.) — read it
> as "what this simplified model says the payoff is worth today," not a market quote or a claim
> that the product could actually be bought or sold at that level. See `Product MtM/README.md`
> for the full explanation.

## Product terms

| Term | Default | Meaning |
|---|---|---|
| `STRIKE` | 100% of entry level | Below strike, return = index return (1:1) |
| `OUTPERFORMANCE_PARTICIPATION` | 200% | Above strike, return = participation × index return |
| `RISK_FREE_RATE` | 4% (flat) | Used only in the option valuation |
| `TICKER` | `^GSPC` (S&P 500) | Any Yahoo Finance ticker - drives the dividend yield and display name automatically |
| `ENTRY_DATE` / `TENOR` | 2020-01-02 / 1 year | The historical window |

Edit the constants at the top of `Outperfomance Certificate.py` to reprice the certificate,
change the underlying, or change the window.

## How the valuation works

The certificate is replicated as an option portfolio:

```
1.0 × LEPO (zero-strike call)  (= owning the underlying via a forward purchase)
+ (participation − 1) × ATM call, struck at the entry level
```

- The **LEPO leg** (`lepo_price_and_greeks`) is a call struck at ~0, always exercised, priced
  via its exact closed form as the risk-neutral PV of receiving one share at maturity,
  `S*e^(-qT)` — economically identical to holding the underlying outright only when the
  underlying pays **no dividend** (`q=0`). For a real dividend-paying stock, the LEPO is worth
  less than spot today, since the certificate holder only receives the share at maturity and so
  misses every dividend paid before then.
- The **ATM call** is struck once, at the entry level, and stays there — it's only
  literally "at the money" on day one, then drifts in- or out-of-the-money as the index moves.
  Priced via **QuantLib** (`AnalyticEuropeanEngine`, closed-form Black-Scholes-Merton).
- Both legs are priced daily using the **shrinking time to maturity** and the **current spot**
  on each date.

**Volatility** depends on whether `TICKER` is an index or a single name:

- **Index** (any `"^"`-prefixed ticker, e.g. `^GSPC`) — the SPX implied-vol term structure: VIX
  (30-day), VIX3M (93-day), and VIX6M (182-day), interpolated each day to match that day's actual
  remaining time to maturity (variance-time interpolation). Beyond 182 days, VIX6M is held flat.
  This avoids two mistakes: using a single 30-day vol for a much longer holding period, and using
  the full window's *realized* volatility, which would leak future information (e.g. a crash) into
  valuations made before it happened.
- **Single-name stock** — there's no free historical implied-vol source for an arbitrary stock the
  way VIX serves the index, so this uses that ticker's own **trailing 2-year realized volatility**
  (computed from real daily closes ending at `ENTRY_DATE`, no look-ahead), held flat for the note's
  life — the same technique already used for Multi-FCN/Multi-RC's per-name vols and the DCI's FX
  vol. This is a genuine, ticker-specific number, but it's backward-looking (realized), not
  forward-looking (implied) — it won't reflect anticipated events like an upcoming earnings date
  the way a real option-implied vol would. Earlier versions of this repo used the SPX VIX term
  structure as an illustrative proxy even for single-name tickers; that's no longer the case.

At maturity, the model value converges exactly to the settle-today payoff, since an option's
time value is exactly zero at expiry.

## Underlying selection: ticker, dividends, and labels

`TICKER` (default `"^GSPC"`) can be set to any Yahoo Finance ticker - an index or a single-name
stock. Two things automatically follow from whatever `TICKER` is set to, no other constants need
touching:

- **Dividend yield** (`fetch_dividend_yield`): a flat, continuous yield `q`, fed into both the
  LEPO leg's closed form and QuantLib's process for the ATM call leg - the same simplification
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

## Assumptions and simplifications

- **Dividend yield is fetched automatically** for the selected `TICKER` (0% for an index).
  Before this, the certificate assumed a flat 0% dividend yield regardless of underlying, which
  meant the model value at inception could look *above* par (100%) purely because the LEPO leg
  was treated as literally free to fund - for a real dividend-paying stock that gap is now
  properly split between the genuine, unfunded cost of the participation rate and the LEPO leg's
  own dividend drag.
- **Flat 4% risk-free rate**, not a historical rate curve.
- **No fees.**
- **European exercise**, no early exercise or path-dependent features.

## Output

Running the script prints:
- The resolved underlying name and dividend yield
- Entry/maturity levels, realized return, certificate return, realized volatility, max drawdown
- The certificate's model value at inception (vs. par)
- Greeks at inception: Delta, Vega, Rho, Theta

...and saves `Outperfomance Certificate.png`: the underlying and certificate payoff on the left
axis, and the certificate's model value (as % of par) on its own right-hand axis. Every label
(legend, axis, chart title) uses the resolved underlying name, not a hardcoded "S&P 500".

## Reading the charts

### `Outperfomance Certificate.png` (the historical approximation chart)

- **Left axis, firebrick line** — the real underlying's return from entry (%).
- **Left axis, dashed line** — the certificate's payoff *if it settled today*: tracks the index
  1:1 below the strike, then steepens above it (leveraged participation kicking in).
- **Right axis, solid line** — the certificate's actual model value (% of par), converging onto
  the dashed payoff line exactly at maturity. Indexed to par (S0), not the certificate's own
  day-1 value - this is what makes it converge exactly onto the payoff line's percentage at
  maturity; the tradeoff is the line starts above 0% on day 1, reflecting the real option
  premium the participation rate costs at that vol (see the console printout for that number).
- Greeks reported alongside are the day-1 (inception) snapshot only.

### `Greek Sensitivity.png` (the Greeks ladder)

Four panels sharing one x-axis: **spot, as % of S0**, from 60% to 140%. Tenor and strike are
pinned at their day-1 values, and so are vol and rate - only spot moves across the ladder. Both
legs are plain vanilla calls, so every Greek here is exact closed-form, not finite difference.

- **Price (% of Par)**: the certificate's model value at that spot level - note it can exceed
  100% even below the strike when `q=0` (the default index case), since owning the underlying
  via the LEPO leg then costs nothing extra to fund; for a dividend-paying stock the LEPO leg
  itself trades below spot, pulling this down somewhat.
- **Delta**: points of certificate value gained/lost per 1-point index move, right now, at that
  spot. Starts near 1.0 well below the strike (behaves like the index) and rises toward the
  participation rate (e.g. 2.0) well above it, since the leveraged leg dominates there.
- **Vega (per 1% change in vol)**: percentage points of par moved per 1-percentage-point move in
  implied vol. **Not a fixed number** - recomputed at every spot, positive here (the certificate
  is net long optionality via the leveraged call), peaking near the strike where that call has
  the most time value at risk, and fading away from it.
- **Rho (per 1% change in rates)**: percentage points of par moved per 1-percentage-point move
  in the risk-free rate. Worked example: a reading of **≈ +0.0053 at spot=100%** means "if rates
  rose from 4% to 5% right now, index unchanged, the certificate's model value would rise by about
  0.53 percentage points of par" - roughly $5.30 on a $1,000-par certificate. Positive (unlike
  the reverse-convertible-style products) because this certificate is net **long** calls, and a
  higher risk-free rate raises a call's forward-looking value.

## Usage

```
pip install -r ../../../requirements.txt
python3 "Outperfomance Certificate.py"
python3 "Greek Sensitivity.py"
```

Data comes from `yfinance` (Yahoo Finance), falling back to FRED for the S&P 500 series if
Yahoo is unavailable.
## Scope and limitations

See `Product MtM/README.md` for the shared assumptions (vol proxy, flat rates, credit spread,
dividend treatment, monitoring approximation, numerical limitations) and the project's
AI-assisted learning-project disclosure.
