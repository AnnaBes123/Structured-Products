# Bonus Outperformance Certificate Historical Approximation

Traces the historical mark-to-market of a **bonus outperformance certificate** along one real historical price path: an
outperformance certificate (leveraged participation above a strike) with an embedded **long
down-and-out put** that provides capital protection below the strike, as long as the index
never trades down to a barrier level. Defaults to the S&P 500 (`TICKER = "^GSPC"`), but any
single-name stock or index ticker works - see "Underlying selection" below.

This is a **deterministic historical approximation** — real historical prices, no simulation or Monte Carlo (the
same style as the plain `Outperformance` certificate's historical approximation in the sibling folder).

## Structure

Take the outperformance certificate replication (1× zero-strike call + (participation − 1)×
ATM call, struck at K = S0) and add:

```
+ 1 × long down-and-out put, strike K = S0, barrier H < K, checked once per trading day
```

**Payoff at maturity** (in price terms):

- **Never breached, S_T ≤ K:** certificate pays S_T + (K − S_T) = K = S0 → **0% return**. The
  put exactly cancels the certificate's 1:1 downside — full capital protection, regardless of
  how far the index fell, as long as it never touched the barrier.
- **Never breached, S_T > K:** put is worthless (out of the money) → same leveraged upside as
  the plain outperformance certificate.
- **Breached at any point during the life:** the put is knocked out for good. From then on the
  certificate behaves exactly like a plain outperformance certificate — 1:1 downside below K,
  leveraged participation above it — regardless of where the index ends up.

## Product terms

| Term | Default | Meaning |
|---|---|---|
| `STRIKE` | 100% of entry level | Certificate strike (K) |
| `OUTPERFORMANCE_PARTICIPATION` | 200% | Participation above strike |
| `BARRIER` | 75% of entry level | Down-and-out barrier (H) for the embedded put, checked once per trading day (CRR lattice) |
| `RISK_FREE_RATE` | 4% (flat) | Used only in the option valuation |
| `TICKER` | `^GSPC` (S&P 500) | Any Yahoo Finance ticker - drives the dividend yield and display name automatically |

## Pricing the embedded put — derivation

An earlier version of this script priced the down-and-out put with the textbook closed form (the
reflection principle / method of images), which is only correct under **true continuous
monitoring**. That assumption doesn't match how the certificate is actually ever observed (once
per trading day, via a daily close), and continuous monitoring systematically overstates the
touch probability relative to discrete daily monitoring. This script now prices the put on a
**Cox-Ross-Rubinstein (CRR) binomial lattice** instead, via QuantLib (`BinomialCRRBarrierEngine`,
`down_and_out_put_crr`), with the barrier checked once per lattice step and the step count set to
the actual number of remaining trading days - so the model's monitoring frequency matches the
frequency at which the certificate's price could ever actually be observed, by construction. See
the `Barrier Reverse Convertible` product's README (`Product MtM/Yield/Barrier Reverse Convertible/`) for the
full rationale and lattice mechanics - the same reasoning applies here unchanged.

**Validation.** An earlier hand-derivation of the closed-form reflection-principle formula was
first found to be wrong (it priced the barrier put as negative for several barrier levels) before
being re-derived and checked against a Brownian-bridge continuity-corrected Monte Carlo - a
useful reminder that barrier-option algebra is easy to get subtly wrong, and part of why this
script now leans on QuantLib (an independently-tested, industry-standard library) rather than
further hand-rolled formulas, including for the dividend extension below.

**Greeks** are computed by **finite-difference bump-and-reprice** (central differences on the
full certificate price) rather than by differentiating the barrier pricing further —
barrier-option Greeks are notoriously messy near the barrier (delta in particular can be
discontinuous), and a numerical bump is simpler and more robust for a client-facing explainer.
The CRR step count is fixed once per Greek evaluation and reused across every bumped price, so a
differing step count between bumps never injects lattice-discreteness noise into the Greek.

The Delta/Vega bump sizes (2% of spot / 2 vol points) are much wider than the "0.1% of spot"
convention used for the closed-form products in this repo. A CRR lattice is rebuilt from scratch
on every pricing call, and its node grid scales multiplicatively with both spot and vol, so a
small bump can land entirely inside a discrete "sawtooth" lattice artifact and return a Greek off
by a large factor - Vega here ranged from roughly 1030 to 1580 depending on bump size before
settling near 1550-1580 past a 2-vol-point bump. See the `Bullish Sharkfin` product's README
(`Product MtM/Capital Protection/`) for the specific bump-size scan that surfaced this.

## Underlying selection: ticker, dividends, and labels

`TICKER` (default `"^GSPC"`) can be set to any Yahoo Finance ticker - an index or a single-name
stock. Two things automatically follow from whatever `TICKER` is set to, no other constants need
touching:

- **Dividend yield** (`fetch_dividend_yield`): a flat, continuous yield `q`, fed into the LEPO
  leg's closed form and into QuantLib's process for both the ATM call and the down-and-out put -
  the same simplification level as the flat `RISK_FREE_RATE`. Indices (any `"^"`-prefixed
  ticker) are treated as paying `q=0`. The fetch prefers yfinance's `trailingAnnualDividendYield`
  field (already a plain fraction) over the differently-scaled `dividendYield` field, which Yahoo
  has at various times returned as a **percentage** rather than a fraction (e.g. `0.33` meaning
  0.33%, not 33%) - blindly using that field would silently overstate the yield by ~100x. Falls
  back to `dividendRate / price` if even that field is missing, and to `q=0` (with a printed
  note) if the fetch fails entirely.
- **Display name** (`fetch_underlying_name`): pulls the company/index name from yfinance metadata
  (`shortName`/`longName`) for every chart title, axis label, and print statement that used to
  hardcode "S&P 500" - falls back to the raw ticker symbol if the metadata fetch fails.

## How the daily MTM tracks the barrier

`bonus_outperformance_certificate_mtm_price_series` tracks `path.cummin() <= barrier` day by
day: once the running minimum of the historical path drops to or through the barrier, the put
leg is priced at exactly 0 for every subsequent date (a knock-out, once triggered, stays
triggered) — the same "sticky" barrier logic used in the settle-today payoff calculation. This
also passes the actual number of remaining trading days as the CRR lattice's step count for each
date, so the forward-looking pricing matches the barrier's real observation frequency.

## Output

Running the script prints the resolved underlying name and dividend yield, the same summary as
the plain Outperformance historical approximation (entry/maturity levels, return, certificate return, realized
vol, max drawdown, barrier level), the certificate's fair value at inception, and its Greeks —
then saves a chart with the underlying's price and certificate payoff on the left axis and the
certificate's fair value (as % of par) on its own right-hand axis, with a dotted grey line
marking the barrier level. Every label (legend, axis, chart title) uses the resolved underlying
name, not a hardcoded "S&P 500".

## Reading the charts

### `Bonus-Outperfomance Certificate.png` (the historical approximation chart)

- **Left axis, firebrick line** — the real underlying's return from entry (%).
- **Left axis, dashed indianred line** — the certificate's payoff *if it settled today* (its
  terminal formula applied to today's level, ignoring time value). This line kinks at the
  strike (leveraged above it, 1:1 below) and would jump if the barrier were ever touched — the
  dotted blue line marks the knock-in barrier level.
- **Right axis, solid darkred line** — the certificate's actual fair value (% of
  par) each day, including time value. It converges onto the dashed line exactly at maturity
  (an option's time value is 0 at expiry), and sits *above or below* it beforehand depending on
  how much time value the embedded put still carries.
- The Greeks box (top right) reports Delta/Vega/Rho/Theta **at inception only** (day 1) — it
  does not update day by day. Use the Greeks ladder chart below for how these change with spot.

### `Greek Sensitivity.png` (the Greeks ladder)

Four panels, all sharing the same x-axis: **spot, as % of S0**, from 60% to 140%. Tenor, strike,
barrier, vol and the risk-free rate are all pinned at their day-1 values for every point on every
panel — spot is the only thing that moves. Each panel is plotted **twice** — a solid line for
"Not Breached" (barrier never touched) and a dashed line for "Breached" (barrier already
touched) — because the barrier changes every Greek's value meaningfully; below the barrier
itself only "Breached" is shown, since you cannot be below the barrier without having touched it.

- **Price (% of Par)**: the certificate's fair value at that spot level. Reading a point: "if
  spot were at 80% of S0 today (and never breached), the certificate would be worth X% of par."
- **Delta**: how many *points* of certificate value move for a 1-point move in the index, right
  now, at that spot. A Delta of 1.5 at spot=120% means a further 1-point rise in the index adds
  about 1.5 points to the certificate's value there (leveraged participation kicking in).
- **Vega (per 1% change in vol)**: how many *percentage points of par* the certificate's value
  moves for a 1-percentage-point move in implied vol (e.g. 18% → 19%), at that spot. Vega is
  **not** held at some fixed number here — it's recomputed at every spot, and typically peaks
  near the strike/barrier (where optionality is most sensitive to vol) and fades toward the
  edges. Note this is *not* the same 1% as a 1% relative change - it is +1 full volatility point.
- **Rho (per 1% change in rates)**: how many percentage points of par the certificate's value
  moves for a 1-percentage-point move in the risk-free rate (e.g. SOFR 4% → 5%), at that spot.
  Worked example: a reading of **-0.008 at spot=100%** means "if SOFR rose from 4% to 5% right
  now, with the index unchanged at its entry level, the certificate's fair value would fall by
  about 0.8 percentage points of par" — e.g. roughly $8 on a $1,000-par certificate. It's small
  and can be either sign here because the certificate nets a *long* call-heavy position (positive
  rho, since higher rates raise a call's forward value) against a *long* put (negative rho) -
  which one dominates depends on spot.

## Usage

```
pip install -r ../../../requirements.txt
python3 "Bonus-Outperfomance Certificate.py"
```

Data comes from `yfinance` (Yahoo Finance), falling back to FRED for the S&P 500 and VIX series
if Yahoo is unavailable.
