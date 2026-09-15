# Twin-Win Certificate Historical Approximation

Traces the historical mark-to-market of a **Twin-Win Certificate** along one real historical price path: a certificate that
profits whichever direction the underlying moves — full 1:1 upside participation above a strike,
and an equal-and-opposite GAIN (not a loss) for a fall below it — as long as the index never
trades down to a barrier. Touch the barrier once and the inversion is gone for good; the
certificate then just tracks the index 1:1, up or down, like a plain tracker. Defaults to the
S&P 500 (`TICKER = "^GSPC"`), but any single-name stock or index ticker works - see "Underlying
selection" below.

This is a **deterministic historical approximation** — real historical prices, no simulation or Monte Carlo
(the same style as the other products in `Product MtM/`).

Structurally close to the `Bonus Certificate` in this same folder (same LEPO + down-and-out put
architecture), but with two differences: the put quantity here is **2x**, not 1x, and there is
**no cap**. Those two changes are what turn a "bonus/floor" payoff into a genuine "twin win"
(inversion) payoff — see the replication section below for the algebra.

## Replication

```
Twin-Win Certificate = Long LEPO (zero-strike call, = holding the underlying via a forward purchase)
                      + 2x Long Down-and-Out Put (struck at STRIKE=100% of S0,
                        barrier H, e.g. 70% of S0, checked once per trading day)
```

The **LEPO leg** alone tracks the underlying 1:1, exactly like the plain `Outperformance`
certificate's zero-strike leg.

The **2x down-and-out puts** are what create the "twin win" inversion. Below the strike
(`K = STRIKE*S0`) and not yet breached, the certificate is worth:

```
LEPO(S) + 2*(K - S) = S + 2K - 2S = 2K - S
```

In return terms, with `K = S0` (the default, at the money): `(2*S0 - S)/S0 - 1 = 1 - S/S0 =
-(S/S0 - 1)` — the exact **negative** of the index's own return. A fall in the index of X%
becomes a **gain** of X% for the certificate, as long as the barrier was never touched. This is
why the put quantity has to be exactly **2x**: 1x would only cancel the LEPO's own downside
(a flat payoff below the strike, no upside from a fall — that's the `Bonus Certificate`'s "floor"
behaviour instead); anything other than 2x leaves either a net residual loss or a leveraged gain
rather than a clean mirror image of the index's move.

Above the strike, both puts are worthless and the certificate is just the LEPO — ordinary 1:1
upside participation.

Touch the barrier once and **both** puts are knocked out for good — the inversion feature is
gone, and the certificate reverts to being just the LEPO for the remainder of its life: ordinary
1:1 index tracking, upside or downside, with no more floor and no more inversion.

**Terminal payoff:**
- **Barrier never touched, `S_T >= Strike`:** `S_T` — ordinary 1:1 upside participation.
- **Barrier never touched, `S_T < Strike`:** `2*Strike - S_T` — the index's own loss, inverted
  into an equal gain.
- **Barrier touched at some point:** `S_T` — plain 1:1 index tracking, whichever direction, no
  inversion left.

## Pricing the embedded puts

Priced on a **Cox-Ross-Rubinstein binomial lattice** (QuantLib `BinomialCRRBarrierEngine`),
barrier checked once per lattice step, step count = remaining trading days — see the `Barrier
Reverse Convertible` product's README (`Product MtM/Yield/`) for the full continuous-vs-discrete-
monitoring rationale, and the `Bullish Sharkfin` product's README
(`Product MtM/Capital Protection/`) for the specific finite-difference bump-size scan that
motivates the wider (2%-of-spot / 2-vol-point) Delta/Vega bumps used here.

**Sanity check** (`verify_against_closed_form`): with the barrier pushed unreachable, the
down-and-out put can never knock out, so its CRR price should converge onto the plain vanilla put
struck at `STRIKE` - confirms the barrier engine collapses to the ordinary closed form in the
no-barrier limit.

**Greeks** are computed by **finite-difference bump-and-reprice** on the full certificate price,
same approach as every other barrier product in this repo — barrier-option Greeks are
notoriously messy near the barrier (delta in particular can be discontinuous, and here it's
amplified further by the 2x put quantity).

## Underlying selection: stocks (with dividends) vs. indices (no dividends)

`TICKER` (default `"^GSPC"`) can be set to any Yahoo Finance ticker - the **same mechanism** used
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

Like the other equity-derivative certificates in this repo (`Outperformance`,
`Bonus-Outperfomance`, `Bonus Certificate`), there is **no separate principal/ZCB leg** — this
isn't debt that returns principal at maturity, it's a pure combination of a LEPO and put legs.
Every leg is priced at `RISK_FREE_RATE` alone; there is no issuer credit spread to apply, since
there is no bond leg for issuer default risk to attach to.

## Product terms

| Term | Default | Meaning |
|---|---|---|
| `STRIKE` | 100% of entry level | LEPO reference / down-and-out put strike (at the money) |
| `BARRIER` | 70% of entry level | Down-and-out barrier (H) for the embedded puts, checked once per trading day (CRR lattice) |
| `PUT_QUANTITY` | 2.0 | Quantity of the down-and-out put - 2x is what makes the inversion exact; set by hand |
| `RISK_FREE_RATE` | 4% (flat) | Used for every leg |
| `TICKER` | `^GSPC` (S&P 500) | Any Yahoo Finance ticker - index (no dividend) or stock (real dividend yield fetched), drives the display name too |
| `ENTRY_DATE` / `TENOR` | 2025-01-02 / 1 year | The historical window |

Edit the constants at the top of `Twin-Win Certificate.py` to reprice the certificate, change
the underlying, or change the window.

## How the daily MTM tracks the barrier

`twin_win_certificate_mtm_price_series` and `twin_win_certificate_running_return` track
`path.cummin() <= barrier_level` day by day (conducted on close, matching the CRR lattice's own
once-per-trading-day observation frequency): once the running minimum of the close drops to or
through the barrier, both puts are priced at exactly 0 for every subsequent date (a knock-out,
once triggered, stays triggered — "sticky"). `twin_win_certificate_mtm_price_series` also passes
the actual number of remaining trading days as the CRR lattice's step count for each date.

## Output

Running `Twin-Win Certificate.py` prints the resolved underlying name and dividend yield,
entry/maturity levels, realized returns, whether the barrier was actually touched in this
historical path, the closed-form verification check, the certificate's fair value at inception
(as % of par), and its Greeks — then saves a chart with the underlying's price and certificate
payoff on the left axis and the certificate's fair value on its own right-hand axis, with dotted
lines marking the strike (blue) and barrier (green), and — if the barrier was touched in this
path — a vertical dashed green line at the knock-out date.

Running `Greek Sensitivity.py` prints and charts the Greeks ladder: strike, barrier, tenor, vol
and the risk-free rate held fixed, only spot varies, shown for both barrier states (Not Breached
/ Breached) where both are physically reachable.

## Reading the charts

### `Twin-Win Certificate.png` (the historical approximation chart)

- **Left axis, firebrick line** — the real underlying's return from entry (%).
- **Left axis, dashed indianred line** — the "Participation Tracker": the terminal payoff formula
  applied to each day's spot. This is **not** what you'd actually receive if the certificate were
  sold or unwound on that date - it ignores all remaining time value in the still-live puts. See
  the right-axis MTM line for the actual fair-value estimate. Above the strike (never breached)
  it tracks the index 1:1; below it (never breached) it mirrors the index's loss into an
  equal-and-opposite gain; once breached, it collapses onto the index's own return line exactly
  (the certificate is just the LEPO at that point).
- **Left axis, dotted lines** — blue marks the strike, green marks the barrier; a dashed green
  vertical line (with a "Barrier Knocked Out" label) marks the date the barrier was actually
  touched, if it was, in this historical path.
- **Right axis, solid darkred line** — the certificate's fair value (% of par), converging onto
  the tracker line exactly at maturity.

### `Greek Sensitivity.png` (the Greeks ladder)

Four panels, x-axis = **spot as % of S0** (60%–140%). Strike, barrier, tenor, vol and rate are
pinned at day-1 values - only spot moves. Each panel is plotted twice — solid for "Not Breached",
dashed for "Breached" — since the barrier changes every Greek's value meaningfully; below the
barrier itself only "Breached" is shown, since you cannot be below the barrier without having
touched it.

- **Price (% of Par)**: when breached, an exact flat line equal to spot itself (the certificate
  is just the LEPO there — a useful sanity check on its own). When not breached, dips lowest
  around the strike (the point where the inversion payoff and the plain upside payoff meet) and
  rises on both sides of it.
- **Delta**: exactly **1.0** when breached (pure LEPO). When not breached, elevated well above 1
  near the barrier (the 2x put quantity amplifies the usual barrier "cliff risk" effect - a small
  further fall meaningfully raises the knock-out probability, which here costs the certificate a
  large chunk of its inverted-gain upside), settling toward more normal levels away from the
  barrier.
- **Vega (per 1% change in vol)**: 0 when breached (LEPO has no vega). Negative near the barrier
  when not breached (higher vol raises the knock-out probability, which is bad news for the
  inversion feature), turning positive further from it (higher vol then just raises the still-live
  puts' or LEPO-adjacent optionality value).
- **Rho (per 1% change in rates)**: 0 when breached (LEPO has no rate exposure - see the README's
  discounting section on why).

## Usage

```
pip install -r ../../../requirements.txt
python3 "Twin-Win Certificate.py"
python3 "Greek Sensitivity.py"
```

Data comes from `yfinance` (Yahoo Finance), falling back to FRED for the S&P 500 and VIX series
if Yahoo is unavailable.
