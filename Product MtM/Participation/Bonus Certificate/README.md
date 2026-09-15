# Bonus Certificate Historical Approximation

Traces the historical mark-to-market of a **Bonus Certificate** along one real historical price path: 1:1 exposure to the
underlying, with a guaranteed minimum ("bonus") redemption level as long as the index never
trades down to a barrier — uncapped on the upside. Defaults to the S&P 500 (`TICKER = "^GSPC"`),
but any single-name stock or index ticker works - see "Underlying selection" below.

This is a **deterministic historical approximation** — real historical prices, no simulation or Monte Carlo
(the same style as the other products in `Product MtM/`).

Distinct from the `Bonus-Outperfomance` certificate in the sibling folder: that product adds
*leveraged* (>1x) participation above its strike on top of the same bonus/barrier mechanic. This
one has plain 1:1 participation throughout, unlimited on the upside — the classic, uncapped Bonus
Certificate. (A short call could be added to fund a richer bonus level in exchange for capping the
upside - that would make this a separate, explicitly "Capped Bonus Certificate" variant, not built
here.)

## Replication

```
Bonus Certificate = Long LEPO (zero-strike call, = holding the underlying via a forward purchase)
                   + Long Down-and-Out Put (struck at BONUS_LEVEL, e.g. 100% of S0,
                     barrier H, e.g. 80% of S0, checked once per trading day)
```

`STRIKE` (100% of S0, the initial fixing/reference level) is not itself a leg of the replication -
the down-and-out put's actual strike is `BONUS_LEVEL`, not `STRIKE`. It exists so `BARRIER` and
`BONUS_LEVEL` can be described and charted relative to a fixed 100% reference, the same way every
other "% of S0" term in this repo is quoted. A genuine bonus certificate typically sets
`BONUS_LEVEL` *above* `STRIKE` (e.g. 110%) - you get back more than the initial reference level
even if the index finishes flat or down, as long as the barrier is never touched; the default
here (`BONUS_LEVEL = 100%`) is the boundary case where the "bonus" is exactly capital protection,
no more.

The **LEPO leg** alone tracks the underlying 1:1, uncapped — no leverage, unlike the
Bonus-Outperformance certificate's extra participation leg.

The **down-and-out put** is what creates the "bonus": as long as the barrier is never touched,
it pays `BONUS_LEVEL - S_T` whenever `S_T < BONUS_LEVEL`, topping the LEPO's value up to exactly
the bonus level — a guaranteed minimum redemption regardless of how far the index fell, provided
it never fell as far as the barrier. Touch the barrier once and the put is knocked out for good —
the **conditional protection is permanently destroyed**, it never comes back even if the index
later recovers above the barrier — and the certificate then behaves like ordinary, uncapped equity
exposure (the LEPO alone), no floor at all but still no cap.

**Terminal payoff:**
- **Barrier never touched:** `max(S_T, BONUS_LEVEL)` — the certificate can't be worth less than
  the bonus level, only more, unlimited.
- **Barrier touched at some point:** `S_T` — the floor is gone for good; ordinary uncapped equity
  exposure with no downside protection at all.

This payoff depends on the **path** the index took (did it ever touch the barrier?), not just
where it ends up.

## Pricing the embedded put

Priced on a **Cox-Ross-Rubinstein binomial lattice** (QuantLib `BinomialCRRBarrierEngine`),
barrier checked once per lattice step, step count = remaining trading days — not the textbook
continuous-monitoring closed form, which overstates the touch probability relative to what daily
data can ever confirm. See the `Barrier Reverse Convertible` product's README
(`Product MtM/Yield/Barrier Reverse Convertible/`) for the full rationale and lattice mechanics; the same reasoning applies
here unchanged. The LEPO leg is priced via its exact closed form.

**Sanity check** (`verify_against_closed_form` in `Bonus Certificate.py`): with the barrier
pushed unreachable (`H → 0`), the down-and-out put can never knock out, so its CRR price should
converge onto the plain vanilla put struck at `BONUS_LEVEL` — confirms the barrier engine
collapses to the ordinary closed form in the no-barrier limit.

**Greeks** are computed by **finite-difference bump-and-reprice** (central differences on the
full certificate price) rather than by differentiating the barrier pricing further —
barrier-option Greeks are notoriously messy near the barrier (delta in particular can be
discontinuous). The CRR step count is fixed once per Greek evaluation and reused across every
bumped price, so a differing step count between bumps never injects lattice-discreteness noise
into the Greek.

The Delta/Vega bump sizes (2% of spot / 2 vol points) are also much wider than the "0.1% of
spot" convention used for the closed-form products in this repo, for the same reason: a CRR
lattice's node grid scales multiplicatively with both spot and vol, so a small bump can land
entirely inside a discrete "sawtooth" lattice artifact and return a Greek off by a large factor.
See the `Bullish Sharkfin` product's README (`Product MtM/Capital Protection/`) for the specific
bump-size scan that surfaced this.

## How the daily MTM tracks the barrier

`bonus_certificate_mtm_price_series` and `bonus_certificate_running_return` track
`path.cummin() <= barrier_level` day by day (conducted on close, matching the CRR lattice's own
once-per-trading-day observation frequency): once the running minimum of the close drops to or
through the barrier, the put leg is priced at exactly 0 for every subsequent date (a knock-out,
once triggered, stays triggered — "sticky", the conditional protection does not come back even if
the index recovers). `bonus_certificate_mtm_price_series` also passes the actual number of
remaining trading days as the CRR lattice's step count for each date.

## Underlying selection: ticker, dividends, and labels

`TICKER` (default `"^GSPC"`) can be set to any Yahoo Finance ticker - an index or a single-name
stock. Two things automatically follow from whatever `TICKER` is set to, no other constants need
touching:

- **Dividend yield** (`fetch_dividend_yield`): a flat, continuous yield `q`, fed into both legs
  (the LEPO leg's closed form and the down-and-out put's CRR lattice) - the same simplification
  level as the flat `RISK_FREE_RATE`. Indices (any `"^"`-prefixed ticker) are treated as paying
  `q=0`. The fetch prefers yfinance's `trailingAnnualDividendYield` field (already a plain
  fraction) over the differently-scaled `dividendYield` field, which Yahoo has at various times
  returned as a **percentage** rather than a fraction (e.g. `0.33` meaning 0.33%, not 33%) -
  blindly using that field would silently overstate the yield by ~100x. Falls back to
  `dividendRate / price` if even that field is missing, and to `q=0` (with a printed note) if the
  fetch fails entirely.
- **Display name** (`fetch_underlying_name`): pulls the company/index name from yfinance metadata
  (`shortName`/`longName`) for every chart title, axis label, and print statement - falls back to
  the raw ticker symbol if the metadata fetch fails.

## Discounting

Unlike the bond-like products in this repo (Reverse Convertible, Fixed Coupon Note, Barrier
Reverse Convertible), a Bonus Certificate has **no separate principal/ZCB leg** — it isn't debt
that returns principal at maturity, it's a pure combination of equity-linked legs (LEPO, put),
same family as the plain `Outperformance`, `Bonus-Outperfomance`, and `Discount
Certificate` products. Every leg is priced at `RISK_FREE_RATE` alone; there is no issuer credit
spread to apply, since there is no bond leg for issuer default risk to attach to.

## Product terms

| Term | Default | Meaning |
|---|---|---|
| `STRIKE` | 100% of entry level | Initial fixing / reference level - doesn't enter the payoff directly, just the anchor `BARRIER`/`BONUS_LEVEL` are quoted against |
| `BONUS_LEVEL` | 100% of entry level | Down-and-out put strike / guaranteed minimum redemption if never breached |
| `BARRIER` | 80% of entry level | Down-and-out barrier (H) for the embedded put, checked once per trading day (CRR lattice) |
| `RISK_FREE_RATE` | 4% (flat) | Used for every leg |
| `TICKER` | `^GSPC` (S&P 500) | Any Yahoo Finance ticker - drives the dividend yield and display name automatically |
| `ENTRY_DATE` / `TENOR` | 2025-01-02 / 1 year | The historical window |

Edit the constants at the top of `Bonus Certificate.py` to reprice the certificate, change the
underlying, or change the window.

## Volatility

Same approach as the other products in this repo: SPX implied vol interpolated from the
VIX / VIX3M / VIX6M term structure for each day's actual remaining time-to-maturity. See the
`Outperformance` folder's README for the full rationale.

## Output

Running `Bonus Certificate.py` prints the resolved underlying name and dividend yield,
entry/maturity levels, realized returns, whether the barrier was actually touched in this
historical path, the closed-form verification check, the certificate's fair value at inception
(as % of par), and its Greeks — then saves a chart with the underlying's price and certificate
payoff on the left axis and the certificate's fair value on its own right-hand axis, with dotted
lines marking the strike (grey), bonus level (blue), and barrier (green), and — if the barrier was
touched in this path — a vertical dashed green line at the knock-out date.

Running `Greek Sensitivity.py` prints and charts the Greeks ladder: bonus level, barrier, tenor,
vol and the risk-free rate held fixed, only spot varies, shown for both barrier states
(Not Breached / Breached) where both are physically reachable.

## Reading the charts

### `Bonus Certificate.png` (the historical approximation chart)

- **Left axis, firebrick line** — the real underlying's return from entry (%).
- **Left axis, dashed indianred line** — the certificate's payoff if settled today: flat at the
  bonus level below it (as long as never breached), tracking 1:1 above it, unlimited.
- **Left axis, dotted lines** — blue marks the bonus level, green marks the barrier; a dashed
  green vertical line (with a "Barrier Knocked Out" label) marks the date the barrier was actually
  touched, if it was, in this historical path.
- **Right axis, solid darkred line** — the certificate's fair value (% of par), converging onto
  the dashed payoff line exactly at maturity.

### `Greek Sensitivity.png` (the Greeks ladder)

Four panels, x-axis = **spot as % of S0** (60%–140%). Bonus level, barrier, tenor, vol and
rate are pinned at day-1 values - only spot moves. Each panel is plotted twice — solid for "Not
Breached", dashed for "Breached" — since the barrier changes every Greek's value meaningfully;
below the barrier itself only "Breached" is shown, since you cannot be below the barrier without
having touched it.

- **Price (% of Par)**: fair value at that spot level. Visibly flattens toward the bonus level at
  low spot when not breached; keeps rising 1:1 at high spot, uncapped.
- **Delta**: points of certificate value per 1-point index move, right now, at that spot. Stays
  near 1 at high spot (the LEPO's own delta, uncapped) and rises above 1 near the bonus level
  while not breached (the still-live put adds extra sensitivity there).
- **Vega (per 1% change in vol)**: percentage points of par per 1-percentage-point vol move.
  **Not a fixed number** - recomputed at every spot; a vol increase raises the chance of ever
  touching the barrier (bad for the floor) but also raises the value of the still-live put while
  not breached.
- **Rho (per 1% change in rates)**: percentage points of par per 1-percentage-point rate move.

## Usage

```
pip install -r ../../../requirements.txt
python3 "Bonus Certificate.py"
python3 "Greek Sensitivity.py"
```

Data comes from `yfinance` (Yahoo Finance), falling back to FRED for the S&P 500 and VIX series
if Yahoo is unavailable.
