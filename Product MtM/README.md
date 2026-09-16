# Product MtM

Deterministic, historical-path "mark-to-market" illustrations for a set of structured products
(capital protection, leverage, participation, and yield/income structures). Each product folder
fetches one real historical price path and a small set of real market inputs (spot, dividend
yield, an implied- or realized-vol proxy), then computes a model value of that product's payoff
along the path using closed-form option pricing (mostly QuantLib), a CRR binomial lattice for the
barrier product, or Monte Carlo for the autocallable products.

This is a personal, AI-assisted learning project about derivatives/structured-product valuation,
not a production pricing library and not investment research. See **Scope and limitations**
below before reading any number here as more than an illustration.

## What "Model Value" means here

Every product prints and charts a **"Model Value" line** — the script's own computed value of the
product's payoff, as a percentage of par, under the assumptions listed in that product's README
(flat rates, a vol proxy, etc.). Read it as *"what this simplified model says the payoff is worth
today,"* not as an executable market quote, a broker's indicative price, or a claim that the
product could actually be bought or sold at that level. No product here is calibrated to an
actual issuer term sheet or broker quote.

**For the coupon-bearing Yield products** — Fixed Coupon Note (plain and Autocallable, including
Multi-FCN), Reverse Convertible (including Multi-RC), and Barrier Reverse Convertible — the Model
Value is explicitly a **model value of the redemption component only, and excludes coupons**. It
covers the principal repayment and the embedded downside/autocall feature that's actually coded
up; it does **not** include the value of the periodic coupon cash flows a real note like this
would also pay. Nowhere in this repo should that redemption-component number be read as "total
investor return" or "full note fair value" — see each of those products' README for the specific
qualification. The other Yield products (Discount Certificate, Dual Currency Investment) have no
omitted coupon leg — their entire economics are already captured by the modeled structure.

## Two different "return" concepts — don't conflate them

Every script also distinguishes two things that are easy to mix up:

- **Model value relative to par** — the "Model Value" line/print described above: what the
  modeled payoff is worth today, as a fraction of the note's original par/principal amount.
- **Investment return relative to a purchase price** — what an investor who actually paid some
  price for the product would have made or lost. Par and purchase price coincide for most
  products here (issued/bought at 100% of par), but not always (e.g. a Discount Certificate is
  bought below par by construction; a warrant's "Warrant Return (on premium paid)" is measured
  against the premium paid, not against the underlying's par level).

The **"Redemption Payoff (Relative to Par)"** line/tracker on every chart is a third, related but
distinct thing: it's the terminal payoff *formula* applied to today's spot, as if today were
maturity. It is an **illustration of the payoff mechanics, not an amount locked in or realizable
by selling the note today** — it ignores every bit of remaining time value in the note's embedded
options, which the Model Value line (see above) does account for. Where a product has a barrier or
knock-in/knock-out feature, this tracker correctly uses the barrier's full historical state (e.g.
"has the barrier been touched at any point up to today"), not just today's spot level in isolation.

## Autocall settlement

For the autocallable products (Fixed Coupon Note (Autocallable), Multi-FCN (Autocallable)), once
the note **actually autocalls** in the historical path being charted, the live Model Value series
**ends at the settlement date** — there is no "live note" left to mark after that. If the chart
still shows something past that date on the right axis, it is a separately labeled, dotted
**"Cash Proceeds After Autocall"** series: the par amount received at settlement, held flat with
no reinvestment assumed, excluding coupons. That series is cash-in-hand accounting, not the
continuing mark-to-market of an outstanding instrument.

## Scope and limitations

- **`ENTRY_DATE` + `TENOR` must fall entirely within already-completed market history.** These
  scripts backtest a note that has already matured, not a live, in-progress one — a script fetches
  real prices from `ENTRY_DATE` through `ENTRY_DATE + TENOR` and treats the *last fetched date* as
  maturity. If that window runs past today, the fetch silently returns a truncated path and the
  script would otherwise (mis)treat that truncated last date as the real maturity, corrupting
  every Greek and the whole valuation curve without any visible error. Every main product script
  now checks for this and raises a clear `RuntimeError` instead of proceeding — if you hit it, pick
  an `ENTRY_DATE`/`TENOR` combination that ends on or before today. Each product's `Greek
  Sensitivity.py` companion does not have this check: it only ever reads the entry level
  (`path.iloc[0]`) off its own path fetch, never the truncated tail as a maturity date, so a
  truncated window there doesn't corrupt anything and the guard would be a no-op.
- **Volatility.** For an index ticker (e.g. `^GSPC`), scripts interpolate an implied-vol proxy
  from the VIX / VIX3M / VIX6M term structure — a reasonable proxy for that index's own implied
  vol. **For a single-name stock, there's no free historical implied-vol source**, so scripts
  instead use that ticker's own **trailing 2-year realized (historical) volatility**, held flat
  for the note's life. This is real and ticker-specific, unlike the SPX-proxy approach an earlier
  version of this repo used for single names — but it's still backward-looking (realized), not
  forward-looking (implied): it won't price in anticipated events (e.g. an upcoming earnings
  date) the way a real option-implied vol would, and it can differ substantially from what that
  stock's actual listed options are pricing at any given moment. The Dual Currency Investment
  and Multi-FCN/Multi-RC products use the same realized-vol approach (on FX and per-name
  equity history respectively) for the identical reason.
- **Rates.** The risk-free rate (SOFR proxy) is a single flat, hand-set number for the life of
  each trade, not a fetched or bootstrapped curve.
- **Credit spread.** The issuer credit spread is one real, sourced data point (a Goldman Sachs 5y
  CDS level) used as an illustrative stand-in for "some investment-grade bank's funding spread,"
  not a spread specific to whichever underlying or currency a given script is configured with.
- **Dividends.** Dividend yield is a flat, continuous-yield approximation (trailing yield from
  `yfinance`, or 0 for an index), not a real discrete-dividend schedule.
- **Monitoring/discreteness.** Barrier and autocall features are checked once per trading day
  (matching daily-close data), not continuously — deliberately different from, and more realistic
  than, the textbook continuous-monitoring closed-form barrier formula, but still an
  approximation of a real note's actual observation convention (which is set by its term sheet).
- **Numerical methods.** Monte Carlo prices (the autocallable and multi-asset products) carry
  simulation noise from a fixed seed and path count; CRR lattice prices (the barrier products)
  carry discretization error from a finite step count. A "closed-form benchmark" here means a
  plain-vanilla option price from QuantLib's exact analytic (Black-Scholes-Merton) engine — no
  lattice, no simulation, no discretization error, just the textbook formula. Each numerical
  script includes its own `verify_against_closed_form` check that engineers a special case where
  its numerical engine is mathematically supposed to collapse onto that exact benchmark (push a
  barrier to an unreachable level so it can never knock in/out, or reduce a multi-asset basket
  down to a single name QuantLib has an exact engine for) and confirms the numbers actually match.
  A pass doesn't validate the barrier/basket case, only that the numerical wiring itself is
  correct there; each product's own README documents the specific benchmark and collapse it uses.
  Printed at runtime as a PASS/FAIL line with a relative difference — read it before trusting a
  given run's output.
- **What using QuantLib (or a real historical spot path) does and doesn't prove.** QuantLib
  supplies correct, industry-standard option-pricing *mechanics* (Black-Scholes-Merton, CRR
  lattices, basket Monte Carlo), and the backtests replay real historical closing prices. Neither
  of those facts validates the *inputs* (flat rates, proxy vols, a single illustrative credit
  spread) against an actual market price at any given date. "Built on QuantLib" and "priced off
  real historical data" are not the same claim as "matches what this note would actually have
  traded at."
- **No coupon pricing anywhere in this repo.** As noted above, the coupon-bearing Yield products
  model only the redemption component. Coupon valuation (discounting the periodic cash flows,
  handling day counts/accrual, etc.) is a deliberately separate piece of work, not started here.

## About this project

This is an AI-assisted learning project: the product selection, structure, and direction were
mine, and [Claude Code](https://claude.com/claude-code) generated the implementation under my
direction. I'm still learning coupon valuation specifically, which is why it's out of scope here
rather than approximated. Treat this as a work in progress under review, not a
validated or independently-checked pricing tool — if you spot a modeling issue beyond the
presentation fixes described above, please treat it as a separate, open question rather than an
already-resolved one.

## Layout

- `Capital Protection/` — Bearish/Bullish Sharkfin, Capital Protected Note
- `Leverage/` — Call Warrant, Put Warrant
- `Participation/` — Bonus Certificate, Bonus-Outperformance Certificate, Outperformance
  Certificate, Twin-Win Certificate
- `Yield/` — Barrier Reverse Convertible, Discount Certificate, Dual Currency Investment, Fixed
  Coupon Note (plain, Autocallable, Multi-FCN), Reverse Convertible (plain, Multi-RC)
- `Price Return Basic/`, `Graphs (Maths)/` — supporting reference material, not product valuations

Each product folder has its own README with the specific replication, assumptions, and chart
guide for that product. Code comments are kept short by design - the six Yield products with
non-trivial derivations (Fixed Coupon Note/Multi-FCN, Reverse Convertible's Multi-RC, Barrier
Reverse Convertible, Discount Certificate, Dual Currency Investment) each have a `MATHEMATICS.md`
alongside their `README.md` with every formula worked out in full; sibling products with
identical math (the plain Reverse Convertible, the barrier products in Capital Protection/
Participation) point back to those rather than duplicating them.
