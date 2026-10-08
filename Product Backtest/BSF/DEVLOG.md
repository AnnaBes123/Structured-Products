# BSF (Bullish Shark Fin) Python-in-Excel — devlog

Chronological decision history for `BSF Python-in-Excel.py`. See `Product Backtest/FCN/DEVLOG.md`,
`Product Backtest/RC/DEVLOG.md`, `Product Backtest/BRC/DEVLOG.md` for the shared Python-in-Excel
conventions this reuses (oversized price range, vertical Params list, `xl()` mechanics, whitespace-
stripped param labels) - this file only covers what's specific to BSF.

## Added (2026-09-22)

A Bullish Shark Fin is principal-protected - no delivery/loss scenario at all, unlike FCN/RC/BRC.
It's an upside participation note capped by an up-and-out knock-out barrier: if the (worst-of)
underlying never touches the barrier, full upside participation is realized at maturity
(PARTICIPATION); if it touches the barrier at any point, participation is knocked out and the
investor just gets principal back, no upside (KNOCKED_OUT). This reports outcome-bucket
percentages only, same as the other three - not the $ payoff itself (participation rate/cap aren't
modeled since they don't change which bucket a launch falls into).

Verified: monotonic sanity check on AMD (barrier further from spot -> lower knock-out rate: 80.4%
-> 59.5% -> 51.0% for 110%/130%/150%, completed cohort), and PARTICIPATION + KNOCKED_OUT +
OUTSTANDING sums to 100% in every case (only two resolved buckets, same two-outcome shape as RC).

## Barrier monitoring toggle + STRIKE added (2026-09-22)

Reversed the "continuous-only, no toggle" call from the entry above - added the same European/
American `Barrier Monitoring` toggle as BRC (European: barrier checked only at maturity; American:
checked every trading day from launch through maturity). Also added STRIKE: once barrier-touched,
STRIKE is irrelevant (the knock-out rebate doesn't depend on it) - so the PARTICIPATION bucket
splits into PARTICIPATION_ITM (never knocked out, finishes >= STRIKE - the participation payoff is
actually positive) and PARTICIPATION_OTM (never knocked out, but finishes < STRIKE - technically
"not knocked out" but the participation payoff itself is zero, same as flat principal back).
Params now match BRC's exactly in shape (`Ticker`/`Strike`/`Barrier`/`Barrier Monitoring`/
`Tenor Months`/`Launch Start`/`Launch End`/`Max Gap Days`, `D1:E8`).

Verified three ways on AMD: (1) buckets still sum to 100%; (2) American shows more knock-outs than
European at the same Strike/Barrier (59.5% vs 42.2% completed) - same monitoring-mode sanity
direction as BRC; (3) raising STRIKE from 1.00 to 1.20 (same Barrier/monitoring) shifted mass from
ITM (14.0% -> 4.1%) to OTM (39.8% -> 49.6%) while Knocked Out stayed EXACTLY unchanged at 42.235% -
confirms STRIKE only splits the non-knocked-out pool and has no effect on the knock-out test itself,
which is the whole point of testing it independently rather than assuming it from the code.

## Standalone Python version added, `BSF.py` (2026-09-29)

Added `BSF.py` alongside (not replacing) `BSF Python-in-Excel.py`: same rule, fetched from yfinance
instead of a FactSet sheet. It is a ~40-line file of terms + `classify` + `self_test()`, running on the
shared engine `../_backtest.py` (see `Product Backtest/FCN/DEVLOG.md`, 2026-09-29). Verified by
running the Excel snippet itself (with `xl()` stubbed to return the same yfinance data) against
`BSF.py`'s `classify`. Every one of 6,297 PFE launches matched, in both European and American monitoring modes.

## Knock-out now resolves on the day it's hit (2026-10-07)

Bug: `classify` returned OUTSTANDING for any launch whose maturity date wasn't in the data yet,
*before* looking at the barrier. Under American monitoring that is wrong - a knock-out is terminal
the day the barrier is touched (same early-resolution shape as an FCN autocall, which the FCN
backtest already reports before maturity). A note launched 3 months ago that has already traded
through the barrier is not "outstanding"; its outcome is known. Fixed in both `BSF.py` and
`BSF Python-in-Excel.py`: the American barrier test now runs first, over launch -> maturity or, if
maturity isn't observable, launch -> last data date.

Unchanged on purpose:
- European monitoring still looks at the maturity day only, so a European launch without a maturity
  date is still OUTSTANDING (nothing can be known early).
- `Completed` still means "maturity date is in the data", exactly like an autocalled FCN. So an early
  knock-out counts in the all-launch mix but not in the completed cohort. Putting early-resolving
  launches into the completed cohort would bias it toward knock-outs (they resolve sooner than
  launches that survive), so the completed-cohort percentages are deliberately unaffected.

Verified on PFE (strike 100%, barrier 115%, 12m, 6,303 launches, data to 2026-10-06): American
all-launch KNOCKED_OUT 42.377% -> 43.519% and OUTSTANDING 3.982% -> 2.840% (72 launches moved);
completed-cohort figures and every European figure identical before/after. `BSF.py`, the BSF Excel
cell, both General cells, and a separate plain-loop reimplementation agree on all 6,303 launches in
both monitoring modes. Three self-test cases added for the early knock-out.
