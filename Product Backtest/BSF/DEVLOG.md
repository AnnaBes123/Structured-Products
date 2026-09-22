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
