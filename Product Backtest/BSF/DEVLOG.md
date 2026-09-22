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

Barrier monitoring is continuous (daily) only, unlike BRC's European/American toggle - real Shark
Fins are conventionally daily-monitored barrier options, and a European (maturity-only) variant
isn't a real product anyone sells under this name, so no toggle was added, matching the "as compact
as possible" request instead of extending BRC's toggle to a product it doesn't apply to.

Verified: monotonic sanity check on AMD (barrier further from spot -> lower knock-out rate: 80.4%
-> 59.5% -> 51.0% for 110%/130%/150%, completed cohort), and PARTICIPATION + KNOCKED_OUT +
OUTSTANDING sums to 100% in every case (only two resolved buckets, same two-outcome shape as RC).
