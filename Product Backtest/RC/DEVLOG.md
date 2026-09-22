# RC (Reverse Convertible) Python-in-Excel — devlog

Chronological decision history for `RC Python-in-Excel.py`. See `Product Backtest/FCN/DEVLOG.md`
for the shared Python-in-Excel conventions this reuses (oversized price range, vertical Params
list, `xl()` mechanics) - this file only covers what's specific to RC.

## Added, derived from FCN's non-autocall path (2026-09-22)

A plain Reverse Convertible has no autocall - the underlying is compared to STRIKE once, at
maturity, full stop. `RC Python-in-Excel.py` is `Product Backtest/FCN/FCN Python-in-Excel.py`'s
`classify_launch` with the entire autocall-observation loop deleted; everything else (gap-tolerant
maturity mapping, worst-of via `.min()`, inclusive strike boundary) is unchanged.

Verified two ways before handing it over: (1) internally, MATURITY_CASH + PHYSICAL_DELIVERY +
OUTSTANDING sum to 100% - no fourth bucket possible with no autocall; (2) externally, against
`FCN.py`'s own `diagnostic_stats` "P(finishes below strike at maturity, IGNORING autocall)" for
AMD at a 90% strike (35.7%) - RC's Completed Physical Delivery % matched exactly, which makes sense
since that diagnostic *is* the RC payoff test run on the same data.

Kept to one sheet ("RC": A:B price pull, D:E params, G1 the `=PY()` formula) and ~70 lines per
request - simpler than FCN's three-sheet layout mainly because there's no autocall-observation loop
or its parameters (frequency/lockout) to plumb through.

## Trimmed further (2026-09-22)

Same trims as `Product Backtest/BRC/BRC Python-in-Excel.py` (see its DEVLOG.md for the two bugs
that first attempt introduced and how they were caught): merged `map_obs`'s branch into one
`return`, folded `pct`+`fmt_pct` into one function, combined related assignments with
tuple-unpacking. Applied carefully this time (single `xl()` call, correct `.set_index`) and
re-verified against the same stubbed-AMD test before considering it done - matches exactly
(61.7%/34.3%/4.0% and completed 64.3%/35.7%).
