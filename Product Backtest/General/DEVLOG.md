# General (all-product) Python-in-Excel cells — devlog

Chronological decision history for `General Python-in-Excel.py` (outcome table) and
`General Price.py` (price chart). Product rules themselves are documented in each product's own
folder; this file only covers what's specific to the combined cells.

## BSF: chart marks knock-outs; knock-out resolves on the day it's hit (2026-10-07)

Two BSF problems, both fixed in both cells:

1. `General Price.py` marked PARTICIPATION_OTM launches for BSF - i.e. launches where the stock
   simply finished below strike. That made the BSF chart look like the FCN/RC one (red wherever the
   stock fell) and never showed the knock-out, which is the event that defines a shark fin. BSF now
   marks KNOCKED_OUT. The chart title also shows the barrier level and monitoring mode for BRC/BSF,
   because the picture depends heavily on it (PFE, 115% barrier: 1,388 marked launches European vs.
   2,743 American).
2. A launch whose maturity wasn't in the data yet was OUTSTANDING even if it had already traded
   through the barrier. Under American monitoring it is now KNOCKED_OUT - see
   `Product Backtest/BSF/DEVLOG.md` (2026-10-07) for the reasoning, the unchanged `Completed`
   convention, and the verification.

## BSF: lines drawn at the knock-out date, not the launch date (2026-10-08)

For BSF only, each marked launch's line now sits on the day the knock-out occurred: the first
trading day the worst-of level is at/above the barrier (American), or the maturity day (European -
the only day the barrier is tested). FCN/RC/BRC are unchanged and still mark the launch date, since
"delivery" has no single event day before maturity.

Many overlapping launches knock out on the same day (one rally takes out every note launched in the
preceding weeks), so the days are de-duplicated before drawing and the legend reports both counts -
PFE, 115% barrier, American: 2,743 knocked-out launches fall on just 347 distinct days. Because the
lines no longer stack, BSF uses a higher alpha (0.5 vs 0.15). A consequence worth knowing when
reading the chart: line density no longer shows *how many* launches knocked out, only *when*.

Verified: knock-out dates match a plain-loop reimplementation for all launches in both monitoring
modes; the chart cell's outcomes still equal the table cell's for FCN, RC, BRC and BSF.

European monitoring still only tests the maturity day - set `Barrier Monitoring` to American for
the path to be observed.
