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

## Wording follows the product; params read by row, not by label (2026-10-12)

Problem: the `General` sheet's G labels are fixed text, so with Product = RC, BRC or BSF the sheet
still showed "Autocall Trigger / Autocall Frequency Months / Autocall Lockout Months" - reading as if
those products autocall. They don't: only FCN has an autocall, and the code has always ignored those
three rows for anything else (`OBS_MONTHS` is empty unless Product is FCN). The wording was wrong,
not the numbers.

Two changes, both cells:

1. Params are read by **row position** in `G1:H12` rather than by matching the G label. The labels
   are now free text, so they can be formulas that reword themselves per product. Trade-off: the 12
   rows must stay in the documented order - inserting or reordering a row now silently misreads
   values instead of raising a KeyError. Rejected alternative: keep label matching and accept a list
   of aliases per row - more code, and every new wording would need a code edit.
2. The table gets a `Terms` row and the chart title uses the same string, built per product: FCN
   shows the autocall, RC shows strike only, BRC says "knock-in barrier", BSF says "knock-out
   barrier" (both with the monitoring mode). Nothing about autocall appears unless Product is FCN.

Label formulas for the sheet (paste into column G; `H2` is the Product cell):

    G4  =SWITCH(UPPER(TRIM($H$2)),"BRC","Knock-In Barrier","BSF","Knock-Out Barrier","Barrier (not used)")
    G5  =IF(OR(UPPER(TRIM($H$2))={"BRC","BSF"}),"Barrier Monitoring","Barrier Monitoring (not used)")
    G6  =IF(UPPER(TRIM($H$2))="FCN","Autocall Trigger","(not used - no autocall)")
    G7  =IF(UPPER(TRIM($H$2))="FCN","Autocall Frequency Months","(not used - no autocall)")
    G8  =IF(UPPER(TRIM($H$2))="FCN","Autocall Lockout Months","(not used - no autocall)")

Verified: ran both cells for FCN, RC, BRC and BSF with every G label replaced by junk text - same
outcomes as before, and the Terms row / chart title match the product in each case.

## Chart: outstanding launches shaded grey (2026-10-12)

Prompted by an AAPL 95%-strike, 12m chart showing no PHYSICAL_DELIVERY lines from 2025 on, which
looked like a bug next to the April 2025 sell-off (close down to ~172). It isn't one: the strike is
tested on the maturity day only, lines sit at the *launch* date, and every 12m note launched in
2025 Q1-Q3 matured (or will mature) with AAPL back above 95% of its launch level - e.g. Q1 2025
launches were struck at closes of 210-247 and matured in Q1 2026 at 247-278. The dip in between is
ignored by a maturity-only test. Shorter tenors do show it: the same terms give 90 delivery launches
dated 2025+ at 3m and 26 at 6m, versus 0 at 12m. Chart lines equal a plain-loop reimplementation's
delivery launches exactly for RC and FCN at 3m/6m/12m.

The real gap was that the last tenor's worth of launches has no outcome at all yet, and the chart
drew that stretch the same as "nothing bad happened". Those launches are now drawn in grey with
their own legend entry.

## BRC: BREACHED_RECOVERED bucket (2026-10-12)

Both cells split BRC's MATURITY_CASH into "never breached" and "breached, recovered to >= strike" -
see `Product Backtest/BRC/DEVLOG.md` (2026-10-12) for the rule, why it is empty under European
monitoring, and the verification. FCN and RC are unaffected (no barrier). The chart still marks
PHYSICAL_DELIVERY only for BRC.
