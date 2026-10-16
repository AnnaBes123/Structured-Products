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

## Inputs: percentages, named autocall frequency, "delay to first observation" (2026-10-13)

Both cells; the 12 rows and their order are unchanged, only what rows 3-4, 6, 7, 8 accept.

- **Levels as percentages (Strike, Barrier, Autocall Trigger).** Enter 130%, not 1.3. A cell Excel
  shows as 130% already holds 1.3, so that needed no code. What is new: a typed plain `130`, or the
  text `130%`, is also read as 130%. Rule: anything above 5 is taken as percentage points (no real
  strike or barrier is above 500%). The cost of that rule: a level genuinely meant as a fraction above
  5 can't be entered, and a plain `5` means 500%, not 5%.
- **Autocall Frequency** (was "Autocall Frequency Months"): Daily, Monthly, Bi-Monthly, Quarterly,
  Semi-Annually. Blank or 0 = no autocall. A bare number is still read as months, so an existing
  sheet with `3` keeps working. Bi-Monthly is read as **every 2 months** (not twice a month) - the
  usual meaning for an observation schedule, but the word is ambiguous. An unrecognised word raises
  an error rather than silently meaning "no autocall".
- **Daily** is new behaviour, not just a new label: every trading day after launch (after the delay)
  and strictly before the maturity day is an observation. With the trigger at 100% that calls almost
  every note that ever closes above its launch level, so expect very high AUTOCALL shares.
- **Autocall Delay to First Observation (Months)** (was "Autocall Lockout Months"): `No delay`, 0 or
  blank, or whole months. Same rule as the old lockout, renamed: the delay is added *before the first
  period starts*, so the first observation is at delay + one period (quarterly + 3 -> month 6, month 3
  skipped). Rejected alternative: treating the number as the month of the first observation itself -
  that would have silently changed results for anyone already using a lockout.
- **Tenor Months** was already live; it now also tolerates `12m`. The `Terms` row / chart title echo
  the tenor the cell actually used, which is the quick way to confirm an edit was picked up.

Verified on AAPL (launches from 2012-01-03): (1) old-style numeric inputs give launch-for-launch the
same outcomes and chart lines as the previous commit, for FCN (4 schedules), RC, BRC and BSF;
(2) `90` / `"90%"` / `0.9`, `Quarterly` / `3`, `No delay` / `0` / blank all give identical results;
(3) every frequency x {no delay, 3m delay} x {6m, 12m tenor} matches a plain-loop reimplementation
on every launch, table and chart; (4) tenor 3/6/9/12/18/24/36 each give a different mix (e.g.
PHYSICAL_DELIVERY 13.2% at 3m, 5.5% at 12m, 0.0% at 24m for strike 90%, quarterly).

The single-product cells (`FCN Python-in-Excel.py` etc.) still take fractions, a months number and
"Autocall Lockout Months" - not changed here.

Label formulas, replacing the 2026-10-12 ones for rows 7-8:

    G7  =IF(UPPER(TRIM($H$2))="FCN","Autocall Frequency","(not used - no autocall)")
    G8  =IF(UPPER(TRIM($H$2))="FCN","Autocall Delay to First Observation (Months)","(not used - no autocall)")

## Multiple underlyings, Settlement row, BSF bucket names (2026-10-14)

- **Worst-of baskets.** Already worked (2+ price columns in A:E, up to 4 tickers; every test uses the
  worst performer vs. its own launch level). New: the `Product` row and chart title say
  "AAPL/MSFT/NVDA worst-of", taken from the price headers when the `Ticker` cell is blank (the
  `Ticker` cell is now display-only).
- **Settlement (new row 13): Physical / Cash, blank = Physical.** Renames the FCN/RC/BRC loss bucket
  to PHYSICAL_DELIVERY or CASH_SETTLEMENT. It changes no outcome: below strike the investor gets the
  worst performer's shares or their value in cash, which are worth the same on the day. Both cells
  now read `G1:H13`; a sheet that still stops at row 12 reads as Physical.
- **Loss-case rows in the table** (only when there are losses):
  - avg and worst value received, % of notional = worst performer's maturity level / strike (e.g.
    strike 80%, worst finishes at 60% -> 75% of notional). Excludes coupons, so it is not the note's
    total return.
  - basket only: which underlying was the worst performer the loss was settled on, % of loss cases.
- BSF buckets renamed to UPSIDE_PAID / NO_UPSIDE_BELOW_STRIKE - see `Product Backtest/BSF/DEVLOG.md`.

Verified: for FCN, RC, BRC, BSF (American and European), on AAPL and on AAPL/MSFT/NVDA, both
settlement settings: every launch's outcome and every chart line equal the previous version's after
renaming. Loss-case stats equal a plain-loop reimplementation (basket RC, strike 80%, 12m: 436 losses,
avg 83.3%, worst 56.2%, settled on AAPL 18.8% / MSFT 7.1% / NVDA 74.1%).

Label formula for the new row (optional):

    G13 =IF(UPPER(TRIM($H$2))="BSF","(not used - no loss case)","Settlement")

## Chart: every underlying shown for a basket (2026-10-14)

`General Price.py` already used the worst-of and already read percentage inputs (130 = "130%" = 1.3;
re-checked: identical chart lines for BSF, BRC and FCN, single and basket). New: with 2+ tickers it
now draws each underlying rebased to 100% at the first date, plus the worst-of in black - the line
the note actually observes - with all names in the legend. The basket y-axis is log-scale, in %,
because one big winner (NVDA ~450x since 2015) otherwise flattens every other line. Note the black
line is the worst-of vs. the *first date of the chart*; each note's own tests use the worst-of vs.
its own launch date, which can be a different underlying. Single-ticker charts are unchanged (price,
linear axis).

## Chart: status selection grid, one colour per outcome (2026-10-15)

`General Price.py` can now draw any of the outcomes, not just the bad one. A two-column grid on the
sheet (`General!J1:K12`, written as a literal `xl("General!J1:K12")` - Excel finds a cell's dependencies from the xl() text, so a variable holding the address may not refresh the chart: status name | TRUE/FALSE, checkbox, Y, x or 1) picks
which; each gets its own colour (autocall blue, cash at maturity / upside paid green, breached-
recovered / no upside amber, loss / knocked out red, outstanding grey). Statuses that don't apply to
the Product are ignored, PHYSICAL_DELIVERY and CASH_SETTLEMENT are interchangeable names for the loss
row, and spacing/case don't matter. No grid, an empty range or nothing ticked falls back to the old
picture (loss or knock-out, plus outstanding), so an existing sheet doesn't change.

Grid matched by label, unlike the params (matched by row): status rows are optional and in any order,
so position would be fragile here. Lines stay at the launch date, except KNOCKED_OUT at the knock-out
date (2026-10-08 entry). Underlyings now use their own palette (purple / brown / pink / slate) so a
ticker line is never the same colour as a status. Legend moved under the plot - with 4 tickers and
4 statuses it covered the data.

Kept in step with the table: the chart's params block and classification rules are the same code as
`General Python-in-Excel.py` (percent levels, named frequencies, delay, settlement, worst-of).
Verified: with every status ticked, each status's lines are exactly the launches the table cell puts
in that bucket (knock-outs: same count, drawn on event dates) - FCN quarterly and daily, RC, BRC,
BSF American and European, on AAPL and AAPL/MSFT/NVDA. Grid parsing checked for: no grid, xl()
raising, lowercase/spaces/"y", the cash-settlement name, non-applicable rows, nothing ticked.

## Chart colours: one per status, red only for the bad outcome (2026-10-15)

Previously the cash-at-maturity / upside-paid pair shared green and the recovered / no-upside pair
shared amber. Now every status has its own colour: autocall blue, cash at maturity green, breached-
recovered amber, upside paid teal, no upside below strike lavender, outstanding grey, and red only for
the product's bad outcome (loss for FCN/RC/BRC, knock-out for BSF - never on the same chart). The bad
outcome is also drawn stronger (alpha 0.25 vs 0.12) so red reads first. Tickers moved to navy /
brown / pink / olive so none matches a status. Outcomes unchanged (same 12-case check as above).

## Pasted code stripped of comments: Excel's 8,192-character cell limit (2026-10-15)

Excel rejected the chart cell: a =PY() cell holds at most 8,192 characters and the pasted part of
`General Price.py` had grown to 9,309. Comments and function docstrings were 2,200 of that. Both cells
now have no comments below the `---8<---` line; the reasons moved into a "Code notes" section in each
file's module docstring (above the cut, never pasted) and stay in this devlog. Sizes now: chart 7,057,
table 5,468 characters - about 1,100 / 2,700 of headroom.

Rejected alternative: having the chart reuse the table cell's functions through Python in Excel's
shared workbook namespace (cells run in row-major order and see earlier cells' globals). It would
roughly halve the chart, but it silently breaks if the chart cell ever sits before the table cell in
calculation order, or if the table cell errors.

Verified: the code (syntax tree, docstrings ignored) is identical to the previous commit, apart from
the literal-xl() grid change above; all 12 chart-vs-table checks still pass.

## Loss-case rows: final level vs. launch, not value received (2026-10-15)

"Avg value received (% of notional)" = worst level / strike was correct but read as confusing next to
the strike: different units (notional vs. launch price), and it is above the strike whenever the
strike is below 100% (e.g. strike 80%, finish at 70% -> 87.5% of notional, because the investor
receives notional / strike-price shares). Replaced with the worst performer's final level as % of
its own launch price - same units as the strike, so "strike 80%, avg loss-case finish 66.6%" compares
directly. "Settled on X" became "worst performer was X". Value received can still be derived:
final level / strike. Verified on AAPL/MSFT/NVDA RC, strike 80%, 12m: 436 losses, avg 66.6%,
lowest 44.9%, equal to a plain-loop reimplementation. Table cell now 5,452 pasted characters.

## Loss-case rows removed (2026-10-15)

Removed the loss-case rows (final level of the worst performer, which underlying it was) from the
table cell, same day they were reworded. The cell is a status calculator - share of launches per
outcome - and payoff-style figures were extra detail that needed explaining every time. The
worst-of basket naming and the Settlement row stay. Outcome percentages unchanged; chart lines still
equal the table's buckets in all 12 cases. Table cell now 4,905 pasted characters.

## Chart says why a status is missing (2026-10-15)

Reported: other ticked statuses drew, AUTOCALL drew nothing. Not reproducible locally (FCN with
AUTOCALL ticked draws blue on every test case), so the cause is on the sheet. Rather than guess, the
chart now reports it:

- Second title line: where the selection came from - "Status grid J1:K12", plus any ticked statuses
  that don't apply to the Product ("n/a for RC: AUTOCALL"), or "DEFAULT VIEW - no ticked status read
  from General!J1:K12" when the grid wasn't read.
- AUTOCALL ticked but no lines: the legend gives the reason - Autocall Frequency blank/0; first
  observation month (delay + one period) not before maturity (e.g. quarterly on a 3m tenor); or no
  observation closed at/above the trigger.

Checked each message on a synthetic case; chart lines still equal the table's buckets in all 12
cases. Chart cell now 7,689 pasted characters (~500 left under Excel's 8,192).

## More forgiving inputs: frequency spellings, status labels (2026-10-15)

- Autocall Frequency (both cells) now also takes "3m", "3 months", "Quarter", "Annually"/"Annual"
  (12 months) and any spacing/hyphenation ("Semi Annual"). Previously those raised an error and the
  whole cell failed.
- Status grid labels (chart) are matched on letters only, so "Auto call", "Auto-Call", "AUTOCALL %"
  and "Autocalled" all mean AUTOCALL; same for the other statuses. Prompted by AUTOCALL not drawing
  on the user's FCN sheet while other statuses did - a label the old exact match didn't recognise is
  the most likely cause (the frequency cell was evidently parsing, or the whole chart would have
  errored).

Checked: every spelling above maps as intended; label variants resolve; all 12 chart-vs-table checks
and the diagnostic messages still pass. Pasted sizes: chart 7,871 (~300 under the 8,192 limit),
table 5,029.

## European BSF knock-outs: normal opacity, "maturity date" in the legend (2026-10-16)

European BSF looked like solid red blocks next to American. Cause: knock-out lines were always drawn
at alpha 0.5, chosen for American, where many launches knock out on the same day and the lines are
few (AAPL, 130% barrier, 12m: 1,636 knock-outs on 227 distinct days). Under European every knock-out
is on its own maturity day, so the lines are nearly one per launch (1,085 on 1,017 days) and 0.5
merged them into blocks. The strong alpha now applies only to American knock-outs; European uses
the normal 0.25. The European legend says "maturity date" (that's when the test happens), and the
day count reads "on N distinct days". Wide European bands are still expected - they are runs of
consecutive maturity days that all finished above the barrier. 12/12 chart-vs-table checks pass.
Chart cell now 7,964 pasted characters - about 230 left under Excel's 8,192.

## Daily stored as "DAILY", not 0 (2026-10-16)

Internally `frequency()` used 0 to mean "every trading day", while a 0 typed into the sheet means
"no autocall". Results were right (a typed 0 never reached the daily path), but the same number meant
opposite things on the sheet and in the code. Daily is now the marker "DAILY" in both cells, so 0 /
blank only ever means no autocall. Daily itself is unchanged and still selected by typing `Daily`.
Verified on AAPL FCN (strike 90%, trigger 100%, 12m): Daily 99.1% autocall, Daily + 3m delay 93.1%,
0 and blank 0.0%, Quarterly 85.8%, 3 + 3m delay 78.8% - table and chart agree in each; 12/12 chart-vs-
table checks pass. Chart cell 8,037 pasted characters (~150 under Excel's 8,192).

## Review pass: robustness, chart clarity, size (2026-10-16)

Full review of both General cells. Outcome rules unchanged; fixes are about bad input and display.

Input robustness (both cells):
- Blank Launch Start used to give zero launches; it now means the first price date.
- Price dates are accepted as real dates, text, or Excel serial numbers (a General-formatted Date
  column arrives as numbers like 45000 - converted from Excel's 1899-12-30 origin, not read as
  nanoseconds since 1970). Non-numeric prices ("#N/A", text) become blank and the row is dropped,
  instead of the cell failing on float conversion.
- A typed Ticker name keeps its case ("Tech basket", not "TECH BASKET").
- An unknown Product raises "Product (H2) must be FCN, RC, BRC or BSF, not X" instead of a KeyError.

Chart clarity:
- Price / underlying lines drawn above the status lines (zorder) so they are never hidden.
- Legend in the same order as the table (OUTSTANDING last), with each status's share of launches
  next to the count - the same numbers as the table's "% " rows.
- Second title line only when there is something to say: default view (no ticked status), or a grid
  label that isn't any status ("Not a status - check J:K: Phys Delivry"). Ticking another product's
  statuses is normal with one shared grid, so it is no longer listed.
- One combined AUTOCALL "none" message replaces the four specific ones (character budget).

Size: Excel's 8,192 limit may count a line break as two characters, so the chart cell is now measured
both ways: 7,924 characters, 8,086 with two-character line breaks. The grid reader and drawing loop
were rewritten more compactly to fit the above; one dead branch removed (a knock-out is always BSF's
bad outcome, so the "non-bad knock-out" opacity could never apply).

Verified: 12/12 chart-vs-table checks; every grid label variant; every frequency spelling; FCN cell vs
General table; annual at 12/24/36m vs a plain loop; blank Launch Start, text dates, serial dates,
"#N/A" rows and blank columns each give the same outcomes as a clean sheet in both cells. Charts
rendered and inspected for FCN and BRC (single), RC and BSF (4-stock basket), default view and a
mistyped label.

## Single quotes in the pasted code: Excel's stored formula length (2026-10-16)

After the review pass, Excel refused to save the chart cell ("an error occurred trying to save
changes") although its visible length (7,924) was under 8,192. Likely cause: Excel stores a Python
cell inside a formula string, where each double quote is stored escaped as two characters - the
chart had 386 of them, so its stored length was ~8,310, over the limit, while the editor showed less.
The review fixes added quotes, which is why it broke only then. All strings in the pasted code of
both General cells and the FCN cell now use single quotes (Python treats them identically); only
the few strings that themselves contain a single quote keep double quotes. Syntax trees are
identical before/after, so behaviour is unchanged. Worst-case stored length (doubled quotes +
two-character line breaks): chart 8,090, table 5,578, FCN 4,884. This is an inference from the
symptoms, not a documented Excel rule - if a save still fails, the chart cell needs trimming further.

## Up to 15 underlyings; new sheet layout, no sheet prefix (2026-10-16)

The 4-ticker cap came from the layout: prices in A:E, params starting in G. New layout, matching the
user's sheet:

- Prices A:P (Date + up to 15 tickers), read as `xl('A1:P100000')`.
- Params R3:S15, by row position in the documented order (R3 Ticker ... R14 Max Gap Days, R15
  Settlement). The user's table is R3:S14; R15 = Settlement is optional (blank = Physical).
- Status grid R23:S30 (chart only).
- No sheet prefix in any range: the cells read the sheet they sit on, so the sheet can have any name
  but prices, params, grid and both cells must be on the same sheet. (Also saves characters.)
- Label wording in the docs: "Autocall Delay (Months)". Labels are still free text (read by row).
- Tickers 5-15 get colours from matplotlib's tab20b palette after the 4 fixed ones; the legend
  switches to 4 columns above 4 tickers. With 15 names some colours are similar - the legend and the
  black worst-of line are the reliable reads.

Label formulas from earlier entries referenced $H$2 for Product; with the new layout Product is S4,
so use $S$4 in them.

Also: strings in the pasted code switched to single quotes (Excel stores each double quote as two
characters toward 8,192), and continuation lines re-indented minimally - syntax trees unchanged.
Sizes, worst case (doubled double quotes + two-character line breaks): chart 8,040, table 5,469.

Verified: 15-ticker RC basket (AAPL ... INTC, strike 70%, 12m) equals a plain-loop worst-of
reimplementation on every launch and chart lines equal the table; 12/12 chart-vs-table checks; grid
label and messy-data checks unchanged; the test stub now fails if any range carries a sheet prefix.
