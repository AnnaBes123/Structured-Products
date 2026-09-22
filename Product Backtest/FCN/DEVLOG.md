# FCN backtest — devlog

Chronological record of *why* things in `FCN.py` are built the way they are — decisions,
rejected alternatives, and bugs that got fixed. This file exists so `FCN.py` itself can stay
short (code + one-line WHY comments); anything long-form lives here instead. `README.md` still
owns the current-state explanation of what the script does — read that first if you just want
to understand the product/mechanics; come here for the "why isn't it simpler" history.

> Retroactively written 2026-09-21, when the repo adopted "short code comments, long rationale
> in DEVLOG.md" as its standing convention (see root `CLAUDE.md`). The entries below summarize
> decisions already present in the code at that point; they aren't in strict chronological order
> of when each was actually made.

## Launch-period handling: explicit bounds instead of an implicit BACKTEST_YEARS

Earlier version anchored the launch period to `BACKTEST_YEARS` counted back from the last launch
whose full tenor was *already* fully observable — which silently shifted the whole launch period
back by ~`TENOR_MONTHS` and dropped every recent launch from the report without saying so. Replaced
with explicit `LAUNCH_START` / `LAUNCH_END` / `DATA_AS_OF`: every trading day in the requested range
is launched and classified, and a launch too recent to have resolved is reported as `OUTSTANDING`
rather than quietly discarded.

## `DATA_AS_OF=None` resolves to yesterday, not today

Old script fetched through `END_DATE + 2 days` without ever trimming back to a real cutoff, which
risked scoring an observation off a trading session that hadn't actually closed yet. Fixed by
resolving `None` to "the most recent COMPLETE calendar day before now."

## Early-autocall observation must resolve strictly before maturity

An autocall observation date that resolves (via `map_observation`) to the *same* actual trading day
as maturity — or, in a badly gapped calendar, even after it — used to get evaluated as an ordinary
early-autocall check. That's wrong: the terminal test at maturity already covers that trading day,
so counting it twice (or letting a post-maturity "early" observation override the terminal test)
double-counts the same price. Fixed in `classify_launch`: such an observation is now marked
`COINCIDES_WITH_OR_AFTER_MATURITY_SKIPPED` and excluded from the early-autocall check. Covered by
synthetic test (k).

## No month-zero autocall, enforced structurally

`build_observation_offsets` always starts at `lockout_months + frequency_months`, never at
`lockout_months` itself, and `frequency_months` is always > 0 — so an offset of exactly 0 is
impossible by construction, not by a runtime check that could be forgotten or bypassed. Covered by
synthetic test (f).

## Observation-gap tolerance: why 5 days and 40 days aren't the same thing

`map_observation` rolls a scheduled observation date forward to the next available trading day
("following" convention). A roll of a few calendar days is an ordinary weekend/holiday roll. A roll
of, say, 40 days is much more likely a vendor data hole or a trading halt — evaluating the
observation against that later price would misrepresent what the note actually observed on its real
contractual date. `MAX_OBSERVATION_GAP_DAYS` draws that line; anything past it is flagged
`UNRESOLVED_GAP` rather than silently used. Covered by synthetic test (h).

## Basket dates: inner join, not forward-fill

For a worst-of basket, a date on which *any* ticker is missing a price (different exchange holiday
calendar) is dropped from *all* tickers, rather than forward-filled — so every remaining row is
genuinely comparable across the whole basket. This can drop a date one ticker actually traded on,
purely because another basket member didn't; the row-count before/after is printed so a large drop
(mismatched calendars, not just a shared holiday) is visible rather than silent.

## Why per-product terms instead of one shared STRIKE/AUTOCALL_TRIGGER

Early version applied one global `STRIKE`/`AUTOCALL_TRIGGER` (round numbers, 70%/100%) to whatever
ticker was plugged in. That's misleading for a high-drift name — "did the worst-of ever close back
above launch level" can be a low bar over a long history purely from drift, independent of whether
100%/quarterly is a sensible level for that specific underlying. Replaced with the `PRODUCTS` list
(one entry per note) plus the "Observed underlying behavior" diagnostic block, so `STRIKE`/
`AUTOCALL_TRIGGER` get picked with actual reference to that underlying's own historical behavior.
See README "Each product is observed and set up separately."

## Coverage reporting

Old script never reported requested-vs-actual launch/data coverage, which silently implied the full
requested period was always fully tested. `run_backtest` now returns a `coverage` dict
(`insufficient_history`, actual data start/end, actual first/last launch) and the report prints it.

## Cross-check as a second, independent code path

`cross_check_autocall_rate` reconstructs the AUTOCALL determination purely from the stored
`Obs*_Ratio`/`Obs*_Status` audit columns — a different code path from `classify_launch`'s own
autocalled/break logic — and asserts an exact match. This exists to catch a class of bug where the
classification logic and the audit table it writes silently drift apart, not merely to re-display a
number the same loop already produced.

## Growth chart with launch dots (added 2026-09-21)

Added a second chart, `<name> - growth.png`, plotting the underlying's (or worst-of basket's)
normalized price path with one dot per tested launch date, colored by that launch's eventual
outcome — same 4-color scheme as the existing stacked-bar chart (dodgerblue/seagreen/firebrick/
lightgrey for AUTOCALL/MATURITY_CASH/PHYSICAL_DELIVERY/OUTSTANDING). Motivation: a headline
"65% autocalled" is hard to sanity-check from percentages alone; seeing dense blue stretches of dots
sitting on a long uptrend (and red/grey dots clustered around drawdowns) makes it visually obvious
*why* the mix looks the way it does, and makes an implausible-looking number easy to spot-check
against the actual price path instead of taken on faith.

## Condensed report split into its own script (2026-09-21)

The percentages-only summary requested started as a block printed inline at the end of
`backtest_product` in `FCN.py`, but that kept the "short and efficient" condensed output bolted
onto the same run as the audit CSV, both charts, and the diagnostics grid. Moved it into a separate
`FCN Condensed.py` that imports `FCN.py`'s fetching/classification functions (`resolve_run_dates`,
`fetch_multi_asset_path`, `validate_price_df`, `run_backtest`) rather than duplicating them, and
only prints the outcome-percentage lines - no CSV, no charts, no diagnostics grid, no synthetic
tests. Runs in ~3s vs. `FCN.py`'s full run. `FCN.py`'s own report now just points to it instead of
also printing the same block inline.

## Excel output script added, then made fully standalone (2026-09-21)

Added `FCN XLSX.py`: same percentages as `FCN Condensed.py`, written to `FCN Percentages.xlsx` via
`pandas.DataFrame.to_excel` instead of printed. First version imported `FCN.py`'s fetching/
classification functions, same as `FCN Condensed.py` does - but the point of this script turned out
to be portability ("move this to a new OS with nothing tying it to this venv or repo"), so it was
rewritten to import nothing from this repo: a compact, standalone reimplementation of
`classify_launch`'s rules (single-file, only third-party deps `pandas`/`yfinance`/`openpyxl`, copy
anywhere and `pip install` those three). It intentionally does NOT share `FCN.py`'s audited code
path or synthetic test suite - use `FCN.py`/`FCN Condensed.py` instead whenever correctness under
edge cases matters more than portability. Kept deliberately minimal - doubles as a first example of
"how do I get a Python result into Excel": build one dict of numbers into a DataFrame, then one
`to_excel(...)` call; the only non-obvious part is the `number_format = "0.0%"` loop, needed because
`to_excel` writes the raw fraction (0.657) and Excel doesn't know to display it as a percentage
without being told. Added `openpyxl` to `requirements.txt` and `*.xlsx` to this folder's
`.gitignore` (same ad hoc/run-specific treatment as the audit CSV and charts).

## Product label derived from TICKERS, not hand-maintained (2026-09-21)

`PRODUCTS` used to be a dict keyed by a hand-written name (e.g. `"PFE FCN"`), used for the printed
report header, chart/CSV filenames, and the condensed report - so swapping `TICKERS` to a different
name (e.g. `RACE`) left the report still saying "PFE FCN" until the key was *also* edited by hand.
Fixed by making `PRODUCTS` a plain list and adding `product_label(terms)`, which derives the label
from `TICKERS` every time (`"RACE FCN"`, or `"AAPL/MSFT FCN"` for a basket) - there is no longer a
separate name to forget to update. `FCN Condensed.py` imports and uses the same `product_label`;
`FCN XLSX.py` (standalone, no import from `FCN.py`) has its own copy of the same one-line rule.

## Excel-hosted, not Excel-output (2026-09-21)

`FCN XLSX.py` briefly wrote its result to `FCN Percentages.xlsx` via `pandas.to_excel` - wrong
goal. The actual goal is to have this logic run *from inside* Excel itself (a workbook that
computes the percentages when opened/refreshed), not a Python script that happens to produce an
.xlsx as its output. Reverted to printing the percentages; the file stays a standalone,
no-repo-import script (see "Excel output script added, then made fully standalone" above) since
that constraint carries over either way. Not yet decided: Microsoft 365's native `=PY()` "Python in
Excel" runs in a sandbox with **no internet access**, so it cannot call `yfinance` directly - the
fetch step would have to happen outside Excel (a separate script caching prices to a file/table
Excel reads) with only the classification math running inside Excel's own Python. The alternative
is a bridge library like `xlwings`, which runs a real local Python process with full internet
access and can call `yfinance` directly from a button/macro in the workbook. Which of these to
actually build hasn't been decided yet.

## Terms read from a Params table, not hardcoded in the =PY() cell (2026-09-21)

`FCN Python-in-Excel.py` first had STRIKE/AUTOCALL_TRIGGER/etc. as constants inside the Python
snippet itself, same as `FCN XLSX.py` - but that means anyone who wants to try a different strike
has to open the formula bar and edit Python, which isn't "easily accessible for people" who aren't
the one who wrote it. Changed to read every term from a `Params` table via `xl(...)` instead. Two
side benefits of doing it through `xl()` rather than hardcoding: (1) the `=PY()` cell now
recalculates automatically whenever a Params cell changes, same as any other Excel formula - no
button, no reopening the formula bar; (2) verified locally by stubbing `xl()` to return the same
shapes Excel would and exec'ing the snippet body against real AMD data - reproduced FCN.py's
audited numbers exactly, so the parameter-plumbing didn't introduce a bug.

First attempt read it as a one-row table at `A1:H2` (headers row 1, values row 2) - the sheet as
actually built didn't match (see below), which produced `ValueError: could not convert string to
float: 'Autocall Trigger'`: headers=True treated row 1 as the header and the *next label down* as
if it were Strike's value. Briefly switched to a vertical label/value-per-row layout instead, then
back again once the sheet was rebuilt to the horizontal form at `B2:H3` (header row 2, values row
3, offset from A1 to leave room for a sheet title/row label) - same `headers=True` reading, just a
different range. Re-verified against the same stubbed-AMD test at each step; all three shapes
(A1:H2, vertical A1:B8, B2:H3) reproduce FCN.py's audited numbers exactly once the code matches
whatever the sheet actually is - the lesson isn't "one layout is right," it's that the `xl(...)`
range and headers=True/off setting have to be re-verified any time the sheet layout changes.

Settled back on the vertical (down-rows) layout at `A1:B8` as the final form, for both Params and
now the output too (see below) - each metric its own row reads better than one very wide row once
there are 8+ columns, and matches how a person naturally lists inputs one-per-line.

## Oversized price range instead of an exact one (2026-09-21)

`xl("Prices!A1:B6300", headers=True)` had to be hand-edited to a new row count every time the
FactSet pull's date range changed, which is exactly the kind of "change the code manually" friction
this whole Params-table effort was meant to remove. Fixed the actual constraint, not just moved it:
`.dropna()` (already present) strips any blank trailing rows, so a range that's too BIG is
harmless - only one that's too SMALL silently truncates real data. Bumped it to `A1:B100000`
(centuries of daily data) once, and it should never need bumping again for a wider date range -
only widening (more columns) if a worst-of basket adds more tickers.

## Ticker label read from its own cell, not the price table's column header (2026-09-21)

The Product label was derived from the price table's own column header (`tickers =
list(prices_df.columns)`), which showed "Close FCN" instead of e.g. "PFE FCN" because FactSet's
pull auto-labeled that column "Close" rather than the ticker. Simplest fix is renaming that header
cell in Excel to the actual ticker - no code change needed, since the label is already derived from
it. If you'd rather leave FactSet's own header alone, the code instead reads the ticker from its own
cell via `xl("Prices!I2")` (adjust to wherever you actually typed it for the FactSet formula) and
uses that for the label - re-verified against the stubbed-AMD test with the price column
deliberately left named "Close" to confirm the label now comes from the separate cell instead.

## Output as a vertical Metric/Value table, percentages pre-formatted as text (2026-09-21)

The result table was one wide row (`Product | Launches | Autocall % | ...`, 10 columns) with raw
fractions (0.711493) rather than percentages. Changed to a two-column `Metric`/`Value` table, one
row per number - reads better spilled into a sheet than a 10-column-wide row - and pre-formats every
percentage as a string (`f"{x:.1%}"`, e.g. "71.1%") rather than leaving it as a raw fraction for a
manual Excel number-format step. Trade-off: those Value cells are text, not numeric, so they can't
be used directly in further Excel arithmetic - acceptable here since this table is a terminal report,
not an intermediate calculation input. Re-verified against the stubbed-AMD test with a padded (too-
big) price range to confirm both changes together still reproduce the audited numbers exactly.

## Terse code / devlog split (2026-09-21)

Repo-wide convention change (see root `CLAUDE.md`): code comments in `FCN.py` (and new/edited files
elsewhere) are now short, WHY-only one-liners; the long-form rationale that used to live in the
module docstring, function docstrings, and printed runtime caveats moved here and to `README.md`
(which already carried most of it). Nothing about the backtest's logic changed — this was a
documentation-location change only.
