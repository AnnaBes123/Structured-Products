# FCN Rolling Historical Backtest

Answers a third, different question from the other two folders in this repo: not "what is this
worth today" (`Product MtM/`) and not "what does a fitted statistical model say the real-world
odds are" (`Estimators/`), but **"what would actually have happened to this note, mechanically,
if you'd launched it on every single trading day of the last N years?"** No option pricing, no
QuantLib, no Monte Carlo — just the note's own payoff rules applied to one real historical price
path (or worst-of basket of paths).

## Known biases — read this before trusting any number below

Two structural biases apply to every product here, and neither is fixed by picking a "better"
`STRIKE`/`AUTOCALL_TRIGGER` or a different underlying — they're properties of the backtest design
itself, not of any one product's parameters:

1. **Period/regime dependence.** Every product here — single-name or index — is backtested over
   whatever one historical window `LAUNCH_START`/`LAUNCH_END`/`DATA_AS_OF` happen to cover for that
   run. Whatever that window's regime was (bull, bear, sideways, one long recovery), the reported
   percentages describe *that* window and that window only. This script does not attempt to
   quantify how much of the result is regime-driven versus name-driven — doing that rigorously
   would mean deliberately testing multiple, materially different historical windows (or different
   markets/eras) side by side, which is out of scope here. Don't extrapolate one run's numbers to
   "what this product would typically do."
2. **Survivorship bias, single-name products only.** A single-name `TICKERS` entry is, by
   construction, a name that's still listed today — a company that got delisted or went bankrupt at
   some point never gets tested, because nobody thinks to plug a dead ticker into `TICKERS`. This
   doesn't apply to a broad index (constituents get replaced when a company fails, rather than the
   index itself vanishing), but it does apply to every single-name product in this file.

Nothing in this script corrects for either bias — doing so would require deliberately sourcing
failed/delisted names and testing alternative historical windows/regimes, which is out of scope for
a tool that (by design) just replays one real historical price path. Treat every percentage below as
"what this one specific historical window produced for this specific still-listed name," not as a
general real-world FCN base rate — and see "these are also not independent trials" further down for
a second, separate reason not to over-read the headline number.

## Product: Fixed Coupon Note (FCN), worst-of and physically settled

A plain FCN has **no knock-in barrier** — that's a `Barrier Reverse Convertible`-specific
feature (see `Product MtM/Yield/Barrier Reverse Convertible/`), not an FCN one. This script
models an FCN as two features:

- **Autocall**: on each observation date, if the worst-performing underlying (relative to its own
  level at launch) is at or above `AUTOCALL_TRIGGER`, the note redeems immediately at 100% of par.
- **Maturity**, if never autocalled: the worst-of level is compared to `STRIKE` **once, at
  maturity only** — European, no interim monitoring, exactly the "only where it ends up matters"
  convention already used by `Product MtM/Yield/Fixed Coupon Note/` (see that README's "Terminal
  payoff" section). At or above `STRIKE`, full capital back. Below it, the investor is delivered
  the worst-performing underlying at a conversion ratio of `1/STRIKE` — a capital loss.

This variant differs from `Product MtM/Yield/Fixed Coupon Note/` in being **worst-of** and
**physically settled** (delivery of the underlying, matching real-world FCN market convention)
rather than single-name and cash-settled (`Principal - max(Strike - S_T, 0)`); that's the flavor
of FCN this backtest's PHYSICAL_DELIVERY outcome is about. Single-name (`TICKERS = ["AAPL"]`) or worst-of
basket (`TICKERS = ["AAPL", "MSFT", ...]`) — same worst-of convention as `Estimators/P(Breach)
Estimator`.

## Each product is observed and set up separately - there is no shared default

Early versions of this script had one global `STRIKE`/`AUTOCALL_TRIGGER` applied to whatever
`TICKERS` you plugged in, using round numbers (70% / 100%) picked without reference to any
specific underlying. That's misleading for a high-drift name: "did the worst-of ever close back
above where it started, at any point in a 12-month window" can be a fairly low bar over a long
enough history purely because of long-run drift, regardless of how volatile the ride was -
independent of whether 100%/quarterly is actually a sensible level for that specific underlying.

So each run of `FCN.py` is one note, with its own `TICKERS`/`STRIKE`/`AUTOCALL_TRIGGER`/
`TENOR_MONTHS`/`AUTOCALL_FREQUENCY_MONTHS` set at the top of the file. The report title and output
filenames come from `TICKERS`, so changing `TICKERS` is enough on its own. Pick `STRIKE`/
`AUTOCALL_TRIGGER` with that underlying's own behavior in mind. The quickest check is to run
`RC.py` (`Product Backtest/RC/`) on the same ticker. Its PHYSICAL_DELIVERY rate is the raw
"finishes below strike, ignoring autocall" rate. The long `FCN.py` (in git history before the
2026-09-29 rewrite) printed a vol/trigger/strike diagnostic grid for this; the rewrite dropped it
(see DEVLOG.md).

## Why "rolling" and not one fixed backtest

Rather than pick one `ENTRY_DATE` (the `Product MtM/` convention), this script launches the note
**every single trading day** in `[LAUNCH_START, LAUNCH_END]` and classifies every one of those
launches independently — launches overlap, and a new launch never waits for an earlier one to
terminate. This trades the single-path illustration of `Product MtM/` for a distribution of
outcomes across many (overlapping) windows of the same underlying's history.

## The four outcomes reported

Every launch is classified into exactly one of four mutually exclusive statuses, as of
`DATA_AS_OF`:

1. **AUTOCALL** — redeemed early at par, at the first observation where the worst-of cleared
   `AUTOCALL_TRIGGER`.
2. **MATURITY_CASH** — never autocalled, ran the full tenor, worst-of finished at/above `STRIKE`:
   full capital back.
3. **PHYSICAL_DELIVERY** — never autocalled, ran the full tenor, worst-of finished below `STRIKE`:
   capital loss, delivery of the worst-performing underlying.
4. **OUTSTANDING** — hasn't autocalled yet, and its nominal maturity date is still after
   `DATA_AS_OF`: outcome not yet known. A recent launch is left here rather than forced into a
   resolved bucket it hasn't actually reached yet.

Two derived subtotals are reported alongside the four (each a combination of two outcomes, not a
fifth/sixth bucket of its own):

- **Reached maturity** = MATURITY_CASH + PHYSICAL_DELIVERY (ran the full tenor, whichever way it
  finished).
- **Repaid at par** = AUTOCALL + MATURITY_CASH (full principal back, either early or at maturity) -
  a subtotal, not a capital-protection guarantee and not conditioned on anything about OUTSTANDING
  launches, which haven't resolved either way yet.

A second, restricted report covers only the **completed-tenor cohort** — launches whose nominal
maturity date is on or before `DATA_AS_OF` — and gives the three resolved-outcome percentages
(AUTOCALL / MATURITY_CASH / PHYSICAL_DELIVERY) for that cohort alone, with OUTSTANDING launches
excluded entirely. A launch that already autocalled early but whose nominal maturity is still in
the future is *also* excluded from this cohort - counting it here while its still-open siblings
stay excluded would bias the comparison toward "resolved" outcomes.

For each bucket: a count, and the count as a percentage of its population (all launches, or the
completed-tenor cohort, depending on the report).

### Why the PHYSICAL_DELIVERY rate can be much lower than the raw "finishes below strike" rate

The raw "P(finishes below strike at maturity, ignoring autocall)" (= `RC.py` on the same terms) and the
PHYSICAL_DELIVERY percentage look at the same underlying data but answer different questions, and
the gap between them can be large for a trending underlying. The mechanism: **autocall removes
paths from the maturity-tested pool before they get a chance to breach** — of the windows that
would have finished below strike, many touch the trigger at some earlier observation on the way
down (or up, before turning down) and get redeemed at par before that decline plays out; once
autocalled, a note is no longer exposed to what the underlying does afterward. Only the minority
that finish below strike *and never once touched the trigger* end up as PHYSICAL_DELIVERY. A low
PHYSICAL_DELIVERY number next to a much higher "ignoring autocall" number is this mechanism
working as intended, not a bug.

## By launch-year cohort: don't trust the one blended headline number either

The headline numbers average every launch across the whole tested period into one figure - and
that blending can hide a genuinely bad stretch almost entirely, or make a recent stretch that
hasn't finished resolving look worse than it is. So every product's report includes a **by
launch-year cohort** breakdown (console table plus a 100%-stacked bar chart, one bar per calendar
year the note could have launched in, now with an explicit OUTSTANDING segment for years too
recent to have fully resolved) right after the headline numbers - that per-year breakdown is what
to actually look at, not the single blended figure.

The cohort table's last column, **AvgUnderlyingRet**, is the underlying's OWN worst-of return
(not the note's payoff), averaged across that year's launches. It is measured at maturity for a
resolved launch and as of the data cutoff for an OUTSTANDING one. It is there so a
suspicious-looking outcome mix can be checked against the real data instead of taken on faith. An
all-autocall year will usually show a positive average, but not always: an autocalled note can
still be down by maturity. If a year looks off, check that year's launches in the audit CSV.

## SEPARATE CAVEAT: these are also not independent trials

This is a second, distinct issue from the regime/survivorship biases above — even setting aside
*which* historical period or underlying you picked, the launches within that one backtest run
aren't independent of each other either, and the console output prints a caveat every run for it.
A long launch period's worth of **daily** launches over a `TENOR_MONTHS`-long note means adjacent
launches share nearly their entire forward window — launching on a Tuesday and launching on the
following Wednesday are almost the same bet. A "6,290 launches, 65.7% autocalled" result is not
6,290 independent draws from some underlying probability distribution; the effective sample size
is much smaller than the raw launch count, closer to (launch-period-years / tenor-years)-ish
*genuinely* distinct windows, heavily oversampled and interpolated between. That dependence means
the headline percentage should be read as a description of what this one particular historical
path produced under daily rolling launches - **not** a statistically independent frequentist
probability estimate, and it says nothing about paths the underlying didn't take. Overlap by
itself doesn't mechanically push the number up or down in either direction; it just means the
by-launch-year cohort table is a more informative guide to how much the outcome mix actually
varies than the one blended number is. Compare against `Estimators/P(Breach) Estimator`, which
instead fits a model to the underlying's own historical behavior and *simulates* forward - a
genuinely different, and less path-dependent-on-one-realization, way of estimating likelihood.
Don't quote this script's percentages as if they were that kind of number.

A second, smaller version of the same caveat: results are sensitive to exactly when the backtest
window happens to end (`DATA_AS_OF`, or "the most recent complete trading day" if left as `None`)
- a script run today and the same script run six months from now will have rolled the window
forward, resolved some previously-OUTSTANDING launches, and can shift the mix of outcomes.

## Launch period, data cutoff, and product terms

`LAUNCH_START` / `LAUNCH_END` / `DATA_AS_OF` (top of `FCN.py`) explicitly bound the run:

| Setting | Meaning |
|---|---|
| `LAUNCH_START` | first candidate launch date (inclusive) |
| `LAUNCH_END` | last candidate launch date (inclusive); `None` -> resolves to `DATA_AS_OF` |
| `DATA_AS_OF` | price-data cutoff (inclusive); `None` -> resolves to the most recent COMPLETE calendar day before now, so an in-progress trading session's close is never used |
| `MAX_GAP_DAYS` (in `../_backtest.py`, shared by all products) | calendar-day tolerance for rolling a scheduled observation date forward to an actual trading day before it's flagged as an unresolved data gap instead of an ordinary holiday roll |

Every trading day the fetched data actually covers within `[LAUNCH_START, LAUNCH_END]` is launched
and classified - not just the subset whose full tenor happens to already be observable. A launch
whose nominal maturity is still in the future relative to `DATA_AS_OF` is reported as OUTSTANDING
rather than silently dropped from the population (see "The four outcomes reported" above).

The note's terms (also top of `FCN.py`), all levels as a fraction of each underlying's own launch
level:

| Setting | Meaning |
|---|---|
| `TICKERS` | 1 = single underlying; 2+ = worst-of basket |
| `STRIKE` | conversion strike; checked ONLY at maturity (no interim monitoring - see above) |
| `AUTOCALL_TRIGGER` | worst-of level that triggers early redemption |
| `TENOR_MONTHS` | note life in calendar months |
| `AUTOCALL_FREQUENCY_MONTHS` | observation spacing (1 = monthly, 3 = quarterly, ...); `None` disables autocall |
| `AUTOCALL_LOCKOUT_MONTHS` | nonnegative integer; months before the first autocall observation is even checked - the first observation actually falls at `LOCKOUT + FREQUENCY`, not at `LOCKOUT` itself (e.g. lockout 0 / quarterly -> first observation month 3; lockout 3 / quarterly -> first observation month 6, skipping month 3 entirely) |

An observation date is always strictly after launch (never a "month-zero" autocall - enforced
structurally, see `OBS_MONTHS`) and must resolve strictly before the note's actual
maturity valuation date to count as an early-autocall check; one that resolves on or after that
date is excluded from the early-autocall test and left to the ordinary terminal test at maturity
instead (see `classify` in `FCN.py`).

## Scope and limitations

- **Regime dependence and single-name survivorship bias — see "Known biases" at the top of this
  README.** This is the dominant limitation of this script, more so than any of the items below,
  and should be read before quoting any of its output.
- **Not independent trials — see the SEPARATE CAVEAT section above.** Overlapping daily launches
  within one run are highly dependent on each other; treat the effective sample size as much
  smaller than the raw launch count.
- **Final valuation date, not settlement date.** Every date this script reports (autocall date,
  final valuation date) is the date the relevant price is *read*, not a settlement date - a real
  note's cash or physical delivery settles some number of business days later per its term sheet,
  and that lag is not modeled here.
- **Price data is un-adjusted `Close` (split-adjusted, NOT dividend-adjusted)** - a deliberate
  choice (a term sheet's reference price is normally the exchange's own closing print, not a
  total-return index level), not a default left unexamined. This script does not model whatever
  bespoke "Potential Adjustment Event" provisions a real term sheet would specify for extraordinary
  corporate actions (special dividends, spin-offs, etc.) - verify against the actual term sheet
  before treating a specific historical episode as fully representative.
- **No coupon economics.** A real FCN's fixed coupon is paid regardless of these capital outcomes
  and isn't valued or accumulated here — this script only classifies what happens to *principal*.
  Total return (capital outcome + coupons received up to that point) is out of scope, consistent
  with coupon valuation being out of scope repo-wide (see root `README.md`).
- **No discounting / time value.** An autocall in month 2 and one in month 11 are both just counted
  as "AUTOCALL" — the time-value difference between getting your capital back early vs. late isn't
  scored.
- **Flat autocall trigger.** Real term sheets sometimes step the autocall trigger down over the
  note's observations; this script uses one constant `AUTOCALL_TRIGGER` for every observation date.
- **No knock-in barrier.** This is deliberate, not a simplification — a plain FCN doesn't have one;
  the strike is only ever compared to the worst-of level at maturity. A note WITH a continuously
  monitored barrier is a different product, already modeled (single-name, cash-settled, one fixed
  `ENTRY_DATE`) in `Product MtM/Yield/Barrier Reverse Convertible/`.
- **Worst-of basket correlation isn't modeled or varied** — the worst-of path is computed directly
  from the basket's actual joint historical behavior, which is exactly the point of a real
  historical backtest, but means results for a basket only ever reflect the correlation regime that
  basket happened to realize over the tested launch period, not a range of possible correlation
  scenarios. A basket also drops any date where at least one ticker has no price (mismatched
  exchange calendars); the console reports how many dates that affected.
- **Observation-to-trading-day mapping tolerance.** A scheduled observation date more than
  `MAX_GAP_DAYS` calendar days from the nearest actual trading day is flagged as an
  unresolved data gap rather than evaluated - see `map_obs` in `../_backtest.py`. This matters most
  for thinly-covered or basket tickers; a continuously-listed single name like the current PFE
  default rarely triggers it in practice.

## Usage

```
python3 "FCN.py"
```

The per-product scripts (`FCN.py`, `../RC/RC.py`, `../BRC/BRC.py`, `../BSF/BSF.py`) only hold their
terms and their `classify` rule. Everything shared lives once in `../_backtest.py`: the yfinance
fetch, the gap-tolerant date mapping, the rolling launch loop, the report and the charts. Each run:

1. Runs `self_test()`: synthetic worst-of paths covering each outcome, the inclusive strike/trigger
   boundaries, no month-0 autocall, an observation colliding with maturity, and a gap-skipped
   observation. The script aborts on the first failed assert, before touching real data.
2. Fetches `TICKERS` from `yfinance`, trimmed to `DATA_AS_OF`. There is no FRED fallback.
3. Prints the overlapping-launch caveat, the all-launches and completed-tenor-cohort breakdowns, and
   the by-launch-year table.
4. Saves a per-launch audit CSV (`Launch`, `Outcome`, `Completed`, `UnderlyingReturn`) and two
   charts: `AMD FCN.png` (outcome mix by launch year) and `AMD FCN - growth.png` (price path with
   one dot per launch, colored by outcome).

Charts and CSVs are ad hoc, run-specific output (like `Estimators/`, not like `Product MtM/`) and
are gitignored.

**`FCN Python-in-Excel.py`** (same folder) is the body of an Excel `=PY()` cell, fed by a FactSet
price pull. It applies the same rule as `classify` here and was checked to match it launch by launch.
