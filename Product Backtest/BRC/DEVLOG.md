# BRC (Barrier Reverse Convertible) Python-in-Excel — devlog

Chronological decision history for `BRC Python-in-Excel.py`. See `Product Backtest/FCN/DEVLOG.md`
and `Product Backtest/RC/DEVLOG.md` for the shared Python-in-Excel conventions this reuses
(oversized price range, vertical Params list, `xl()` mechanics) - this file only covers what's
specific to BRC.

## Added, terminal payoff matched to Product MtM's existing BRC (2026-09-22)

Rather than invent BRC mechanics from scratch, matched the terminal payoff convention already
established in `Product MtM/Yield/Barrier Reverse Convertible/README.md`: barrier never touched ->
full principal back regardless of where the underlying ends up; barrier touched but finishes >=
STRIKE -> still full principal back; barrier touched AND finishes < STRIKE -> physical delivery.
No autocall (same as RC).

Added a monitoring-mode toggle the `Product MtM` version doesn't need (it only ever does daily/
"American" monitoring): European checks the barrier only at the single maturity observation;
American checks it on every trading day from launch through maturity (a running-min touch test,
closer to what `Product MtM`'s `path.cummin() <= barrier_level` does daily).

## Whitespace-stripped param labels (2026-09-22)

Same backport as `Product Backtest/RC/DEVLOG.md` describes - FCN hit a real `KeyError: 'Strike'` in
Excel, fixed there by stripping whitespace off Params labels before building the dict; applied here
too since BRC builds its Params dict identically. Re-verified against the stubbed-AMD test in both
monitoring modes - unchanged (European 72.5%/23.5%/4.0%, American 60.9%/35.1%/4.0%).

## Trimmed further, ~67 -> ~48 lines of code (2026-09-22)

Merged `map_obs`'s two-line `if` into one `return`, folded `pct`+`fmt_pct` into a single function,
combined several one-purpose assignment lines with commas/tuple-unpacking, and cut the extra blank
line between function defs. First attempt at this introduced two real bugs in the process - worth
recording since it's a reminder that "shortening" isn't risk-free: (1) called `xl("BRC!D1:E8")`
twice instead of storing it once, (2) `.set_index(0)` instead of `.set_index(prices_df.columns[0])`
- positional `0` doesn't work once headers=True gives the column a real name like "Date", so this
would have raised `KeyError: 0`. Caught both by re-running the same European/American verification
before considering it done - re-ran, matched the pre-trim numbers exactly (72.5%/23.5%/4.0% and
60.9%/35.1%/4.0%). Lesson: re-verify after a "just shortening, logic unchanged" edit same as any
other change - "unchanged" is a claim, not a guarantee, until it's actually been checked.

Verified three ways:
  1. Internally - MATURITY_CASH + PHYSICAL_DELIVERY + OUTSTANDING sum to 100% in both modes.
  2. Monitoring sanity - American shows more delivery than European on the same AMD data/params
     (36.5% vs 24.5% completed-cohort) - continuous monitoring can only touch the barrier at least
     as often as an endpoint-only check, never less.
  3. Degenerate-case cross-check - with STRIKE=1.00, European monitoring collapses to exactly "did
     the worst-of finish below BARRIER at maturity", since touching the barrier at maturity already
     implies finishing below STRIKE=1.00. At BARRIER=0.70 this matched FCN.py's own
     `diagnostic_stats` "P(finishes below strike at maturity, ignoring autocall)" for AMD at 70%
     (24.487%) to the decimal - the same degenerate-limit-check idea this repo's MATHEMATICS.md
     files use for QuantLib products, applied here to a compact reimplementation instead.
