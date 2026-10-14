# Fixed Coupon Note — devlog

Chronological decision history for the scripts in this folder. Current-state explanation lives in
`README.md` / `MATHEMATICS.md`; this file is why things are the way they are.

## FCN MtM Checker added (2026-10-08)

**Trigger.** A live EWY autocallable FCN (strike 65%, trade date 2026-05-23, initial fixing ~178,
quarterly observations on the 19th, 6m) was being checked by setting `ENTRY_DATE = "2025-11-23"`,
`TENOR = 0.874` in `Fixed Coupon Note (Autocallable).py` so the window ended today. The chart
showed it autocalled, which it has not been. Three separate causes, all from forcing a live note
through a completed-window script:

- the entry date was a year early, so S0 was the 2025-11-24 close (90.30) instead of ~178, and
  every later date sat far above the trigger;
- observation dates are generated as even slices of `TENOR`, giving 2026-03-09 / 2026-06-24
  instead of the term sheet's dates;
- today was treated as maturity, so even with the right entry date the remaining life (and the
  model value) would have been wrong.

**Decision: a separate script, not a "live mode" flag.** `Fixed Coupon Note (Autocallable).py`
derives its path end, observation schedule and S0 from `ENTRY_DATE`/`TENOR`, and its
`fetch_index_path` deliberately errors on an in-progress window. A live note needs none of those
derivations - it needs explicit dates. Threading both through one `__main__` would have meant an
if/else on almost every line, so the checker is its own file with its own inputs.

**Engine imported, not copied.** `simulate_forward_price` already takes an arbitrary valuation
date, maturity and remaining-observation list, so the checker loads the sibling file with
`importlib` (its filename has spaces/parentheses, so a plain `import` is not possible) and calls it
with `strike`/`trigger` passed explicitly. Cost: the sibling's module-level
`fetch_risk_free_rate(ENTRY_DATE)` runs once on import. `finite_difference_greeks` is *not* reused
because it calls the engine without `strike`/`trigger`, i.e. it would silently use the sibling's
constants rather than the checker's - hence the small local `bumped_greeks`.
Rejected: copying the engine into the checker (what `Greek Sensitivity (Autocallable).py` does) -
a second copy to keep in sync.

**Inputs re-read per valuation date.** The completed-window scripts freeze vol and rate at entry.
For a mark, each date uses trailing 2y realized vol up to that date and that date's DGS1.
`fetch_risk_free_rate_series` was added to `_common.py` so this is one FRED call rather than one
per trading day. DGS1 is a 1y rate used for a <6m remaining life - kept for consistency with the
rest of `Product MtM/`; at this tenor the effect on the mark is a few bp.

**Initial fixing is pinned.** No EWY close around the trade date equals 178 (2026-05-19: 174.02,
05-20: 180.11, 05-22: ~182; 2026-05-23 is a Saturday), so deriving S0 from a close would have put
the trigger and strike in the wrong place. `INITIAL_FIXING = 178.0` is the user's approximate
figure and should be replaced with the exact term-sheet level.

**Assumed, not confirmed, for the EWY note:** trigger 100%; a single autocall observation on
2026-08-19; final fixing 2026-11-19 (redemption ~11-23); issuer spread left at the GS default.

**Verification (2026-10-07 close, EWY 183.69 = 103.2% of fixing).** 2026-08-19 observation: close
174.43 = 98.0% of fixing, not called - consistent with the note still being live. With no autocall
observation left the note is `ZCB - (1/strike) * put`; closed form 99.44% of par vs. Monte Carlo
99.44% (diff 0.001%). Delta and vega are ~0 because the 65% strike is ~59% below spot with 43 days
left; the -0.56% vs. par is discounting of the principal, not option risk.

## IMPLIED_VOL_OVERRIDE added (2026-10-09)

Single absolute vol level for the valuation-date mark, default `None` (trailing realized vol, as
before). Applied to the valuation date only: a quoted implied vol is an observation for that day,
and back-filling it over the history would misstate what the inputs were on earlier dates.
Rejected: a relative "vol shift" parameter - an absolute level covers both a real quote and a
what-if, with one input.

What-if at the 2026-10-08 close (realized 44.64%, base mark 99.45% of par): 53.4% vol (about x1.2)
-> 99.39%; 64.5% vol (about +20 points) -> 99.19%. Closed-form check passes at each level. The
mark barely moves because the 65% strike is far out of the money with six weeks left.
