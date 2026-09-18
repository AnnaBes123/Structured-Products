# Structured Products — working notes for Claude

A personal, AI-assisted learning project about structured-product valuation and real-world
probability modeling (see the root `README.md` for the full framing). The user directs and reviews
every change; nothing here is a validated pricing library or investment research. Keep that in mind
when phrasing outputs - hedge appropriately, don't overstate confidence in a number.

## The two-folder split - never blur these

- **`Product MtM/`** — risk-neutral pricing ("what is this worth today"). Each product is replicated
  as an option portfolio and priced with QuantLib / a CRR lattice / Monte Carlo along one real
  historical path.
- **`Estimators/`** — real-world (physical-measure) probability ("how likely is this to actually
  happen"). Fits a model to an underlying's own historical behavior and simulates forward. Not a
  pricer, not risk-neutral, not directly comparable to `Product MtM/` numbers.

A number from one folder is never a substitute for, or directly comparable to, a number from the
other, even when they share an underlying and a strike.

## Per-product folder convention (`Product MtM/<Category>/<Product Name>/`)

- `<Product Name>.py` — the model/replication + historical MtM chart.
- `Greek Sensitivity.py` (some have a second `Greek Sensitivity (Autocallable).py` variant).
- `README.md` — product framing, terminology, scope/limitations.
- `MATHEMATICS.md` (where the replication is non-trivial, e.g. barrier products) — full derivation
  and verification checks (parity, closed-form convergence, degenerate-limit checks).
- Shared helpers live in `Product MtM/_common.py` (data fetching, vol/rate proxies, QuantLib process
  construction, ZCB pricing) - add there instead of duplicating a helper into a new product folder.

`Estimators/<Tool Name>/` follows the same README (+ MATHEMATICS.md where relevant) pattern but
isn't tied to one fixed underlying - it's routinely re-run against whatever ticker/basket/strike is
currently of interest.

## QuantLib over hand-rolled formulas

Options with any real complexity (barriers/knock-in-out especially) are priced with QuantLib, not
hand-rolled closed-form or lattice code. This was a deliberate switch after hand-rolled barrier
formulas became impossible to independently validate - see the root README's "Why QuantLib?"
section. Plain vanilla Black-Scholes is still fine to hand-roll (e.g. via `_quantlib_process` +
`AnalyticEuropeanEngine`, which is itself QuantLib, not truly "hand-rolled" math).

## Chart (`.png`) versioning differs by folder - check before assuming

- `Product MtM/` charts are illustrative reference output and **are committed** (no `.gitignore`
  entry for them at the time of writing).
- `Estimators/` charts are ad hoc, run-specific exploration output and are **gitignored** (see e.g.
  `Estimators/P(Breach) Estimator/.gitignore`) - don't be surprised if a chart there never shows up
  in `git status` as untracked, and don't assume a missing `.gitignore` elsewhere means the same
  convention applies.

## Cross-language companions: add, never replace

Where Python lacks a needed capability (e.g. no mature multivariate/DCC-GARCH library), a companion
script in another language is added **alongside** the Python original in the same folder - e.g.
`Estimators/P(Breach) Estimator/P(Breach) Estimator.R` exists solely for `rugarch`/`rmgarch`'s
genuine DCC-GARCH with a fitted multivariate Student's t, which nothing in Python's `arch` package
provides. The original Python file is never renamed, gutted, or repurposed into the new language -
both files stay, cross-reference each other in their header comments, and the shared README
documents both. When adding a feature to one, check whether its companion needs the same feature
(see the folder's own README for what's currently in sync vs. not).

## `Redundant (Past)/`

Superseded exploratory scripts, kept for history only. Not maintained, not held to the conventions
above, not a source of current numbers - skip unless specifically asked about earlier iterations.

## Running things

Every script under `Product MtM/` and `Estimators/` is standalone: `python3 "Script Name.py"` (or
`Rscript "Script Name.R"` for the R companion), fetching its own data from `yfinance` (falling back
to FRED for S&P 500/VIX series). `pip install -r requirements.txt` from a venv first; see each
folder's own README for exact commands and prerequisites (the R companion additionally needs
`rugarch`, `rmgarch`, `quantmod`, `xts` from CRAN).

## Documentation density is intentional here

This repo explains *why*, at length, in module docstrings, inline comments on non-obvious
subtleties, and dedicated README/MATHEMATICS.md sections - a deliberate departure from a terser
default, because the user is actively learning this material and needs to be able to independently
follow and verify the reasoning, not just trust the output. Match that density when extending
existing files in this repo; don't strip comments down to a terser house style.
