# P(Breach) Estimator

A general-purpose **real-world (physical-measure) probability estimator**: what's the actual
likelihood that an underlying — or a worst-of basket of underlyings, as in a Multi-FCN or Multi-RC
— breaches a given level by a given date? It's not tied to any one product structure; it's a
standalone tool for asking "how likely is this move, really?" for any ticker or basket. This is a
genuinely different question from everything in `Product MtM/`, which prices products under
risk-neutral dynamics. It's not a pricer, and it doesn't belong alongside the pricing scripts for
that reason. This file fits a model to how each underlying has actually behaved historically (its
own real average return, its own real volatility pattern) and simulates forward from that — no
risk-free-rate assumption, no risk-neutral comparison, and no claim of being a fair-value or
pricing check.

## Model selection: ARMA-GARCH-family, chosen by information criterion

18 candidates (3 mean specs × 6 volatility specs, all with Student's t errors) plus two
Markov-switching candidates are fit and compared by AIC and BIC every time the model is
(re-)estimated (9 of the 18 in single-asset mode only — see "Single-asset vs. basket mode" below
for why three volatility specs are basket-incompatible):

- **Mean**: Zero, Constant, AR(1)
- **Volatility**: ARCH(1), GARCH(1,1), GJR-GARCH(1,1,1), EGARCH(1,1,1), APARCH(1,1,1),
  FIGARCH(1,1)
- **Regime-switching**: 2-regime and 3-regime Markov-switching (Gaussian, switching mean and
  variance)

Widening this grid trades off directly against cost: walk-forward validation refits the *entire*
grid from scratch at every `REFIT_FREQUENCY_DAYS` checkpoint, so the total fitting cost scales
linearly with grid size — doubling the volatility specs roughly doubles validation runtime.

**AIC vs. BIC vs. "SBIC"**: BIC (Bayesian Information Criterion) and SBIC (Schwarz BIC) are the
same statistic — Schwarz (1978) derived it, so the two names refer to one formula
(`-2·loglik + k·log(n)`), not two different criteria. AIC (`-2·loglik + 2k`) penalizes extra
parameters less harshly and is known to be more overfitting-prone for time-series order selection
at sample sizes like this. This file selects the winner by BIC but always prints the full
comparison table and flags it explicitly if AIC would have picked a different model.

**"GJR" vs. "TGARCH"**: the asymmetric/leverage candidate here uses `arch`'s GJR-GARCH
parameterization (Glosten-Jagannathan-Runkle 1993 — a `gamma·ε²·1(ε<0)` term in the variance
recursion), not Zakoian's (1994) TGARCH (which models conditional standard deviation with
absolute-value terms). The two capture the same "bad news raises vol more than good news" effect
but are genuinely different parameterizations — this file is explicit about using GJR
specifically.

**The other three additions**:
- **ARCH(1)** is GARCH(1,1) with the lagged-variance term dropped (`β` forced to 0) — a simpler,
  more parsimonious baseline that BIC will pick over GARCH(1,1) when the extra `β` parameter isn't
  earning its keep.
- **APARCH(1,1,1)** (Ding, Granger & Engle 1993) generalizes GARCH/GJR/TGARCH into one family by
  *estimating* the power term `δ` (instead of fixing it at 2, i.e. working in `|ε|^δ` space) on
  top of the same `γ·1(ε<0)` asymmetry term — genuinely more flexible, at the cost of one more
  estimated parameter.
- **FIGARCH(1,1)** (Baillie, Bollerslev & Mikkelsen 1996) allows *fractionally* integrated
  volatility persistence (a long-memory `d` parameter between 0 and 1, vs. GARCH's implicit
  integer-order persistence) — useful if a name's volatility shocks decay noticeably slower than
  a standard GARCH's exponential decay implies.

**Distribution** is fixed at Student's t throughout (fat tails are an essentially undisputed
feature of daily equity returns) rather than searched over, to keep the grid a comprehensible
mean × variance search.

**VAR/VARMA** are not part of the grid — those jointly model several return *series'* own
conditional means together. In basket mode, this file instead fits each name's own best
univariate model independently and correlates the residual shocks across names (see "Basket
mode" below) — a deliberately simpler alternative chosen so the joint simulator stays
hand-verifiable rather than reaching for multivariate machinery this repo has no library support
for.

## Single-asset vs. basket mode

One ticker in `TICKERS` runs single-asset mode. Two or more switches automatically to **basket
(worst-of) mode** — `min_i(S_i/S0_i)` vs. `STRIKE`, the same worst-of convention as `Multi-RC`
and `Multi-FCN` (`Product MtM/Yield/Reverse Convertible/Multi-RC/`,
`Product MtM/Yield/Fixed Coupon Note/Multi-FCN/`). Try `TICKERS = ["AAPL", "JPM", "XOM"]`.

**Single-asset**: forward simulation uses `arch`'s own native `.forecast(method="simulation")`
for every GARCH-family winner (no hand-rolling needed — `arch` handles all six volatility specs'
simulation natively, including APARCH's power-transformed recursion and FIGARCH's truncated
ARCH(∞) expansion), or a hand-rolled simulator for Markov-switching winners (statsmodels has no
built-in `.simulate()` for regime-switching models — see `simulate_markov_forward`).

**Basket mode**: each name gets its own independently best-fit model (by BIC, from a candidate
grid restricted to `ARCH(1)`, `GARCH(1,1)`, and `GJR-GARCH(1,1,1)` — see `BASKET_INCOMPATIBLE_VOL_SPECS`
below for why `EGARCH`, `APARCH`, and `FIGARCH` are single-asset only; different names can still
land on different specifications within that restricted set). What ties them together is the
**correlation of the standardized shocks** — each name's return, once its own conditional
mean/vol is divided out, is correlated with every other name's same-day shock via a real,
estimated correlation matrix. This is a Constant Conditional Correlation ("CCC-GARCH", Bollerslev
1990) joint model — simpler than a full DCC/BEKK multivariate GARCH (no mature library support for
that in Python — see "Are there libraries that make this easier?" below — and hand-rolling it in
full wasn't worth the risk), but a genuine, standard way to couple separately-fit conditional
variance processes.

Because there's no way to inject externally-correlated shocks into `arch`'s own
`.forecast(method="simulation")` (its random draws are entirely internal), basket mode instead
hand-rolls a **one-step variance recursion** directly from each name's already-fitted parameters,
driven by Cholesky-correlated shocks. That hand-rolled recursion only implements the
`ω + α·ε² [+ γ·ε²·1(ε<0)] + β·σ²_prev` form shared by ARCH/GARCH/GJR-GARCH (ARCH(1) is just this
with `β` forced to 0 — no separate code path needed). Those shocks are drawn as correlated
standard normals and then remapped, name by name through a Gaussian copula, onto each GARCH-family
name's own fitted Student's t marginal (its own fitted `nu`, rescaled to unit variance) —
preserving each name's fat tails, at the cost of an approximate (copula, not exact multivariate-t)
joint tail dependence. `BASKET_INCOMPATIBLE_VOL_SPECS` (`EGARCH`, `APARCH`, `FIGARCH`) are excluded
from the basket candidate grid for the same reason each of them isn't in that shared formula:
EGARCH's recursion is in log-variance space, APARCH's is in `|ε|^δ` space with its own estimated
power `δ`, and FIGARCH's is parameterized entirely differently (`φ`, `d`, `β`, no `α`/`γ` at all)
— hand-rolling three more one-step recursions was judged not worth the added implementation risk
for comparatively little gain. The Markov-switching branch similarly keeps each name's own regime
draws independent even in basket mode, using its own Gaussian shocks (matching how
`MarkovRegression` itself was fitted, rather than the copula) — both deliberate scope cuts made so
the joint simulator can actually be checked against the single-asset engines, not trusted on
faith.

## What's held fixed vs. what's simulated

Only the **fitted model parameters** are frozen across a simulation batch - the variance-recursion
coefficients, mean coefficients, and Student's t degrees of freedom (§2 in MATHEMATICS.md), plus,
in basket mode, the shock correlation matrix (§4) - all held at whatever the most recent fit (or
walk-forward refit) produced, never re-estimated mid-simulation.

Everything else evolves stochastically, path by path:

- **Spot** - the cumulative output of simulated daily returns, obviously.
- **Volatility** - genuinely simulated, *not* held flat. Each simulated day updates the conditional
  variance from that path's own previous simulated shock via the fitted GARCH recursion (see
  MATHEMATICS.md §5) - a path that draws a large early shock runs hot in variance for the rest of
  that path, a calm path stays calm. This is the opposite of `Product MtM/`'s single-name products,
  which hold one flat vol number for a note's entire life (see that folder's README) - flattening
  volatility here would defeat the entire point of fitting a variance-clustering model in the first
  place.
- **Regime** (Markov-switching candidates only) - which regime a path is in also evolves
  stochastically each simulated day, via the fitted transition matrix.

**Not modeled at all**: there is no interest rate or discounting anywhere in this file. Real-world
(physical-measure) probability estimation has nothing to discount - see this README's opening
section for why that's a deliberate difference from `Product MtM/`, not an oversight.

## `PRODUCT_TYPE`: what "breach" means

`PRODUCT_TYPE` picks which condition the model actually estimates:

- **`"FCN"` / `"RC"`** — European, terminal-only: `P(breach)` = `P(worst-of < STRIKE at maturity)`.
  Both this repo's plain Fixed Coupon Note and Reverse Convertible are a worst-of put struck at
  `STRIKE` and only ever looked at, at maturity, so the math is identical between the two —
  `PRODUCT_TYPE` just picks the label. `BARRIER` is ignored.
- **`"BRC"`** — a genuine down-and-in put, the same structure as `../../Product MtM/Yield/Barrier
  Reverse Convertible/`. The put only activates if `BARRIER` is ever touched (continuously
  monitored, daily, from `ENTRY_DATE` to maturity); `P(breach)` = `P(barrier ever touched AND
  worst-of < STRIKE at maturity)` — a genuinely path-dependent, joint condition, not just a
  terminal one. Requires `0 < BARRIER < STRIKE`. Barrier knock-in is treated as irreversible: once
  the REAL (not simulated) path has ever closed below `BARRIER` as of some rolling date, every
  later rolling date treats the put as already active regardless of what the simulated forward
  path does from there.
- **`"GENERIC"`** (default) — no product semantics assumed; reports the same two independent stats
  off the one `STRIKE` level this file has always reported: `P(breach at maturity)` and `P(ever
  touches STRIKE)` along the path. `BARRIER` is ignored.

Walk-forward validation treats each checkpoint as a fresh note starting at that checkpoint date
(same convention the rest of the validation already uses), so there's no "already knocked in"
state carried in from an earlier checkpoint the way there is for the live rolling estimate's single
continuous note. The naive historical baseline stays terminal-only even under `"BRC"` — a true
BRC-aware naive baseline would need to scan every historical window for an interim barrier touch,
not just compare endpoints; see `naive_historical_baseline`'s docstring.

## Rolling probability estimation

At each date along the real historical backtest path, using only information available up to
that date (the fitted model's parameters, which don't change between refits, plus every real
return realized since, which updates the conditional variance/regime state), the model simulates
forward to the fixed maturity date and reads off the empirical fraction of paths that breach —
both **at maturity** (the relevant number for a plain, barrier-free breach condition) and **ever**
along the simulated path (the more literal "hitting a strike" reading). Both are plotted.

`STRIKE` is fixed relative to the underlying's own **entry level** `S0`, but each simulation run from a
later rolling date produces returns relative to THAT date's spot, not `S0`. Before comparing a
simulated outcome to `STRIKE`, it's rescaled by `(spot at that rolling date) / S0` — so a name
that has already moved since entry is judged against its remaining distance to the strike, not
against a fresh full `STRIKE` move measured from wherever it happens to be today.

The horizon passed into the simulators is also converted from calendar days (`maturity_date -
date`, or `TENOR` years, expressed in calendar-day terms) to an equivalent **trading-day** count —
the models advance one fitted trading-day return per simulated step, so simulating a
calendar-day-sized number of steps would run roughly `365/252` too many days' worth of shocks per
path.

## Walk-forward validation: does it actually predict anything?

Fitting well (low AIC/BIC on training data) and predicting well (calibrated probabilities on data
the model has never seen) are different claims. `FULL_HISTORY_YEARS` (15) of real data are
fetched ending at `ENTRY_DATE`; the most recent `VALIDATION_YEARS` (8) are held out as a genuine
walk-forward test period — the model is never allowed to see a test-period return before it has
to predict across it, and the test period ends `TENOR` years before `ENTRY_DATE` so it never
overlaps the live forecast window either.

**"Self-learning"**: every `REFIT_FREQUENCY_DAYS` (252 trading days, ~1 year), the model is fully
**re-estimated** — not just re-filtered — on an expanding window of every real observation
available by that point, and the AIC/BIC search gets to pick a (possibly different) winner each
time, the same way a model gets periodically re-trained on newly arrived data. Between refits,
predictions at `EVAL_FREQUENCY_DAYS`-spaced checkpoints (21 trading days, ~1 month) use whichever
model was most recently refit, conditioned on real returns realized since (re-filtered, not
re-estimated — same mechanism as the live rolling estimate).

At each checkpoint, the currently-active model simulates forward `TENOR` years and the predicted
`P(breach)` is paired with the **actual** outcome — already known, since every checkpoint is far
enough in the past that its outcome is real history, not a forecast. A model with genuine
predictive skill should be **calibrated**: among all the times it said "70% chance," breaches
should have actually happened roughly 70% of the time.

- **Brier score** (mean squared error of the probability forecast — 0 is perfect, 0.25 is the
  "always guess 50%" baseline under class balance) summarizes this in one number, compared
  against a **naive baseline**: at each checkpoint, the empirical breach frequency among all
  `TENOR`-horizon returns observable using ONLY price history up to and including that checkpoint
  (`naive_historical_baseline`) — a forecast that could actually have been made at the time, not
  the mean of the validation set's own (still-future, as of that checkpoint) outcomes. A model
  that can't beat "just use the historical rate as of that point" isn't adding predictive value.
- **Calibration table**: predictions bucketed into quintiles, mean predicted probability vs.
  empirical breach frequency in each bucket.

This validation is single-asset only — the README's "Basket mode" scope cuts (independent regime
draws, no EGARCH) were made to keep the joint simulator hand-verifiable, but a joint,
basket-level walk-forward validation (jointly scoring the worst-of statistic's calibration,
not each name's own) is a natural extension not implemented here. In basket mode, this file
still validates each name's own univariate model independently and reports each.

## Product terms

| Term | Default | Meaning |
|---|---|---|
| `PRODUCT_TYPE` | `"GENERIC"` | `"FCN"` / `"RC"` (terminal-only), `"BRC"` (barrier + terminal, joint), or `"GENERIC"` (both stats, no product semantics) - see above |
| `STRIKE` | 90% | Breach level, as a fraction of each name's own entry level |
| `BARRIER` | `None` | `"BRC"` only - down-and-in knock-in level, as a fraction of entry level; must be `< STRIKE` |
| `TICKERS` | `["^GSPC"]` | 1 = single-asset, 2+ = worst-of basket |
| `ENTRY_DATE` / `TENOR` | 2025-01-02 / 1 year | The live forecast window |
| `FULL_HISTORY_YEARS` | 15 | Total real history fetched, ending at `ENTRY_DATE` |
| `VALIDATION_YEARS` | 8 | Most recent slice of that held out for walk-forward validation |
| `REFIT_FREQUENCY_DAYS` | 252 (~1y) | How often the model is fully re-estimated ("self-learning") |
| `EVAL_FREQUENCY_DAYS` | 21 (~1mo) | Prediction checkpoint spacing during validation |
| `N_SIMULATIONS` | 4,000 | Monte Carlo paths per rolling origin date |

## Output

Running `P(Breach) Estimator.py`:

1. Fetches `FULL_HISTORY_YEARS` of real daily closes ending at `ENTRY_DATE`.
2. Runs the walk-forward validation over the held-out `VALIDATION_YEARS`, printing each refit's
   selected model, a summary of out-of-sample predictions vs. actual outcomes, Brier scores
   (this model / naive), and a calibration table — then saves
   `P(Breach) Estimator - Validation (<ticker>).png` (a calibration scatter plus a Brier-score bar
   chart) per ticker. Any such chart left over from a PREVIOUS run (a different ticker) is deleted
   first, so only charts for the current `TICKERS` ever exist on disk.
3. Fits the **live** model — the last link in the same walk-forward chain, using all data through
   `ENTRY_DATE` — printing the full AIC/BIC comparison table.
4. Runs the sequential rolling Monte Carlo across the real `ENTRY_DATE`-to-maturity path, prints
   day-1 and final rolling probability estimates and what actually happened in this historical
   path.
5. Saves `P(Breach) Estimator.png`: each underlying's own return path, the rolling
   "P(breach at maturity)" and "P(ever touches strike)" estimates on the right axis, and the
   strike level.

## Output files aren't versioned

Every chart this script saves (`P(Breach) Estimator.png`, `P(Breach) Estimator - Validation
(<ticker>).png`) is ad hoc, run-specific output - this is a general exploration tool routinely
re-run against whatever ticker/basket/strike is currently of interest, not a fixed product with
one canonical chart. Its `.gitignore` excludes `*.png` for that reason; if you want to keep a
particular run's chart, copy it out or rename it before re-running.

## Usage

```
pip install -r ../../requirements.txt
python3 "P(Breach) Estimator.py"
```

Data comes from `yfinance` (Yahoo Finance), falling back to FRED for the S&P 500. Fitting relies
on the `arch` package (GARCH-family models) and `statsmodels` (Markov-switching).

## Are there libraries that make this easier?

- **Univariate GARCH-family fitting/simulation** — `arch` (already the sole dependency here) is
  the standard, well-maintained Python package for this, and is what makes all six volatility
  specs above a one-line `arch_model(..., vol=...)` call each; there isn't a more capable
  alternative in Python worth switching to.
- **The mean × volatility grid search itself** (looping every spec, fitting, comparing by
  AIC/BIC) has no off-the-shelf equivalent in Python analogous to `pmdarima.auto_arima` for ARIMA
  order selection — this hand-rolled loop (`fit_garch_family_candidates` / `select_best_model`) is
  the standard way this kind of search gets done, not a gap unique to this file.
- **True multivariate GARCH** (DCC-GARCH, BEKK, a genuine multivariate-t innovation) — Python's
  library support is thin. R's `rmgarch`/`rugarch` are the mature, standard choice for this. The
  current CCC-GARCH + Gaussian-copula-onto-fitted-t approach here is the pragmatic middle ground
  in pure Python: real, estimated correlation and each name's own genuine fat tails, without an
  R dependency.

## Naive GBM benchmark: `P(Breach) Estimator (Naive GBM).py`

This folder also has a deliberately **naive** companion script, `P(Breach) Estimator (Naive GBM).py`
- plain geometric Brownian motion with ONE constant `(mu, sigma)` per name, estimated once from a
`GBM_LOOKBACK_YEARS` (2y) trailing window, driving a single Monte Carlo batch straight from
`ENTRY_DATE` to maturity. It exists as a floor: the GARCH-based estimator above should be beating
this, not the other way round, and now there's a companion script that actually computes what that
floor is, instead of it being an abstract claim.

Same `PRODUCT_TYPE`/`STRIKE`/`BARRIER`/`TICKERS`/`ENTRY_DATE`/`TENOR` convention as the main file,
so the two are directly comparable on the same note. Differences:

- **Not sequential/rolling** - one simulation batch from `ENTRY_DATE`, not a walk over every
  historical date. No walk-forward validation, no refitting, no "self-learning" - just the single
  live estimate.
- **No volatility clustering, no fat tails** - constant `sigma` for the whole horizon (the thing
  the "What's held fixed vs. what's simulated" section above says GARCH deliberately does NOT do),
  Gaussian shocks instead of a fitted Student's t.
- **Closed-form cross-checks, single-asset only** - because GBM has an exact analytical solution,
  its own simulation can be checked against two continuous-monitoring closed forms (a lognormal
  terminal CDF, and a reflection-principle first-passage probability) - see MATHEMATICS.md section
  7. There's no equivalent closed form for a correlated worst-of basket, so basket mode skips this
  check.
- **Basket correlation needs no copula** - because GBM shocks are Gaussian to begin with, the same
  Cholesky-correlation trick used in the main file's basket mode works directly, with no
  Gaussian-copula remapping step required.

Run it the same way: `python3 "P(Breach) Estimator (Naive GBM).py"`.

## R companion: `P(Breach) Estimator.R`

This folder also has a full R port, `P(Breach) Estimator.R`, that exists for exactly one reason:
**genuine DCC-GARCH with a fitted multivariate Student's t** for the basket case, via `rmgarch` -
a real upgrade over this Python file's CCC-GARCH + Gaussian-copula approximation, not reachable
without an R dependency (see above). Same product terms, same STRIKE-relative-to-entry rebasing,
same calendar-to-trading-day conversion, same walk-forward-safe naive baseline - it answers the
same question with a better basket model. Differences from this file:

- **No Markov-switching** (this file has it off by default anyway - `INCLUDE_MARKOV_SWITCHING =
  False`). R's `MSGARCH` package would be the natural way to add it back.
- **No charts** - console output only.
- **Every basket name gets the full 6-spec volatility grid**, not just the 3 that are
  basket-compatible here (`BASKET_INCOMPATIBLE_VOL_SPECS` above) - `rmgarch`'s DCC framework fits
  each name's own univariate model independently, so it doesn't care what that name's own
  variance recursion looks like, unlike this file's hand-rolled one-step basket recursion.
- Walk-forward validation is single-asset only there too (same scope cut as this file).

Run it with `Rscript "P(Breach) Estimator.R"` (needs `rugarch`, `rmgarch`, `quantmod`, `xts` from
CRAN).
