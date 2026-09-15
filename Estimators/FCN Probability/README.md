# FCN Probability Estimator

A **real-world (physical-measure) probability estimator**: what's the actual likelihood that an
underlying — or a worst-of basket of underlyings, as in a Multi-FCN — breaches a strike by
maturity? This is a genuinely different question from everything else in `Product MtM/`, which
prices products under risk-neutral dynamics. It's not a pricer, and it doesn't belong alongside
the pricing scripts for that reason - see "What the 'risk-neutral' number actually means" below.

## What the "risk-neutral" number actually means

**This file's main number** (the real-world estimate) is a genuine forecast: fit a model to how
this stock has actually behaved historically (its own real average return, its own real
volatility pattern), then simulate forward. Straightforward.

**The risk-neutral number is a different kind of thing entirely — not a forecast, not a
probability anyone believes, and not derived from any real market price.** It answers one
narrow, artificial question: *"If — purely as a mathematical assumption, not because anyone
thinks it's true — this stock grew at exactly the risk-free rate instead of its real historical
average return, what fraction of simulated paths would end below the strike?"*

**A concrete example**, with made-up round numbers to isolate just the one thing that changes:
say a stock has 40% annualized volatility (its own real, historical number) and you're asking
about a 35%-drop barrier over 6 months.
- Assume it grows at the risk-free rate, ~4%/year → some probability, say ~15%, of ending below
  the barrier.
- Assume instead it grows at its own real historical average, say 15%/year → a noticeably LOWER
  probability of ending below the barrier, because on average it's starting from further above
  the barrier by the time 6 months pass.

Same stock, same volatility, same barrier — the only thing that moved is which growth rate you
plugged in. That's the entire difference between "risk-neutral" and "real-world" here.

**Why would you ever plug in a growth rate you don't believe?** Because of a specific result from
option-pricing theory (unrelated to what this file does or forecasts): if a bank sells you a
derivative, it can hedge itself by continuously trading the underlying stock plus a risk-free
bond. That hedge works — and produces one specific, arbitrage-free price — regardless of what the
bank or anyone else actually believes the stock's real growth rate is. It turns out the price that
hedge produces is mathematically identical to "pretend the stock grows at the risk-free rate, and
compute the expected payoff." That's the entire content of "risk-neutral pricing" — it's a
consequence of the hedging argument, not a belief about the future. Every pricing script elsewhere
in this repo (`Product MtM/`) uses that same "pretend it grows at `r`" assumption because they're
actually computing a note's price, and that's what a real price has to satisfy.

**This file doesn't price anything, though** — it estimates a real-world probability. The
risk-neutral number is printed purely as a labeled contrast, to make visible how much of a gap
there'd be if this were treated as a pricing problem instead of a forecasting one. It's also worth
being clear that it isn't even a real, market-calibrated risk-neutral number: `risk_neutral_comparison()`
/ `risk_neutral_worst_of_comparison()` use this file's own realized historical volatility (not a
real quoted implied volatility — no options chain is fetched anywhere here), so treat it as "what
a textbook Black-Scholes formula would say using this stock's own historical volatility," not as
a real market fair-value check.

**What differs in the actual computation** (both draw from the same real price history — the
difference is entirely in what's DONE with it):

| | Real-world (this file's main estimate) | Risk-neutral (`risk_neutral_*` functions) |
|---|---|---|
| Growth rate assumed | Each name's own fitted historical average (GARCH mean) | The risk-free rate `r` (`RISK_FREE_RATE`), by assumption, not estimation |
| Volatility | Each name's own fitted, time-varying conditional variance (GARCH/GJR/EGARCH) | A single flat realized volatility number |
| Distribution | Student's t (fat tails, fitted `nu`) | Lognormal (plain GBM, closed-form) |
| Correlation (basket) | Real, estimated correlation of standardized shocks | Same real correlation matrix, used only so the comparison is apples-to-apples |

Because the growth-rate assumption is deliberately different (and the volatility modeling is
different too), there's no reason to expect the two percentages to match, and neither one checks
the other.

## Model selection: ARMA-GARCH-family, chosen by information criterion

Nine candidates (3 mean specs × 3 volatility specs, all with Student's t errors) plus two
Markov-switching candidates are fit and compared by AIC and BIC every time the model is
(re-)estimated:

- **Mean**: Zero, Constant, AR(1)
- **Volatility**: GARCH(1,1), GJR-GARCH(1,1,1), EGARCH(1,1,1)
- **Regime-switching**: 2-regime and 3-regime Markov-switching (Gaussian, switching mean and
  variance)

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
for GARCH-family winners (no hand-rolling needed — `arch` handles GARCH, GJR-GARCH, and EGARCH
simulation correctly and natively), or a hand-rolled simulator for Markov-switching winners
(statsmodels has no built-in `.simulate()` for regime-switching models — see
`simulate_markov_forward`).

**Basket mode**: each name gets its own independently best-fit model (by BIC — different names
can land on different specifications). What ties them together is the **correlation of the
standardized shocks** — each name's return, once its own conditional mean/vol is divided out, is
correlated with every other name's same-day shock via a real, estimated correlation matrix. This
is a Constant Conditional Correlation ("CCC-GARCH", Bollerslev 1990) joint model — simpler than a
full DCC/BEKK multivariate GARCH (no library support for that here, and hand-rolling it in full
wasn't worth the risk), but a genuine, standard way to couple separately-fit conditional variance
processes.

Because there's no way to inject externally-correlated shocks into `arch`'s own
`.forecast(method="simulation")` (its random draws are entirely internal), basket mode instead
hand-rolls the **one-step GARCH / GJR-GARCH variance recursion** directly from each name's
already-fitted parameters, driven by Cholesky-correlated shocks. This is specifically why EGARCH
is excluded from the basket candidate grid (its recursion is in log-variance space and adds real
implementation risk for comparatively little gain), and why the Markov-switching branch keeps
each name's own regime draws independent even in basket mode (only the continuous within-regime
shock is correlated across names) — both deliberate scope cuts made so the joint simulator can
actually be checked against the single-asset engines, not trusted on faith.

## Rolling probability estimation

At each date along the real historical backtest path, using only information available up to
that date (the fitted model's parameters, which don't change between refits, plus every real
return realized since, which updates the conditional variance/regime state), the model simulates
forward to the fixed maturity date and reads off the empirical fraction of paths that breach —
both **at maturity** (the number that actually matters for a plain, barrier-free FCN) and **ever**
along the simulated path (the more literal "hitting a strike" reading). Both are plotted.

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
  against two baselines: the **unconditional historical breach frequency** over the same test
  period (a model that can't beat "just use the historical average rate" isn't adding predictive
  value), and the **risk-neutral GBM** probability (included as a contrast, not a fair
  competitor — it isn't trying to forecast in the first place).
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
| `STRIKE` | 90% | Breach level, as a fraction of each name's own entry level |
| `TICKERS` | `["^GSPC"]` | 1 = single-asset, 2+ = worst-of basket |
| `ENTRY_DATE` / `TENOR` | 2025-01-02 / 1 year | The live forecast window |
| `FULL_HISTORY_YEARS` | 15 | Total real history fetched, ending at `ENTRY_DATE` |
| `VALIDATION_YEARS` | 8 | Most recent slice of that held out for walk-forward validation |
| `REFIT_FREQUENCY_DAYS` | 252 (~1y) | How often the model is fully re-estimated ("self-learning") |
| `EVAL_FREQUENCY_DAYS` | 21 (~1mo) | Prediction checkpoint spacing during validation |
| `N_SIMULATIONS` | 4,000 | Monte Carlo paths per rolling origin date |
| `RISK_FREE_RATE` | 4% (flat) | Used only in `risk_neutral_comparison()` |

## Output

Running `FCN Probability Estimator.py`:

1. Fetches `FULL_HISTORY_YEARS` of real daily closes ending at `ENTRY_DATE`.
2. Runs the walk-forward validation over the held-out `VALIDATION_YEARS`, printing each refit's
   selected model, a summary of out-of-sample predictions vs. actual outcomes, Brier scores
   (this model / naive / risk-neutral), and a calibration table — then saves
   `FCN Probability Estimator - Validation.png` (a calibration scatter plus a Brier-score bar
   chart) per ticker.
3. Fits the **live** model — the last link in the same walk-forward chain, using all data through
   `ENTRY_DATE` — printing the full AIC/BIC comparison table.
4. Runs the sequential rolling Monte Carlo across the real `ENTRY_DATE`-to-maturity path, prints
   day-1 and final rolling probability estimates, what actually happened in this historical path,
   and the risk-neutral comparison.
5. Saves `FCN Probability Estimator.png`: each underlying's own return path, the rolling
   "P(breach at maturity)" and "P(ever touches strike)" estimates on the right axis, and the
   strike level.

## Usage

```
pip install -r ../../requirements.txt
python3 "FCN Probability Estimator.py"
```

Data comes from `yfinance` (Yahoo Finance), falling back to FRED for the S&P 500. Fitting relies
on the `arch` package (GARCH-family models) and `statsmodels` (Markov-switching).
