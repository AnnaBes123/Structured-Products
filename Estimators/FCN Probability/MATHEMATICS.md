# The Maths, Fully Decomposed

Every formula this script actually computes, in the order the pipeline uses them. This file is
the mechanical/mathematical companion to `README.md` (which covers the "why" and the product
framing) - here the goal is that nothing is left as an unexplained black box.

Notation: `r` = one asset's daily log return in **percent** (`log_returns_pct`: the script scales
by ×100 throughout, purely for `arch`'s numerical stability - all formulas below are in that same
percent scale unless stated otherwise). `t` indexes trading days. `i`, `j` index names in a basket.

---

## 1. Input: log returns

For each ticker's real closing price series `P_t`:

```
r_t = 100 · ln(P_t / P_{t-1})
```

This is the only transformation applied to raw price data before any model touches it.

---

## 2. The candidate model grid (single name)

Nine GARCH-family candidates = 3 mean specifications × 3 volatility specifications, every one
fit with Student's t-distributed innovations. Two more Markov-switching candidates are added when
`INCLUDE_MARKOV_SWITCHING = True` (off by default - see README for why). All fitting is done by
the `arch` / `statsmodels` packages via maximum likelihood; nothing below is hand-derived except
where explicitly marked "hand-rolled."

### 2.1 Mean specifications

| Name | Equation |
|---|---|
| Zero | `μ_t = 0` |
| Constant | `μ_t = μ` (one estimated constant) |
| AR(1) | `μ_t = c + φ₁·r_{t-1}` |

### 2.2 Volatility specifications

All three build a conditional variance `σ²_t`, i.e. `r_t = μ_t + ε_t`, `ε_t = σ_t·z_t`, and
`z_t` is the standardized innovation (Student's t below). Let `ε_{t-1}` be the previous day's
residual (`r_{t-1} - μ_{t-1}`).

**GARCH(1,1)**
```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```

**GJR-GARCH(1,1,1)** (Glosten-Jagannathan-Runkle 1993 - adds an asymmetric "leverage" term so a
negative shock raises variance more than a positive one of the same size)
```
σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·1(ε_{t-1} < 0) + β·σ²_{t-1}
```
(`1(·)` is the indicator function: 1 if the previous shock was negative, else 0.)

**EGARCH(1,1,1)** (Nelson 1991 - log-variance recursion, guarantees `σ²_t > 0` without
parameter constraints; fit and simulated entirely inside the `arch` package - this repo never
hand-rolls its recursion, see §5)
```
ln(σ²_t) = ω + β·ln(σ²_{t-1}) + α·(|z_{t-1}| - E|z_{t-1}|) + γ·z_{t-1},   z_{t-1} = ε_{t-1}/σ_{t-1}
```

### 2.3 Innovation distribution

Every candidate uses Student's t, standardized to unit variance, with the degrees-of-freedom
parameter `ν` itself estimated by maximum likelihood (not fixed):
```
z_t ~ standardized-t(ν),   ε_t = σ_t · z_t
```
Fatter tails than a Gaussian for any finite `ν`; `ν → ∞` recovers the Gaussian case. Fixed at t
throughout (not searched over) - see README.

### 2.4 Markov-switching candidates (k = 2, 3 regimes; off by default)

A regime `S_t ∈ {1,...,k}` follows a first-order Markov chain. Conditional on the regime, returns
are i.i.d. Gaussian with a regime-specific mean and variance:
```
r_t = c_{S_t} + σ_{S_t}·z_t,   z_t ~ N(0,1)
P(S_t = i | S_{t-1} = j) = T[i,j]        (the transition matrix, column-stochastic)
```
Fit via `statsmodels`' Hamilton-filter maximum likelihood (`MarkovRegression`); the filter and
smoother are library code, not hand-rolled here.

---

## 3. Model selection: log-likelihood, AIC, BIC

For each fitted candidate with `k` parameters, `n` observations, and maximized log-likelihood
`logL`:
```
AIC = -2·logL + 2k
BIC = -2·logL + k·ln(n)
```
The candidate with the **lowest BIC** is selected (`select_best_model`); AIC's pick is also
computed and printed for comparison, since AIC's lighter parameter penalty (`2k` vs. `k·ln(n)`,
and `ln(n) > 2` for any `n > 7`) makes it more prone to over-fitting for a search this size - see
README for the AIC/BIC/"SBIC" discussion.

---

## 4. Basket correlation: tying separately-fit names together

Each name in a basket gets its **own** independently best-fit model (§2). What couples them is
the correlation of each name's **standardized shock** on the same calendar day:

**GARCH-family name:**
```
z_t = ε_t / σ_t              (arch's own std_resid - the fitted model's own standardized residual)
```

**Markov-switching name**, weighted by the day's filtered regime probabilities `p_t(i)`:
```
z_t = (r_t - Σ_i p_t(i)·c_i) / (Σ_i p_t(i)·σ_i)
```

The basket's correlation matrix `Σ` is then just the real Pearson sample correlation of these
standardized-shock series across names, over the shared fit window (`estimate_shock_correlation`):
```
Σ_{ij} = corr(z^i, z^j)
```

This is the "CCC-GARCH" (Constant Conditional Correlation, Bollerslev 1990) approach: separate
univariate conditional-variance processes, coupled by one constant, real, estimated correlation
matrix of their standardized residuals - simpler than a full DCC/BEKK multivariate GARCH (no
library support here for that, see README).

To turn `Σ` into a way of drawing **correlated** random shocks, the script Cholesky-decomposes it:
```
Σ = L·Lᵀ
correlated_shock = independent_standard_normal_vector @ Lᵀ
```
so that each simulated day, every name's shock is drawn from an independent standard normal
vector rotated by `L`, giving draws with exactly the target correlation structure `Σ` (to Monte
Carlo sampling error).

**A real simplification worth being explicit about**: the shocks used to drive the *basket* joint
simulation (§5.3) are **Gaussian**, not Student's t - even though every underlying GARCH-family
model was itself fit with Student's t innovations. Implementing a correlated multivariate Student's
t draw (which needs its own shared degrees-of-freedom structure across names, not just a
correlation matrix) was judged not worth the added implementation risk for a basket-mode
simulator that's already a hand-rolled recursion (see README). The practical effect: single-asset
simulations (§5.1) keep the fitted fat tails exactly; basket simulations understate how fat the
*joint* tails really are, because the marginal fat-tailedness of each name individually is still
present in its own GARCH recursion, but a multivariate Gaussian copula pulls in the joint
extremes relative to a true multivariate-t copula. The risk-neutral worst-of comparison (§7) also
uses Gaussian shocks, which is correct there (a GBM assumption *is* Gaussian by construction) -
it's only an approximation in the real-world basket simulator.

---

## 5. Forward simulation: turning a fitted model into a breach probability

All simulation is **from a fixed origin date**, using only the model's already-fitted parameters
(never re-estimated mid-simulation) plus every real return realized up to and including that
origin date, to set the correct starting conditional variance / regime state / residual. The
`origin_date` is then simulated forward `horizon_days` trading days to maturity.

### 5.1 Single-asset, GARCH-family (`simulate_garch_forward`)

Uses `arch`'s own native `.forecast(method="simulation")` on a model truncated to end exactly at
`origin_date` (so the origin is the last available data point, and simulation starts from there
with the model's fitted `(μ, σ², ε)` state, drawing genuine Student's t innovations forward via
the recursion in §2.2). For each of `n_sims` simulated paths, the daily simulated returns
(percent) `r̂_1, ..., r̂_H` are turned into:
```
terminal_relative = exp( (Σ_{t=1}^{H} r̂_t) / 100 )                              # S_H / S_0
path_min_relative = exp( min_{h=1..H} (Σ_{t=1}^{h} r̂_t) / 100 )                  # min over the path
```
(a cumulative-sum-then-running-minimum, not a running-minimum-of-increments - the latter would be
wrong, since "worst point so far" is about the cumulative level, not the size of any one day's
move).

### 5.2 Single-asset, Markov-switching (`simulate_markov_forward`, hand-rolled)

`statsmodels` has no built-in forward simulator for `MarkovRegression`, so this is hand-rolled:

1. **Regime distribution today**: re-filter (not re-estimate) the already-fitted parameters
   through real data up to `origin_date`, giving filtered regime probabilities
   `p(S_{origin} = i)`. (Verified empirically that filtering an extended series with fixed
   parameters exactly reproduces the original fit's own filtered probabilities on the overlapping
   period.)
2. Draw each simulated path's starting regime from that distribution, then advance day by day
   using the real transition matrix `T` (§2.4) and regime-conditional Gaussian draws:
   ```
   S_t ~ Categorical(T[:, S_{t-1}])
   r̂_t = c_{S_t} + σ_{S_t} · z_t,   z_t ~ N(0,1)
   ```
3. Same `terminal_relative` / `path_min_relative` construction as §5.1.

### 5.3 Basket mode, GARCH-family (`simulate_basket_garch_forward`, hand-rolled)

Because `arch`'s own simulator draws its random innovations internally (no way to inject
externally-correlated shocks), basket mode hand-rolls the **one-step** GARCH/GJR-GARCH variance
recursion directly, driven by the Cholesky-correlated shock stream from §4:

Starting state at `origin_date` (`σ²_0`, `ε_0`, and the last return level for an AR(1) mean) comes
from truncating each name's own real return series to `origin_date` and re-fixing (not
re-estimating) its already-selected parameters on that truncated data - **this step is why the
0.00% bug happened and was fixed**: reading state straight off the original `last_obs`-restricted
fit left it `NaN` exactly at every origin date used here (see git history / session notes); the
current code truncates first, mirroring §5.1.

Then, for each day `t = 1..H`, for every name `i` simultaneously (using the shared correlated
shock stream `shock_t^i` from §4):
```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}  [+ γ·ε²_{t-1}·1(ε_{t-1}<0) if GJR]
μ_t  = μ_const  (or  c + φ₁·level_{t-1}  for an AR(1) mean)
r̂_t  = μ_t + σ_t · shock_t^i
ε_t  = r̂_t - μ_t
level_t = r̂_t
```
(EGARCH names are excluded from basket mode entirely - its log-variance recursion, §2.2, was
judged not worth hand-rolling here, so a basket containing an EGARCH-selected name falls back to
GARCH/GJR only within the candidate grid; see README.)

Per name, `terminal_relative` / `path_min_relative` as in §5.1. The basket's **worst-of** figure
is then simply the elementwise minimum across names, per simulated path:
```
worst_of_terminal = min_i( terminal_relative^i )
worst_of_path_min = min_i( path_min_relative^i )
P(breach at maturity)  = mean_over_paths( worst_of_terminal < STRIKE )
P(ever touches strike) = mean_over_paths( worst_of_path_min < STRIKE )
```

---

## 6. Realized (historical) annualized volatility

Used only as the volatility input to the risk-neutral comparison (§7), NOT anywhere in the
real-world model itself (which uses each name's own fitted, time-varying GARCH conditional
volatility instead):
```
σ_realized = std( ln(P_t / P_{t-1}) ) · √252
```
(Note this is computed from raw log returns, not the ×100-scaled series used for GARCH fitting -
`realized_annualized_vol` operates directly on prices.)

---

## 7. Risk-neutral comparison (a theoretical contrast, not a market-calibrated number)

See README for what this number does and doesn't mean. Mechanically:

### 7.1 Single name, closed-form (`risk_neutral_comparison`)

Plain Black-Scholes / GBM under the risk-neutral measure (drift forced to `r`, using realized
volatility `σ_realized` from §6 as the sole volatility input - there is no real implied
volatility available here):
```
d2 = [ln(S0 / K) + (r - q - ½σ²)·T] / (σ·√T)
P(breach) = P(S_T < K) = Φ(-d2)        (Φ = standard normal CDF)
```
With `S0 = 1` and `K = STRIKE` in this file's usage (everything is already expressed relative to
each name's own entry level), `q = 0` (no dividend yield input available), `T = TENOR` in years.

### 7.2 Basket worst-of, Monte Carlo (`risk_neutral_worst_of_comparison`)

No closed form exists for an N > 2 worst-of option under GBM (same reasoning as Multi-RC /
Multi-FCN's own risk-neutral pricing), so this is a plain correlated-GBM Monte Carlo, reusing the
SAME real correlation matrix `Σ` from §4 so the comparison against the real-world basket estimate
is apples-to-apples:
```
Σ = L·Lᵀ                                           (same Cholesky decomposition as §4)
z = independent_standard_normal_matrix @ Lᵀ         (n_sims × n_names, one draw per name per path)
S_T^i / S_0^i = exp( (r - ½σ_i²)·T + σ_i·√T · z^i )
worst_of = min_i( S_T^i / S_0^i )
P(breach) = mean_over_paths( worst_of < STRIKE )
```
This is a single terminal draw per path (no daily stepping needed - there's no path-dependent
GARCH state to propagate under plain GBM), so it's cheap to run at a larger `n_sims` (default
50,000) than the daily-stepping real-world basket simulator.

---

## 8. Walk-forward validation: scoring real predictive skill

At each of several past checkpoint dates (spaced `EVAL_FREQUENCY_DAYS` apart, using whichever
model was most recently refit as of `REFIT_FREQUENCY_DAYS`-spaced refit dates - see README for the
"self-learning" design), the model forecasts `TENOR` years forward (§5.1/5.2) to get a predicted
probability `p̂`, which is paired with the **already-known real outcome** `y ∈ {0, 1}`
(`1` = the underlying actually breached the strike by that checkpoint's own target date).

### 8.1 Brier score
```
Brier = mean( (p̂ - y)² )
```
0 is a perfect forecaster; 0.25 is what "always guess 50%" scores under class balance. Computed
for this model, for a naive baseline (`p̂ = ` the unconditional historical breach frequency over
the same test period, held fixed), and for the risk-neutral baseline (`p̂ = ` §7.1's number, held
fixed) - three numbers on the same real outcomes, so lower is a genuine, comparable measure of
predictive skill, not just goodness-of-fit.

### 8.2 Calibration table

Predictions are bucketed into quantiles of `p̂` (`pd.qcut`, 5 buckets by default), and each
bucket reports:
```
mean_predicted      = mean(p̂)         within the bucket
empirical_frequency = mean(y)          within the bucket
```
A model with real predictive skill should show `mean_predicted ≈ empirical_frequency` in every
bucket - the diagonal of a calibration plot.
