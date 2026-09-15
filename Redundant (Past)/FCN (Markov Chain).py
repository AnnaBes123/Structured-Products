###python3 -m venv path/to/venv <- creates environment
###source path/to/venv/bin/activate <- activates environment - need to activate it next time env used
###python3 -m pip install xyz <- downloads library
#These libraries are used to read the knowledge:
# pip install pypdf
# pip install python-docx

# ---------------------------------------------------------------------------
# FCN Probability Editor -- Markov Chain
#
# Same note mechanics as "FCN (Statistical Baseline).py" (worst-of baskets,
# autocall, American/European strike observation, coupon schedule, tenor
# curve, drift decoupling) -- but a completely different mathematical engine
# underneath. The baseline files model the return process as GARCH (a
# continuous-state process with time-varying but smoothly evolving variance)
# and Markov-Switching (a continuous emission distribution with a small
# number of LATENT regimes inferred via filtering). This file instead fits a
# discrete-state, first-order MARKOV CHAIN directly on the stock's own
# historical daily returns:
#
#   1. Daily log returns are cut into k quantile bins ("states") -- e.g. with
#      k=4, roughly the worst quartile of days is state 0, the next quartile
#      state 1, and so on. Unlike Markov-Switching's regimes, these states
#      are DIRECTLY OBSERVED from realized returns, not inferred/filtered --
#      there is no latent-variable estimation step.
#   2. A one-step transition matrix P[i, j] = P(state_{t+1}=j | state_t=i) is
#      estimated by counting historical transitions between bins.
#   3. k (2 to 8) is chosen by BIC, trading off how finely the return
#      distribution is sliced against how much data supports each transition
#      count, among candidates that pass a usability check (rejects any k
#      where a state is nearly unvisited or has ~zero within-state variance).
#
# The probability of principal loss is estimated from two complementary
# angles, reported side by side, and BOTH the autocall and the maturity
# breach are evaluated on the same simulated Monte Carlo price paths (not a
# separate closed-form approximation). Both share the SAME fitted transition
# matrix and states -- they differ only in the shape assumed for returns
# WITHIN a state once the chain lands there:
#
#   A) Empirical Bootstrap -- each simulated day's return is resampled
#      (with replacement, via linear-interpolated inverse-ECDF) from the
#      ACTUAL historical returns that occurred in the state the chain just
#      transitioned into. Nonparametric: whatever skew/fat tails that state
#      really had historically come through exactly, no distributional
#      assumption.
#   B) Markov-t -- same transition matrix, but each state's returns are drawn
#      from a fitted Student's t (mean/std matched to the state's own
#      empirical moments, degrees of freedom fit per state by MLE) instead of
#      resampled directly. A parametric sensitivity check against A: if the
#      two disagree by a lot, the empirical engine's within-state shape (not
#      just its state dynamics) is doing real work. t, not Normal, because
#      daily equity returns are leptokurtic even conditional on a vol
#      regime/state -- a Normal-within-state engine would systematically
#      understate the odds of a large single-day move, understating exactly
#      the tail risk this tool exists to measure. t also isn't capped at the
#      worst day actually observed in a state the way the empirical bootstrap
#      is -- it can extrapolate tail scenarios beyond the historical sample.
#
# For a basket, both engines simulate all underlyings JOINTLY: each name's
# OWN state evolves independently via its own transition matrix (a full
# joint/shared-state model is out of scope, same simplification the
# baseline's Markov-Switching engine makes for regimes), but the within-day
# draw is cross-sectionally correlated via a Gaussian copula -- a correlation
# matrix is estimated from each name's own state-standardized residuals
# (aligned on common trading dates), and every simulated day draws a single
# correlated uniform vector that is mapped through each name's own
# state-conditional quantile function (empirical inverse-ECDF for engine A,
# fitted Student's t ppf for engine B) before being fed into its own transition-implied
# state mean/std. This preserves each name's individual within-state shape
# while inducing realistic co-movement across names -- exactly what a
# worst-of payoff is sensitive to.
#
# DRIFT_MODE: exactly as in the baseline -- the chain's own fitted state
# means are a trailing historical average and compounded over the tenor can
# badly bias the estimate, so the forward simulation's drift is decoupled
# from the fitted historical mean and replaced with one of:
#   "risk_neutral" (recommended here) -- drift = risk-free rate - dividend
#       yield, the textbook-correct assumption for pricing an embedded
#       option.
#   "zero"         -- simplest, maximally conservative driftless assumption.
#   "historical"   -- keep each state's own fitted mean (NOT recommended for
#       pricing -- reference only, shows what naive trend extrapolation
#       implies).
#
# VOL_MODE is fixed to "empirical" in this file -- the chain's own fitted
# within-state dispersion is used as-is, with no options-implied rescaling
# (kept as a named constant for parity with the baseline files' function
# signatures, not a real toggle here).
#
# Note on backtesting: a true walk-forward backtest of the horizon-ahead
# breach probability would only have a handful of independent windows in a
# few years of daily data, which isn't statistically reliable -- so it's
# deliberately not implemented here.
#
# How to use in VSCode:
#   1. Make sure the venv is selected as your Python interpreter
#      (Cmd+Shift+P -> "Python: Select Interpreter" -> venv).
#   2. Edit the variables in the CONFIG section below.
#   3. Click "Run Python File" (the play button, top right) or press
#      Ctrl+Alt+N / F5, depending on your setup.
#   4. Results print in the terminal; four chart windows open separately.
# ---------------------------------------------------------------------------

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")  # statsmodels' optimizers are chatty about convergence details

# ----------------------------- CONFIG ---------------------------------
TICKERS = ["MSFT"]      # one ticker = single name; several = worst-of basket
PERIOD = "5y"                    # how much history to pull: "1y", "2y", "5y", "10y", "max"
N_SIMS = 100_000                 # number of Monte Carlo paths for the headline probability + charts

# Set the strike EITHER as a % of today's spot OR (single-name only) as an
# exact price. Baskets always use STRIKE_PCT, applied to each name's own
# initial level (that's what "worst of, normalized" means).
STRIKE_PCT = 60                  # e.g. 70 means strike = 70% of current spot
STRIKE_PRICE = None              # e.g. 150.00; single-name only, overrides STRIKE_PCT if set

# --- Note structure ---
NOTE_TENOR_MONTHS = 12
TRADING_DAYS_PER_MONTH = 21      # approximation used to convert months -> trading days

AUTOCALL_ENABLED = True
AUTOCALL_BARRIER_PCT = 100        # % of (each name's own) spot; called if worst-of >= this
AUTOCALL_DELAY_MONTHS = 0         # 0 = no delay (first observation at month 1, the earliest
                                   # possible); 1-6 = skip that many additional monthly
                                   # observations on top of that. Must be < tenor.

COUPON_FREQUENCY = "monthly"    # "monthly" | "quarterly" | "semiannual" | "annual"
                                   # schedule only -- amounts need option pricing, not done here

# How the strike (put) is observed for paths that reach maturity uncalled:
#   "european" -- only the closing level AT maturity is checked (standard reverse
#       convertible / FCN put).
#   "american" -- the worst-of level is checked on EVERY trading day of the note's
#       life; if it ever falls below strike at any point, the put is treated as
#       exercised regardless of where it ends up at maturity. Strictly riskier than
#       "european" since there are far more chances to breach.
STRIKE_OBSERVATION = "european"   # "european" | "american"

# --- Forward-looking drift assumption (see header comment) ---
DRIFT_MODE = "historical"       # "risk_neutral" | "zero" | "historical"
RISK_FREE_RATE = 0.04             # annualized; used by "risk_neutral"
DIVIDEND_YIELD = 0.0              # annualized; used by "risk_neutral"

# Volatility level is always the chain's own fitted within-state dispersion in
# this file (no options-implied rescaling) -- kept as a named constant for
# parity with the other files' function signatures, not a real toggle here.
VOL_MODE = "empirical"

# Candidate number of discrete return-states to search; best picked by BIC.
N_STATES_GRID = [2, 3, 4, 5, 6, 7, 8]

# Probability-vs-tenor curve: sensitivity view of the underlying breach
# probability across tenors, ignoring autocall.
TENOR_GRID_MONTHS = [1, 3, 6, 12, 18, 24, 30, 36]
N_SIMS_TENOR = 3_000
# ------------------------------------------------------------------------

HORIZON_DAYS = NOTE_TENOR_MONTHS * TRADING_DAYS_PER_MONTH
TENOR_GRID_DAYS = [m * TRADING_DAYS_PER_MONTH for m in TENOR_GRID_MONTHS]
COUPON_MONTHS_PER_PAYMENT = {"monthly": 1, "quarterly": 3, "semiannual": 6, "annual": 12}
TRADING_DAYS_PER_YEAR = 252


def fetch_prices(ticker, period):
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data found for '{ticker}'. Check the ticker symbol.")
    close = data["Close"].dropna()
    close.name = "Close"
    return close.squeeze()


def resolve_drift(spot):
    """Returns (daily_drift_pct, label). daily_drift_pct is None only for
    DRIFT_MODE == 'historical', signaling 'keep each state's own fitted mean'."""
    if DRIFT_MODE == "risk_neutral":
        drift = (RISK_FREE_RATE - DIVIDEND_YIELD) / TRADING_DAYS_PER_YEAR * 100
        return drift, f"risk-neutral (rf={RISK_FREE_RATE:.1%}, div={DIVIDEND_YIELD:.1%})"

    if DRIFT_MODE == "zero":
        return 0.0, "zero (driftless)"

    if DRIFT_MODE == "historical":
        return None, "historical (fitted state means -- NOT recommended for pricing)"

    raise ValueError(f"Unknown DRIFT_MODE: {DRIFT_MODE!r}")


def resolve_vol_scale():
    """Returns (vol_scale, label). Always 1.0 in this file -- no options-implied
    rescaling -- kept as a function for parity with the other files' signatures."""
    return 1.0, "empirical (chain's own fitted within-state dispersion)"


def run_diagnostics(log_returns, label):
    """Check whether the series shows genuine first-order dependence -- i.e. whether
    a Markov chain is actually justified over just treating returns as i.i.d."""
    print("=" * 70)
    print(f"STEP 1 [{label}]: Diagnostics")
    print("=" * 70)

    adf_stat, adf_p, *_ = adfuller(log_returns, autolag="AIC")
    stationary = adf_p < 0.05
    print(f"ADF stationarity test:   stat={adf_stat:.3f}  p={adf_p:.4f}  "
          f"-> {'stationary' if stationary else 'NOT stationary'}")

    lb = acorr_ljungbox(log_returns - log_returns.mean(), lags=[10], return_df=True)
    lb_p = float(lb["lb_pvalue"].iloc[0])
    lb_autocorr = lb_p < 0.05
    print(f"Ljung-Box on raw returns (lag 10): p={lb_p:.4f}  "
          f"-> {'linear autocorrelation present' if lb_autocorr else 'no significant linear autocorrelation'}")

    up = (log_returns.values > 0).astype(int)
    table = np.zeros((2, 2), dtype=int)
    np.add.at(table, (up[:-1], up[1:]), 1)
    chi2_stat, chi2_p, *_ = stats.chi2_contingency(table)[:4] if table.min() > 0 else (np.nan, np.nan, None, None)
    dependence = bool(chi2_p < 0.05) if np.isfinite(chi2_p) else False
    if np.isfinite(chi2_p):
        print(f"Chi-square test, up/down transition independence: stat={chi2_stat:.3f}  p={chi2_p:.4f}  "
              f"-> {'first-order dependence detected (Markov structure justified)' if dependence else 'no significant first-order dependence'}")
    else:
        print("Chi-square test, up/down transition independence: skipped (a transition cell was empty)")

    if not (stationary and (lb_autocorr or dependence)):
        print("WARNING: diagnostics do not strongly support Markov-chain modeling for this series. "
              "Proceeding anyway, but treat the results with caution.")
    print()


def build_state_bins(log_returns_pct, k):
    """Quantile-based bin edges so each state has ~equal historical occupancy."""
    edges = np.quantile(log_returns_pct, np.linspace(0, 1, k + 1))
    edges = edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def assign_states(log_returns_pct, edges):
    return np.clip(np.digitize(log_returns_pct, edges[1:-1]), 0, len(edges) - 2)


def transition_counts(states, k):
    n = np.zeros((k, k))
    np.add.at(n, (states[:-1], states[1:]), 1)
    return n


def transition_matrix_from_counts(counts):
    row_sums = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)


def chain_loglik(counts, P):
    mask = counts > 0
    return float(np.sum(counts[mask] * np.log(P[mask])))


def chain_is_usable(states, k, log_returns_pct, min_occupancy=0.01, min_var=1e-8):
    """Reject a candidate state count where a state is nearly unvisited (unreliable
    transition estimates in/out of it) or has ~zero within-state variance (a
    degenerate bin, e.g. many identical zero-return days collapsed into one edge)."""
    values = log_returns_pct.values
    for s in range(k):
        mask = states == s
        occ = mask.mean()
        if occ < min_occupancy:
            return False
        if mask.sum() > 1 and values[mask].var() < min_var:
            return False
    return True


def select_markov_chain_model(log_returns_pct, label):
    """Grid-search the number of quantile-bin states, pick the best by BIC among
    fits that pass a usability check (see chain_is_usable)."""
    print("=" * 70)
    print(f"STEP 2 [{label}]: Markov chain state-count selection")
    print("=" * 70)

    candidates, rejected = [], 0
    for k in N_STATES_GRID:
        edges = build_state_bins(log_returns_pct, k)
        states = assign_states(log_returns_pct.values, edges)
        if not chain_is_usable(states, k, log_returns_pct):
            rejected += 1
            continue
        counts = transition_counts(states, k)
        P = transition_matrix_from_counts(counts)
        n_trans = len(states) - 1
        params = k * (k - 1)
        ll = chain_loglik(counts, P)
        bic = -2 * ll + params * np.log(n_trans)
        aic = -2 * ll + 2 * params
        candidates.append({"k": k, "edges": edges, "states": states, "aic": aic, "bic": bic})

    if rejected:
        print(f"({rejected} candidate(s) rejected as degenerate -- a near-unvisited "
              f"or near-zero-variance state)")

    if not candidates:
        raise RuntimeError(f"No usable Markov chain state count converged for {label}.")

    table = pd.DataFrame([{"states": c["k"], "AIC": c["aic"], "BIC": c["bic"]}
                           for c in candidates]).sort_values("BIC").reset_index(drop=True)
    print(table.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
    print()

    best = min(candidates, key=lambda c: c["bic"])
    print(f"Selected: {best['k']}-state chain (lowest BIC = {best['bic']:,.1f})")
    print()
    return best["k"], best["edges"], best["states"]


def fit_state_t_dof(values_in_state, min_obs=8, fallback_nu=200.0):
    """MLE degrees of freedom for a Student's t fit to one state's historical
    returns -- used only for tail SHAPE; loc/scale from the fit are discarded in
    favor of the state's own empirical mean/std so both engines agree on the
    dispersion they're simulating and differ only in tail shape. Falls back to a
    large nu (effectively Normal) when a state has too few observations for a
    3-parameter MLE to be trustworthy, or if the fit fails outright."""
    if len(values_in_state) < min_obs:
        return fallback_nu
    try:
        nu, _loc, _scale = stats.t.fit(values_in_state)
        return float(np.clip(nu, 2.1, fallback_nu))
    except Exception:
        return fallback_nu


def describe_markov_chain(k, edges, states, log_returns_pct, label):
    """Per-state mean/vol as direct (not filtered/smoothed) moments -- states here
    are observed outright from realized returns, not inferred -- plus the
    transition matrix, expected durations, each state's own historical return
    pool (for the empirical bootstrap engine), and each state's own fitted
    Student's t degrees of freedom (for the Markov-t engine)."""
    values = log_returns_pct.values
    means_pct = np.array([values[states == s].mean() for s in range(k)])
    stds_pct = np.array([values[states == s].std(ddof=0) for s in range(k)])
    occupancy = np.array([(states == s).mean() for s in range(k)])
    state_returns_sorted = {s: np.sort(values[states == s]) for s in range(k)}
    nus = np.array([fit_state_t_dof(values[states == s]) for s in range(k)])

    counts = transition_counts(states, k)
    P = transition_matrix_from_counts(counts)
    durations = 1.0 / np.clip(1 - np.diag(P), 1e-9, None)
    high_vol_idx = int(np.argmax(stds_pct))

    print(f"[{label}] {'state':>7}  {'ann. mean':>10}  {'ann. vol':>9}  {'occupancy':>10}  "
          f"{'persistence':>11}  {'exp. dur':>10}  {'t dof':>7}")
    for s in range(k):
        ann_mean = means_pct[s] / 100 * TRADING_DAYS_PER_YEAR * 100
        ann_vol = stds_pct[s] / 100 * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        tag = "  (high-vol)" if s == high_vol_idx else ""
        print(f"{'':>9}{s:>7}  {ann_mean:>9.1f}%  {ann_vol:>8.1f}%  {occupancy[s]:>9.1%}  "
              f"{P[s, s]:>10.1%}  {durations[s]:>7.0f}d  {nus[s]:>5.1f}{tag}")
    print()

    denom = np.where(stds_pct[states] > 0, stds_pct[states], np.nan)
    std_resid = pd.Series((values - means_pct[states]) / denom, index=log_returns_pct.index)
    high_vol_rolling = pd.Series((states == high_vol_idx).astype(float),
                                  index=log_returns_pct.index).rolling(21, min_periods=1).mean()
    state_vol_series = pd.Series(stds_pct[states], index=log_returns_pct.index) \
        / 100 * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

    return {
        "k": k, "edges": edges, "states": states, "P": P,
        "means_pct": means_pct, "stds_pct": stds_pct, "occupancy": occupancy,
        "state_returns_sorted": state_returns_sorted, "nus": nus, "current_state": int(states[-1]),
        "std_resid": std_resid, "high_vol_idx": high_vol_idx,
        "high_vol_rolling": high_vol_rolling, "state_vol_series": state_vol_series,
    }


def stationary_distribution(P):
    vals, vecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(vals - 1)))
    pi = np.real(vecs[:, idx])
    pi = np.clip(pi, 0, None)
    return pi / pi.sum()


def chain_unconditional_annualized_vol_pct(P, means_pct, stds_pct):
    """Law of total variance: within-state variance (stationary-weighted) plus
    between-state variance of the means, annualized."""
    pi = stationary_distribution(P)
    overall_mean = float(np.sum(pi * means_pct))
    within = float(np.sum(pi * stds_pct ** 2))
    between = float(np.sum(pi * (means_pct - overall_mean) ** 2))
    daily_var_pct = within + between
    return np.sqrt(daily_var_pct * TRADING_DAYS_PER_YEAR)


def build_correlation_matrix(std_resid_by_ticker, tickers):
    df = pd.DataFrame(std_resid_by_ticker)[tickers].dropna()
    corr = df.corr().to_numpy()
    corr = corr + np.eye(len(tickers)) * 1e-8  # tiny ridge for numerical stability
    return corr, len(df)


def empirical_ppf(sorted_arr, u):
    """Linear-interpolated inverse-ECDF: maps uniforms in (0,1) to the historical
    return distribution observed within a single state. Reduces to nearest-value
    lookup at the extremes; the u array itself is what carries cross-sectional
    correlation (a Gaussian copula rank), so this is what makes engine A's shocks
    correlated across tickers while staying fully empirical in shape."""
    m = len(sorted_arr)
    if m == 1:
        return np.full_like(u, sorted_arr[0])
    p = (np.arange(1, m + 1) - 0.5) / m
    return np.interp(u, p, sorted_arr)


def markov_chain_basket_monte_carlo(chain_infos, corr, tickers, horizon_days, n_sims, rng,
                                     drift_by_ticker, vol_scale_by_ticker, quantile_mode):
    """Each ticker's state evolves independently via its OWN transition matrix, but
    the within-day draw is cross-sectionally correlated via a Gaussian copula, so
    the worst-of payoff still sees realistic co-movement. quantile_mode='empirical'
    resamples actual historical returns from the landed-on state (engine A); 't'
    draws from a fitted Student's t (state's own mean/std, state's own fitted
    degrees of freedom) instead (engine B) -- everything else (state dynamics,
    correlation, drift decoupling) is identical between the two calls."""
    n = len(tickers)
    L = np.linalg.cholesky(corr)
    k = {t: chain_infos[t]["k"] for t in tickers}
    P = {t: chain_infos[t]["P"] for t in tickers}
    means = {t: chain_infos[t]["means_pct"] for t in tickers}
    stds = {t: chain_infos[t]["stds_pct"] for t in tickers}
    nus = {t: chain_infos[t]["nus"] for t in tickers}
    state_returns = {t: chain_infos[t]["state_returns_sorted"] for t in tickers}

    state = {t: np.full(n_sims, chain_infos[t]["current_state"], dtype=int) for t in tickers}
    cum_log_return = {t: np.zeros(n_sims) for t in tickers}
    path_log_return = {t: np.empty((n_sims, horizon_days)) for t in tickers}

    for day in range(horizon_days):
        z = rng.standard_normal((n_sims, n)) @ L.T
        u = np.clip(stats.norm.cdf(z), 1e-10, 1 - 1e-10)
        for i, t in enumerate(tickers):
            next_state = np.empty(n_sims, dtype=int)
            for j in range(k[t]):
                mask = state[t] == j
                if mask.any():
                    next_state[mask] = rng.choice(k[t], size=mask.sum(), p=P[t][j])
            state[t] = next_state

            if quantile_mode == "empirical":
                draw_pct = np.empty(n_sims)
                for j in range(k[t]):
                    mask = state[t] == j
                    if mask.any():
                        draw_pct[mask] = empirical_ppf(state_returns[t][j], u[mask, i])
                shock_pct = (draw_pct - means[t][state[t]]) * vol_scale_by_ticker[t]
            elif quantile_mode == "t":
                shock_pct = np.empty(n_sims)
                for j in range(k[t]):
                    mask = state[t] == j
                    if mask.any():
                        nu = nus[t][j]
                        t_std = stats.t.ppf(u[mask, i], nu) / np.sqrt(nu / (nu - 2))
                        shock_pct[mask] = t_std * stds[t][j] * vol_scale_by_ticker[t]
            else:
                raise ValueError(f"Unknown quantile_mode: {quantile_mode!r}")

            mean_component = drift_by_ticker[t] if drift_by_ticker[t] is not None else means[t][state[t]]
            total_pct = mean_component + shock_pct
            cum_log_return[t] += total_pct / 100.0
            path_log_return[t][:, day] = cum_log_return[t]

    return {t: np.exp(path_log_return[t]) for t in tickers}


def worst_of_index(paths_by_ticker, tickers):
    """Elementwise minimum of normalized paths across tickers, on a par-100 basis.
    With one ticker this is just that ticker's own normalized path * 100."""
    stacked = np.stack([paths_by_ticker[t] for t in tickers], axis=0)
    return stacked.min(axis=0) * 100.0


def breach_values(worst_of_path, strike_observation):
    """The per-path value strike breach is judged against: the path minimum for
    'american' (breached if it EVER dipped below, checked every day), or just the
    terminal value for 'european' (checked only at maturity)."""
    if strike_observation == "american":
        return worst_of_path.min(axis=1)
    if strike_observation == "european":
        return worst_of_path[:, -1]
    raise ValueError(f"Unknown STRIKE_OBSERVATION: {strike_observation!r}")


def breach_probability_curve(terminal_values, level_grid):
    return (terminal_values[:, None] < level_grid[None, :]).mean(axis=0)


def build_coupon_schedule(tenor_months, frequency):
    step = COUPON_MONTHS_PER_PAYMENT[frequency]
    return list(range(step, tenor_months + 1, step))


def build_autocall_observation_months(tenor_months, delay_months):
    """Monthly observations after the lockout, strictly before maturity -- the final
    month (maturity) is always evaluated via the strike/put payoff instead, not the
    (typically much higher) autocall barrier). delay_months=0 means "no delay": the
    earliest possible observation, month 1, right after the first month elapses.
    delay_months=N skips the first N of those observations on top of that."""
    return list(range(1 + delay_months, tenor_months, 1))


def evaluate_autocall(worst_of_path, strike_pct, autocall_level_pct, observation_days, strike_observation):
    """Walk each simulated basket PATH through the autocall observation dates in
    order. The first date a path's worst-of level closes at/above the autocall
    barrier, that trial is marked called and removed from further consideration --
    NO principal loss is possible on a called path (par is paid, full stop; the put
    is never reached). Any trial never called is evaluated against the strike:
      - "european": only the closing level AT maturity is checked.
      - "american": the ENTIRE path up to maturity is checked -- if the worst-of
        level ever dipped below strike at any point, the put is exercised even if
        it recovered by maturity. Only checked for paths that were never called,
        since a called path's redemption is already settled at par regardless of
        what the underlying does afterward."""
    n_sims, horizon = worst_of_path.shape
    called = np.zeros(n_sims, dtype=bool)
    call_day = np.full(n_sims, -1, dtype=int)
    active = np.ones(n_sims, dtype=bool)
    survival = [(0, 1.0)]

    for day in observation_days:
        newly_called = active & (worst_of_path[:, day] >= autocall_level_pct)
        call_day[newly_called] = day
        called |= newly_called
        active &= ~newly_called
        survival.append((day, float(active.mean())))

    breached = breach_values(worst_of_path, strike_observation) < strike_pct
    principal_loss = active & breached  # put exercised, underlying delivered

    return {"called": called, "call_day": call_day, "matured_uncalled": active,
            "principal_loss": principal_loss, "survival": survival}


def tenor_curve(chain_infos, corr, drift_by_ticker, vol_scale_by_ticker, strike_pct,
                 tickers, tenor_months_list, n_sims, rng, strike_observation):
    rows = []
    for months in tenor_months_list:
        h = months * TRADING_DAYS_PER_MONTH

        e_paths = markov_chain_basket_monte_carlo(chain_infos, corr, tickers, h, n_sims, rng,
                                                    drift_by_ticker, vol_scale_by_ticker, "empirical")
        e_worst = worst_of_index(e_paths, tickers)

        t_paths = markov_chain_basket_monte_carlo(chain_infos, corr, tickers, h, n_sims, rng,
                                                    drift_by_ticker, vol_scale_by_ticker, "t")
        t_worst = worst_of_index(t_paths, tickers)

        rows.append({"tenor_months": months,
                      "empirical_prob": float(np.mean(breach_values(e_worst, strike_observation) < strike_pct)),
                      "t_prob": float(np.mean(breach_values(t_worst, strike_observation) < strike_pct))})
    return pd.DataFrame(rows)


def main():
    tickers = list(dict.fromkeys(TICKERS))  # de-dupe, preserve order
    assert tickers, "TICKERS cannot be empty"
    assert STRIKE_PRICE is None or len(tickers) == 1, "STRIKE_PRICE is single-name only; use STRIKE_PCT for baskets"
    assert 1 <= NOTE_TENOR_MONTHS <= 36, "NOTE_TENOR_MONTHS must be between 1 and 36"
    if AUTOCALL_ENABLED:
        assert 0 <= AUTOCALL_DELAY_MONTHS <= 6, "AUTOCALL_DELAY_MONTHS must be between 0 (no delay) and 6"
        assert AUTOCALL_DELAY_MONTHS < NOTE_TENOR_MONTHS, "AUTOCALL_DELAY_MONTHS must be less than the tenor"
    assert COUPON_FREQUENCY in COUPON_MONTHS_PER_PAYMENT, \
        f"COUPON_FREQUENCY must be one of {list(COUPON_MONTHS_PER_PAYMENT)}"
    assert DRIFT_MODE in ("risk_neutral", "zero", "historical"), \
        f"Unknown DRIFT_MODE: {DRIFT_MODE!r}"
    assert STRIKE_OBSERVATION in ("european", "american"), \
        f"STRIKE_OBSERVATION must be 'european' or 'american', got {STRIKE_OBSERVATION!r}"
    assert VOL_MODE == "empirical", f"This file only supports VOL_MODE == 'empirical', got {VOL_MODE!r}"
    assert len(N_STATES_GRID) > 0 and min(N_STATES_GRID) >= 2, "N_STATES_GRID must contain integers >= 2"

    rng = np.random.default_rng()

    print(f"Underlying(s):     {', '.join(tickers)}" + ("  (worst-of basket)" if len(tickers) > 1 else ""))
    print(f"Tenor:             {NOTE_TENOR_MONTHS} months ({HORIZON_DAYS} trading days)")
    print(f"Strike:            {STRIKE_PCT}% of each name's own spot" if STRIKE_PRICE is None
          else f"Strike:            ${STRIKE_PRICE:,.2f} (single name)")

    obs_months = build_autocall_observation_months(NOTE_TENOR_MONTHS, AUTOCALL_DELAY_MONTHS) if AUTOCALL_ENABLED else []
    obs_days = [m * TRADING_DAYS_PER_MONTH for m in obs_months]

    if AUTOCALL_ENABLED:
        print(f"Structure:         Auto-callable fixed coupon note")
        print(f"Autocall barrier:  {AUTOCALL_BARRIER_PCT}% of each name's own spot")
        if obs_months:
            print(f"Autocall observations: monthly, months {obs_months[0]}-{obs_months[-1]} "
                  f"({len(obs_months)} dates)")
        else:
            print(f"Autocall observations: NONE -- tenor ({NOTE_TENOR_MONTHS}m) is too short for any "
                  f"observation before maturity given AUTOCALL_DELAY_MONTHS={AUTOCALL_DELAY_MONTHS}; "
                  f"this note cannot autocall and behaves like a reverse convertible")
    else:
        print(f"Structure:         Reverse convertible (no early redemption)")
    coupon_months = build_coupon_schedule(NOTE_TENOR_MONTHS, COUPON_FREQUENCY)
    print(f"Coupon schedule:   {COUPON_FREQUENCY}, months {coupon_months} "
          f"(dates only -- amounts need option pricing, not computed here)")
    print(f"Strike observation: {STRIKE_OBSERVATION}"
          + ("  (checked every day -- riskier)" if STRIKE_OBSERVATION == "american"
             else "  (checked only at maturity)"))
    print(f"Drift assumption:  {DRIFT_MODE}")
    print(f"Vol assumption:    {VOL_MODE} (fixed in this file)")
    print()

    # --- per-ticker: fetch, diagnostics, Markov chain state-count selection ---
    T = {}
    for tkr in tickers:
        close = fetch_prices(tkr, PERIOD)
        log_returns = np.log(close / close.shift(1)).dropna()
        log_returns_pct = log_returns * 100

        run_diagnostics(log_returns, tkr)
        k, edges, states = select_markov_chain_model(log_returns_pct, tkr)
        chain_info = describe_markov_chain(k, edges, states, log_returns_pct, tkr)

        spot = float(close.iloc[-1])
        drift_pct, drift_label = resolve_drift(spot)
        vol_scale, vol_label = resolve_vol_scale()

        T[tkr] = dict(close=close, log_returns=log_returns, log_returns_pct=log_returns_pct,
                      chain_info=chain_info, spot=spot, drift_pct=drift_pct, drift_label=drift_label,
                      vol_scale=vol_scale, vol_label=vol_label)

    # --- per-ticker summary: spot, resolved drift/vol vs. fitted historical ---
    print("=" * 70)
    print("Per-ticker summary")
    print("=" * 70)
    for tkr, d in T.items():
        print(f"[{tkr}]  spot=${d['spot']:,.2f}")
        print(f"  Drift used in simulation: {d['drift_label']}")
        print(f"  Vol used in simulation:   {d['vol_label']}")
        print()

    # --- correlation matrix (from state-standardized residuals, aligned on common dates) ---
    print("=" * 70)
    print("STEP 3: Correlation")
    print("=" * 70)
    std_resid = {tkr: T[tkr]["chain_info"]["std_resid"] for tkr in tickers}
    corr, n_common = build_correlation_matrix(std_resid, tickers)
    if len(tickers) > 1:
        print(f"Estimated from {n_common} common trading days of state-standardized residuals:")
        print(pd.DataFrame(corr, index=tickers, columns=tickers).round(2).to_string())
    else:
        print("Single name -- no cross-asset correlation to estimate.")
    print()

    chain_infos = {tkr: T[tkr]["chain_info"] for tkr in tickers}
    drift_by_ticker = {tkr: T[tkr]["drift_pct"] for tkr in tickers}
    vol_scale_by_ticker = {tkr: T[tkr]["vol_scale"] for tkr in tickers}
    fitted_drift_pct = {tkr: float(T[tkr]["log_returns_pct"].mean()) / 100 * TRADING_DAYS_PER_YEAR * 100
                         for tkr in tickers}

    # --- Empirical Bootstrap Monte Carlo (basket-correlated) ---
    print("=" * 70)
    print("STEP 4: Empirical Bootstrap Monte Carlo simulation")
    print("=" * 70)
    e_paths = markov_chain_basket_monte_carlo(chain_infos, corr, tickers, HORIZON_DAYS, N_SIMS, rng,
                                               drift_by_ticker, vol_scale_by_ticker, "empirical")
    e_worst = worst_of_index(e_paths, tickers)
    prob_e = float(np.mean(breach_values(e_worst, STRIKE_OBSERVATION) < STRIKE_PCT))
    for tkr in tickers:
        print(f"  [{tkr}] fitted historical drift: {fitted_drift_pct[tkr]:+.1f}%/yr  "
              f"-> replaced with: {T[tkr]['drift_label']}")
    print(f"P(worst-of < strike at +{HORIZON_DAYS}d), ignoring autocall  [Empirical Bootstrap]:  {prob_e:.1%}")
    print()

    # --- Markov-t Monte Carlo (basket-correlated) ---
    print("=" * 70)
    print("STEP 5: Markov-t Monte Carlo simulation")
    print("=" * 70)
    t_paths = markov_chain_basket_monte_carlo(chain_infos, corr, tickers, HORIZON_DAYS, N_SIMS, rng,
                                               drift_by_ticker, vol_scale_by_ticker, "t")
    t_worst = worst_of_index(t_paths, tickers)
    prob_t = float(np.mean(breach_values(t_worst, STRIKE_OBSERVATION) < STRIKE_PCT))
    print(f"P(worst-of < strike at +{HORIZON_DAYS}d), ignoring autocall  [Markov-t]:  {prob_t:.1%}")
    print()

    # --- autocall / redemption analysis, evaluated on the SAME simulated paths above ---
    print("=" * 70)
    print("STEP 6: Autocall / redemption analysis")
    print("=" * 70)
    e_ac = evaluate_autocall(e_worst, STRIKE_PCT, AUTOCALL_BARRIER_PCT, obs_days, STRIKE_OBSERVATION)
    t_ac = evaluate_autocall(t_worst, STRIKE_PCT, AUTOCALL_BARRIER_PCT, obs_days, STRIKE_OBSERVATION)

    if AUTOCALL_ENABLED:
        for label, ac in (("Empirical Bootstrap", e_ac), ("Markov-t", t_ac)):
            p_called, p_matured, p_loss = ac["called"].mean(), ac["matured_uncalled"].mean(), ac["principal_loss"].mean()
            print(f"[{label}]")
            if ac["called"].any():
                avg_call_month = (ac["call_day"][ac["called"]] / TRADING_DAYS_PER_MONTH).mean()
                print(f"  P(autocalled before maturity):        {p_called:.1%}  (avg. call month: {avg_call_month:.1f})")
            else:
                print(f"  P(autocalled before maturity):        {p_called:.1%}")
            print(f"  P(runs to maturity uncalled):          {p_matured:.1%}")
            print(f"  P(principal loss)  [unconditional]:    {p_loss:.1%}")
            if p_matured > 0:
                print(f"  P(principal loss | reaches maturity):  {p_loss / p_matured:.1%}")
            print()
    else:
        print("Reverse convertible mode (AUTOCALL_ENABLED = False) -- no early redemption; "
              "principal-loss probability equals the terminal breach probability from STEP 4/5.")
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if AUTOCALL_ENABLED:
        print(f"P(principal loss) over the note's life ({NOTE_TENOR_MONTHS}m tenor, "
              f"{AUTOCALL_BARRIER_PCT}% autocall barrier, {AUTOCALL_DELAY_MONTHS}m lockout):")
        print(f"  Empirical Bootstrap (nonparametric within-state):  {e_ac['principal_loss'].mean():.1%}")
        print(f"  Markov-t (parametric within-state):                {t_ac['principal_loss'].mean():.1%}")
        print(f"  (terminal-only breach probability, ignoring autocall, was "
              f"{prob_e:.1%} Empirical / {prob_t:.1%} Markov-t)")
    else:
        print(f"P(worst-of < strike at +{HORIZON_DAYS}d)  [reverse convertible: principal loss probability]")
        print(f"  Empirical Bootstrap (nonparametric within-state):  {prob_e:.1%}")
        print(f"  Markov-t (parametric within-state):                {prob_t:.1%}")
    print()

    # --- tenor curve (sensitivity view, ignores autocall) ---
    print("=" * 70)
    print("STEP 7: Probability vs tenor (terminal breach only, ignoring autocall)")
    print("=" * 70)
    tenor_df = tenor_curve(chain_infos, corr, drift_by_ticker, vol_scale_by_ticker, STRIKE_PCT,
                            tickers, TENOR_GRID_MONTHS, N_SIMS_TENOR, rng, STRIKE_OBSERVATION)
    print(tenor_df.to_string(index=False, formatters={
        "tenor_months": "{:.0f}".format, "empirical_prob": "{:.1%}".format, "t_prob": "{:.1%}".format}))
    print()

    # ------------------------------- CHARTS -------------------------------
    forecast_dates = pd.bdate_range(pd.Timestamp.today().normalize() + pd.Timedelta(days=1), periods=HORIZON_DAYS)

    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Price history, each name rebased so TODAY (spot) = 100 -- NOT the start of the
    # window. STRIKE_PCT/AUTOCALL_BARRIER_PCT are defined as % of each name's current
    # spot, so the strike/barrier lines are only meaningful against that same basis.
    common_start = max(T[tkr]["close"].index.min() for tkr in tickers)
    for i, tkr in enumerate(tickers):
        s = T[tkr]["close"]
        s = s[s.index >= common_start]
        axes[0, 0].plot(s.index, s.values / s.iloc[-1] * 100, color=colors[i % len(colors)], label=tkr)
    axes[0, 0].axhline(STRIKE_PCT, color="red", linestyle="--", label="Strike")
    if AUTOCALL_ENABLED:
        axes[0, 0].axhline(AUTOCALL_BARRIER_PCT, color="green", linestyle="--", label="Autocall barrier")
    axes[0, 0].set_title("Price history (rebased so today = 100)")
    axes[0, 0].set_ylabel("Index level")
    axes[0, 0].legend(fontsize=8)

    # state-conditional annualized volatility over time, per ticker -- stepwise by
    # construction (the state is a discrete label, not a smoothly evolving process)
    for i, tkr in enumerate(tickers):
        vs = T[tkr]["chain_info"]["state_vol_series"]
        axes[0, 1].plot(vs.index, vs.values, color=colors[i % len(colors)], linewidth=0.9, label=tkr)
    axes[0, 1].set_title("State-conditional volatility (annualized %, stepwise)")
    axes[0, 1].set_ylabel("Annualized vol (%)")
    axes[0, 1].legend(fontsize=8)

    # rolling fraction of days in the high-vol state, per ticker
    for i, tkr in enumerate(tickers):
        hv = T[tkr]["chain_info"]["high_vol_rolling"]
        axes[0, 2].plot(hv.index, hv.values, color=colors[i % len(colors)], linewidth=0.9, label=tkr)
    axes[0, 2].axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].set_title("Fraction of trailing 21d in high-vol state")
    axes[0, 2].set_ylabel("Fraction")
    axes[0, 2].legend(fontsize=8)

    # autocall survival curve: fraction of simulated worst-of paths not yet called
    if AUTOCALL_ENABLED and obs_days:
        e_days, e_surv = zip(*e_ac["survival"])
        t_days, t_surv = zip(*t_ac["survival"])
        axes[1, 0].step(e_days, e_surv, where="post", color="tab:blue", label="Empirical Bootstrap")
        axes[1, 0].step(t_days, t_surv, where="post", color="tab:red", label="Markov-t")
        axes[1, 0].axvline(HORIZON_DAYS, color="black", linestyle=":", linewidth=0.8, label="Maturity")
        axes[1, 0].set_ylim(0, 1.05)
        axes[1, 0].set_title("Fraction of simulated paths not yet autocalled")
        axes[1, 0].set_xlabel("Trading day")
        axes[1, 0].set_ylabel("Fraction alive")
        axes[1, 0].legend(fontsize=8)
    else:
        axes[1, 0].axis("off")

    # fan chart: worst-of index percentile bands over the whole horizon
    pct_levels = [5, 25, 50, 75, 95]
    e_pct = np.percentile(e_worst, pct_levels, axis=0)
    t_pct = np.percentile(t_worst, pct_levels, axis=0)
    axes[1, 1].fill_between(forecast_dates, e_pct[0], e_pct[4], color="tab:blue", alpha=0.15)
    axes[1, 1].fill_between(forecast_dates, e_pct[1], e_pct[3], color="tab:blue", alpha=0.3)
    axes[1, 1].plot(forecast_dates, e_pct[2], color="tab:blue", label="Empirical Bootstrap median")
    axes[1, 1].fill_between(forecast_dates, t_pct[0], t_pct[4], color="tab:red", alpha=0.15)
    axes[1, 1].fill_between(forecast_dates, t_pct[1], t_pct[3], color="tab:red", alpha=0.3)
    axes[1, 1].plot(forecast_dates, t_pct[2], color="tab:red", label="Markov-t median")
    axes[1, 1].axhline(STRIKE_PCT, color="black", linestyle="--", label="Strike")
    if AUTOCALL_ENABLED:
        axes[1, 1].axhline(AUTOCALL_BARRIER_PCT, color="green", linestyle="--", label="Autocall barrier")
    axes[1, 1].set_title("Worst-of index fan chart (5/25/50/75/95th pct)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].tick_params(axis="x", rotation=30)

    # CDF / breach-probability curve, worst-of level per STRIKE_OBSERVATION
    level_grid = np.linspace(30, 130, 150)
    e_curve = breach_probability_curve(breach_values(e_worst, STRIKE_OBSERVATION), level_grid)
    t_curve = breach_probability_curve(breach_values(t_worst, STRIKE_OBSERVATION), level_grid)
    axes[1, 2].plot(level_grid, e_curve, color="tab:blue", label="Empirical Bootstrap")
    axes[1, 2].plot(level_grid, t_curve, color="tab:red", label="Markov-t")
    axes[1, 2].axvline(STRIKE_PCT, color="black", linestyle="--")
    axes[1, 2].axhline(prob_e, color="tab:blue", linestyle=":", linewidth=0.8)
    axes[1, 2].axhline(prob_t, color="tab:red", linestyle=":", linewidth=0.8)
    axes[1, 2].set_title(f"P(worst-of < level) at +{HORIZON_DAYS}d ({STRIKE_OBSERVATION}), ignoring autocall")
    axes[1, 2].set_xlabel("Worst-of index level (par = 100)")
    axes[1, 2].set_ylabel("Probability")
    axes[1, 2].legend(fontsize=8)

    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(tenor_df["tenor_months"], tenor_df["empirical_prob"], marker="o", color="tab:blue", label="Empirical Bootstrap")
    ax2.plot(tenor_df["tenor_months"], tenor_df["t_prob"], marker="o", color="tab:red", label="Markov-t")
    ax2.set_title(f"P(worst-of < {STRIKE_PCT}%) vs tenor, ignoring autocall")
    ax2.set_xlabel("Tenor (months)")
    ax2.set_ylabel("Probability")
    ax2.legend()
    fig2.tight_layout()

    # --------------------------- fig3: model diagnostics charts ---------------------------
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))

    # drift comparison: fitted historical vs. resolved/used, per ticker
    x = np.arange(len(tickers))
    used_drift = [drift_by_ticker[tkr] / 100 * TRADING_DAYS_PER_YEAR * 100
                  if drift_by_ticker[tkr] is not None else fitted_drift_pct[tkr] for tkr in tickers]
    axes3[0, 0].bar(x - 0.2, [fitted_drift_pct[tkr] for tkr in tickers], width=0.4,
                     color="tab:gray", label="Fitted (historical)")
    axes3[0, 0].bar(x + 0.2, used_drift, width=0.4, color="tab:blue", label="Used (resolved)")
    axes3[0, 0].axhline(0, color="black", linewidth=0.8)
    axes3[0, 0].set_xticks(x, tickers)
    axes3[0, 0].set_title(f"Drift: fitted vs. used ({DRIFT_MODE})")
    axes3[0, 0].set_ylabel("Annualized drift (%)")
    axes3[0, 0].legend(fontsize=8)

    # vol: chain's own unconditional fitted level (used == fitted always in this file)
    fitted_vol = [chain_unconditional_annualized_vol_pct(
        T[tkr]["chain_info"]["P"], T[tkr]["chain_info"]["means_pct"], T[tkr]["chain_info"]["stds_pct"])
        for tkr in tickers]
    axes3[0, 1].bar(x, fitted_vol, width=0.4, color="tab:gray", label="Fitted (chain, unconditional) = used")
    axes3[0, 1].set_xticks(x, tickers)
    axes3[0, 1].set_title("Volatility: chain's own unconditional fitted level (no rescaling)")
    axes3[0, 1].set_ylabel("Annualized vol (%)")
    axes3[0, 1].legend(fontsize=8)

    # correlation heatmap (baskets only)
    if len(tickers) > 1:
        im = axes3[1, 0].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        axes3[1, 0].set_xticks(range(len(tickers)), tickers, rotation=45, ha="right")
        axes3[1, 0].set_yticks(range(len(tickers)), tickers)
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                axes3[1, 0].text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig3.colorbar(im, ax=axes3[1, 0], fraction=0.046, pad=0.04)
        axes3[1, 0].set_title("Correlation (state-standardized residuals)")
    else:
        axes3[1, 0].axis("off")
        axes3[1, 0].text(0.5, 0.5, "Single name --\nno correlation to show",
                          ha="center", va="center", transform=axes3[1, 0].transAxes, color="gray")

    # path-outcome funnel: autocalled vs. matured-safe vs. matured-loss, both engines
    categories = ["Autocalled", "Matured, no loss", "Matured, loss"]
    e_vals = [e_ac["called"].mean(),
              e_ac["matured_uncalled"].mean() - e_ac["principal_loss"].mean(),
              e_ac["principal_loss"].mean()]
    t_vals = [t_ac["called"].mean(),
              t_ac["matured_uncalled"].mean() - t_ac["principal_loss"].mean(),
              t_ac["principal_loss"].mean()]
    xc = np.arange(len(categories))
    axes3[1, 1].bar(xc - 0.2, e_vals, width=0.4, color="tab:blue", label="Empirical Bootstrap")
    axes3[1, 1].bar(xc + 0.2, t_vals, width=0.4, color="tab:red", label="Markov-t")
    axes3[1, 1].set_xticks(xc, categories)
    axes3[1, 1].set_title("Path outcomes (fraction of simulated paths)")
    axes3[1, 1].set_ylabel("Fraction")
    axes3[1, 1].legend(fontsize=8)

    fig3.tight_layout()

    # --------------------------- fig4: payoff diagram + terminal distribution ---------------------------
    fig4, (ax_payoff, ax_hist) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    level_grid_payoff = np.linspace(0, 130, 300)
    payoff = np.where(level_grid_payoff >= STRIKE_PCT, 100.0, 100.0 * level_grid_payoff / STRIKE_PCT)
    ax_payoff.plot(level_grid_payoff, payoff, color="black", linewidth=1.8)
    ax_payoff.axvline(STRIKE_PCT, color="red", linestyle="--", label="Strike")
    if AUTOCALL_ENABLED:
        ax_payoff.axvline(AUTOCALL_BARRIER_PCT, color="green", linestyle="--",
                           label="Autocall barrier (redeems at par earlier -- not shown here)")
    ax_payoff.set_ylabel("Note value at maturity (par = 100)")
    ax_payoff.set_title("Payoff at maturity if never autocalled, vs. simulated terminal outcomes")
    ax_payoff.legend(fontsize=8)
    if STRIKE_OBSERVATION == "american":
        ax_payoff.text(0.02, 0.05, "Note: STRIKE_OBSERVATION='american' -- some paths shown below the\n"
                                    "kink already lost protection mid-life even if they end up right of it.",
                        transform=ax_payoff.transAxes, fontsize=7, color="gray", va="bottom")

    e_matured_terminal = e_worst[e_ac["matured_uncalled"], -1]
    t_matured_terminal = t_worst[t_ac["matured_uncalled"], -1]
    ax_hist.hist(e_matured_terminal, bins=60, color="tab:blue", alpha=0.5, label="Empirical Bootstrap", density=True)
    ax_hist.hist(t_matured_terminal, bins=60, color="tab:red", alpha=0.5, label="Markov-t", density=True)
    ax_hist.axvline(STRIKE_PCT, color="red", linestyle="--")
    ax_hist.set_xlabel("Worst-of terminal level (par = 100)")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(fontsize=8)
    ax_hist.set_xlim(0, 130)

    fig4.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
