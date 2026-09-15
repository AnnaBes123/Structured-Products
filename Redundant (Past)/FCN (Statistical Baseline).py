###python3 -m venv path/to/venv <- creates environment
###source path/to/venv/bin/activate <- activates environment - need to activate it next time env used
###python3 -m pip install xyz <- downloads library
#These libraries are used to read the knowledge:
# pip install pypdf
# pip install python-docx

# ---------------------------------------------------------------------------
# FCN Probability Editor -- Statistical Baseline
#
# Same model as "FCN (First time).py" (GARCH-t + Markov-Switching, worst-of
# baskets, autocall, American/European strike observation), with the two
# externally-sourced forecast inputs removed: analyst price targets/ratings
# (used there to set the forward drift) and options-implied volatility (used
# there to rescale the forward vol level). This file is purely statistical --
# every number comes from the underlying's own historical price series, no
# outside forecast data. Useful as a comparison baseline against the other
# file if you suspect the analyst/options layer is adding more apparent
# precision than it's actually earning.
#
# A Fixed Coupon Note (FCN) can be decomposed into a zero-coupon bond plus a
# short put option on the underlying(s). If AUTOCALL_ENABLED is off, the note
# is a reverse convertible: it runs to maturity, and the investor takes
# downside risk only if the put is exercised, i.e. the underlying is
# delivered because it closed below the strike at maturity. There is NO
# principal loss on an autocalled note -- the zero-coupon bond leg pays par,
# full stop; the put can only be exercised at maturity, and an autocalled
# note never reaches maturity. If AUTOCALL_ENABLED is on, it's an
# auto-callable fixed coupon note: on monthly observation dates (after an
# initial lockout) the note redeems early at par if the underlying(s) are
# at/above the autocall barrier.
#
# TICKERS can hold more than one symbol, in which case the note is a
# "worst-of" basket: every barrier (autocall and strike) is checked against
# whichever underlying has performed WORST, normalized to its own initial
# level. All of the machinery below -- diagnostics, GARCH, Markov-Switching,
# autocall, tenor curve -- is identical whether TICKERS has 1 or several
# names; internally everything operates on normalized (% of each
# underlying's own spot) paths so a single name is just a basket of one.
#
# The probability of principal loss is estimated from two complementary
# angles, reported side by side, and BOTH the autocall and the maturity
# breach are evaluated on the same simulated Monte Carlo price paths (not a
# separate closed-form approximation):
#
#   A) ARMA-GARCH + Monte Carlo -- captures short-term volatility clustering.
#      Grid-search mean spec x variance family (GARCH, GJR-GARCH/"TARCH",
#      EGARCH, FIGARCH, APARCH, HARCH) x order, all with Student's t
#      innovations, picked by BIC among fits that pass a usability check
#      (rejects fits that report "converged" but are numerically degenerate --
#      e.g. a near-unit-root fit on a short sample that simulates a literally
#      deterministic path regardless of the random draws).
#   B) Markov-Switching + Monte Carlo -- captures structural breaks / regime
#      shifts (calm vs. crisis) that a single GARCH process under-reacts to.
#      2-5 regime switching mean/variance model, picked by BIC among fits
#      that pass a similar usability check (rejects degenerate/near-empty
#      regimes).
#
# For a basket, both engines simulate all underlyings JOINTLY: a correlation
# matrix is estimated from each name's own GARCH standardized residuals
# (aligned on their common trading dates), and every simulated day draws a
# single correlated shock vector via a Gaussian copula, which is then mapped
# to each underlying's own Student's t marginal (its own fitted nu) before
# being fed into its own fitted variance process. This preserves each name's
# individual fat-tailed, vol-clustering behavior while inducing realistic
# co-movement across names -- which is exactly what a worst-of payoff is
# sensitive to (it's the CORRELATION, not just the individual marginals,
# that drives how often the worst performer breaches).
#
# DRIFT_MODE: the fitted GARCH/MS mean is a trailing historical average, and
# compounded over the tenor it can badly bias the estimate -- a stock coming
# off a strong rally shows an artificially LOW breach probability purely
# from extrapolated momentum. The forward simulation's drift is decoupled
# from the fitted historical mean and replaced with one of:
#   "risk_neutral" (recommended here) -- textbook-correct assumption for
#       pricing an embedded option: drift = risk-free rate - dividend yield.
#   "zero"         -- simplest, maximally conservative driftless assumption.
#   "historical"   -- keep the fitted historical mean (NOT recommended for
#       pricing -- reference only, shows what naive trend extrapolation
#       implies).
# (No "analyst_target" option in this file -- see header.)
#
# VOL_MODE is fixed to "garch" in this file -- GARCH's/MS's own fitted
# volatility level is used as-is, with no options-implied rescaling.
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
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

warnings.filterwarnings("ignore")  # arch's / statsmodels' optimizers are chatty about convergence details

# ----------------------------- CONFIG ---------------------------------
TICKERS = ["INTC"]      # one ticker = single name; several = worst-of basket
PERIOD = "5y"                    # how much history to pull: "1y", "2y", "5y", "10y", "max"
N_SIMS = 100_000                 # number of Monte Carlo paths for the headline probability + charts

# Set the strike EITHER as a % of today's spot OR (single-name only) as an
# exact price. Baskets always use STRIKE_PCT, applied to each name's own
# initial level (that's what "worst of, normalized" means).
STRIKE_PCT = 60                  # e.g. 70 means strike = 70% of current spot
STRIKE_PRICE = None              # e.g. 150.00; single-name only, overrides STRIKE_PCT if set

# --- Note structure ---
NOTE_TENOR_MONTHS = 6
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
DRIFT_MODE = "risk_neutral"       # "risk_neutral" | "zero" | "historical"
RISK_FREE_RATE = 0.04             # annualized; used by "risk_neutral"
DIVIDEND_YIELD = 0.0              # annualized; used by "risk_neutral"

# Volatility level is always GARCH's/MS's own fitted level in this file (no
# options-implied rescaling) -- kept as a named constant for parity with the
# other file's function signatures, not a real toggle here.
VOL_MODE = "garch"

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
    DRIFT_MODE == 'historical', signaling 'keep the fitted model's own mean'."""
    if DRIFT_MODE == "risk_neutral":
        drift = (RISK_FREE_RATE - DIVIDEND_YIELD) / TRADING_DAYS_PER_YEAR * 100
        return drift, f"risk-neutral (rf={RISK_FREE_RATE:.1%}, div={DIVIDEND_YIELD:.1%})"

    if DRIFT_MODE == "zero":
        return 0.0, "zero (driftless)"

    if DRIFT_MODE == "historical":
        return None, "historical (fitted model mean -- NOT recommended for pricing)"

    raise ValueError(f"Unknown DRIFT_MODE: {DRIFT_MODE!r}")


def garch_current_annualized_vol_pct(garch_res):
    """GARCH's own most-recent fitted conditional volatility, annualized."""
    return float(garch_res.conditional_volatility.iloc[-1]) / 100 * np.sqrt(TRADING_DAYS_PER_YEAR) * 100


def resolve_vol_scale():
    """Returns (vol_scale, label). Always 1.0 in this file -- no options-implied
    rescaling -- kept as a function for parity with garch_monte_carlo's signature."""
    return 1.0, "garch (model's own fitted volatility)"


def run_diagnostics(log_returns, label):
    """Check whether the series is a reasonable candidate for GARCH modeling."""
    print("=" * 70)
    print(f"STEP 1 [{label}]: Diagnostics")
    print("=" * 70)

    adf_stat, adf_p, *_ = adfuller(log_returns, autolag="AIC")
    stationary = adf_p < 0.05
    print(f"ADF stationarity test:   stat={adf_stat:.3f}  p={adf_p:.4f}  "
          f"-> {'stationary' if stationary else 'NOT stationary'}")

    arch_lm_stat, arch_lm_p, *_ = het_arch(log_returns - log_returns.mean())
    arch_effects = arch_lm_p < 0.05
    print(f"ARCH-LM test (clustering): stat={arch_lm_stat:.3f}  p={arch_lm_p:.4f}  "
          f"-> {'ARCH effects present' if arch_effects else 'no significant ARCH effects'}")

    lb = acorr_ljungbox((log_returns - log_returns.mean()) ** 2, lags=[10], return_df=True)
    lb_p = float(lb["lb_pvalue"].iloc[0])
    lb_clustering = lb_p < 0.05
    print(f"Ljung-Box on squared returns (lag 10): p={lb_p:.4f}  "
          f"-> {'volatility clustering' if lb_clustering else 'no clustering detected'}")

    if not (stationary and (arch_effects or lb_clustering)):
        print("WARNING: diagnostics do not strongly support GARCH modeling for this series. "
              "Proceeding anyway, but treat the results with caution.")
    print()


def garch_fit_is_usable(res, n_check=200, horizon_check=5):
    """Reject fits that LOOK converged (arch's own convergence_flag == 0) but are
    numerically degenerate -- e.g. a near-unit-root fit on a short sample that
    produces a literally deterministic simulated path regardless of the random
    draws (seen in practice: omega/beta standard errors many orders of magnitude
    smaller than the parameters themselves, a classic spurious-optimum signature).
    A cheap small simulation catches this directly rather than guessing at
    indirect proxies."""
    if res.convergence_flag != 0:
        return False
    try:
        fc = res.forecast(horizon=horizon_check, method="simulation", simulations=n_check, reindex=False)
        sim = fc.simulations.values[0]
        return bool(np.all(sim.std(axis=0) > 1e-6))
    except Exception:
        return False


def select_garch_model(log_returns_pct, label):
    """Grid-search mean x variance specs, all with Student's t errors, pick best by BIC
    among fits that pass a usability check (see garch_fit_is_usable)."""
    print("=" * 70)
    print(f"STEP 2 [{label}]: GARCH model selection, Student's t errors")
    print("=" * 70)

    mean_specs = [("Zero", dict(mean="Zero")), ("Constant", dict(mean="Constant")),
                  ("AR(1)", dict(mean="AR", lags=1))]
    vol_specs = [
        ("GARCH(1,1)", dict(vol="GARCH", p=1, o=0, q=1)),
        ("GARCH(1,2)", dict(vol="GARCH", p=1, o=0, q=2)),
        ("GARCH(2,1)", dict(vol="GARCH", p=2, o=0, q=1)),
        ("GJR-GARCH/TARCH(1,1)", dict(vol="GARCH", p=1, o=1, q=1)),
        ("EGARCH(1,1)", dict(vol="EGARCH", p=1, o=1, q=1)),
        ("EGARCH(2,1)", dict(vol="EGARCH", p=2, o=1, q=1)),
        ("FIGARCH(1,1)", dict(vol="FIGARCH", p=1, q=1)),
        ("APARCH(1,1)", dict(vol="APARCH", p=1, o=1, q=1)),
        ("HARCH(1,5,22)", dict(vol="HARCH", p=[1, 5, 22])),
    ]

    results, rejected = [], 0
    for mean_name, mean_kwargs in mean_specs:
        for vol_name, vol_kwargs in vol_specs:
            try:
                am = arch_model(log_returns_pct, dist="studentst", **mean_kwargs, **vol_kwargs)
                res = am.fit(disp="off")
                if not np.isfinite(res.bic):
                    continue
                if not garch_fit_is_usable(res):
                    rejected += 1
                    continue
                results.append({"mean": mean_name, "vol": vol_name, "aic": res.aic,
                                 "bic": res.bic, "result": res})
            except Exception:
                continue

    if rejected:
        print(f"({rejected} candidate(s) fit but rejected as numerically degenerate -- "
              f"e.g. deterministic simulated paths despite reporting convergence)")

    if not results:
        raise RuntimeError(f"No usable ARMA-GARCH model converged for {label}.")

    table = pd.DataFrame([{"mean": r["mean"], "vol": r["vol"], "AIC": r["aic"],
                            "BIC": r["bic"]} for r in results]).sort_values("BIC").reset_index(drop=True)
    print(table.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
    print()

    best = min(results, key=lambda r: r["bic"])
    print(f"Selected: {best['mean']} mean + {best['vol']}, Student's t (lowest BIC = {best['bic']:,.1f})")
    print()
    return best["result"], best["mean"], best["vol"]


def ms_regimes_are_usable(ms_res, k, log_returns_pct, min_occupancy=0.01):
    """Reject fits with a spurious/degenerate regime -- collapsed onto a handful of
    observations (near-zero occupancy) or with near-zero variance -- a known failure
    mode when the regime count is pushed higher than the data can actually support."""
    smoothed = ms_res.smoothed_marginal_probabilities
    for i in range(k):
        w = smoothed.iloc[:, i].values
        if w.mean() < min_occupancy:
            return False
        mean_pct = np.average(log_returns_pct.values, weights=w)
        var_pct = np.average((log_returns_pct.values - mean_pct) ** 2, weights=w)
        if var_pct < 1e-8:
            return False
    return True


def select_ms_model(log_returns_pct, label):
    """Fit 2-5 regime switching mean/variance models, pick the best by BIC among
    fits that pass a usability check (see ms_regimes_are_usable)."""
    print("=" * 70)
    print(f"STEP 3 [{label}]: Markov-Switching model selection")
    print("=" * 70)

    candidates = []
    for k in (2, 3, 4, 5):
        try:
            mod = MarkovRegression(log_returns_pct, k_regimes=k, trend="c", switching_variance=True)
            res = mod.fit()
            if not ms_regimes_are_usable(res, k, log_returns_pct):
                print(f"  k={k} regimes: rejected (degenerate/near-empty regime)")
                continue
            candidates.append((k, res))
        except Exception as exc:
            print(f"  k={k} regimes: fit failed ({exc})")

    if not candidates:
        raise RuntimeError(f"No usable Markov-Switching model converged for {label}.")

    table = pd.DataFrame([{"regimes": k, "AIC": res.aic, "BIC": res.bic}
                           for k, res in candidates]).sort_values("BIC").reset_index(drop=True)
    print(table.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
    print()

    best_k, best_res = min(candidates, key=lambda kr: kr[1].bic)
    print(f"Selected {best_k}-regime model (lowest BIC = {best_res.bic:,.1f})")
    print()
    return best_res, best_k


def describe_ms_model(ms_res, k, log_returns_pct, label):
    """Regime means/vols as smoothed-probability-weighted moments (robust to
    statsmodels' internal parameter naming), plus transition matrix, durations, and
    the dates where the dominant regime switched (structural breaks)."""
    smoothed = ms_res.smoothed_marginal_probabilities
    filtered = ms_res.filtered_marginal_probabilities

    regimes = []
    for i in range(k):
        w = smoothed.iloc[:, i].values
        mean_pct = np.average(log_returns_pct.values, weights=w)
        var_pct = np.average((log_returns_pct.values - mean_pct) ** 2, weights=w)
        regimes.append({"mean_pct": mean_pct, "std_pct": np.sqrt(var_pct)})

    trans = np.asarray(ms_res.regime_transition).mean(axis=2)  # (k, k)
    if np.allclose(trans.sum(axis=0), 1, atol=1e-3):
        pass  # trans[i, j] = P(regime_t = i | regime_{t-1} = j), columns sum to 1
    elif np.allclose(trans.sum(axis=1), 1, atol=1e-3):
        trans = trans.T
    else:
        raise RuntimeError(f"Could not determine Markov transition matrix orientation for {label}.")

    durations = np.asarray(ms_res.expected_durations)
    high_vol_idx = int(np.argmax([r["std_pct"] for r in regimes]))

    print(f"[{label}] {'regime':>7}  {'ann. mean':>10}  {'ann. vol':>9}  {'persistence':>11}  {'exp. dur':>10}")
    for i, r in enumerate(regimes):
        ann_mean = r["mean_pct"] / 100 * TRADING_DAYS_PER_YEAR * 100
        ann_vol = r["std_pct"] / 100 * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        tag = "  (high-vol)" if i == high_vol_idx else ""
        print(f"{'':>9}{i:>7}  {ann_mean:>9.1f}%  {ann_vol:>8.1f}%  {trans[i, i]:>10.1%}  "
              f"{durations[i]:>7.0f}d{tag}")

    dominant = smoothed.to_numpy().argmax(axis=1)
    change_idx = np.where(np.diff(dominant) != 0)[0] + 1
    break_dates = smoothed.index[change_idx]
    print(f"[{label}] Structural breaks (dominant regime switches): {len(break_dates)}")
    print()

    return {
        "regimes": regimes, "transition": trans, "start_probs": filtered.iloc[-1].values,
        "smoothed_high_vol_prob": smoothed.iloc[:, high_vol_idx], "high_vol_idx": high_vol_idx,
    }


def garch_monte_carlo(res, horizon_days, n_sims, drift_pct_per_day, vol_scale=1.0, rng_fn=None):
    """Simulate forward on the fitted model's own time-varying conditional variance.
    Returns the FULL normalized price path per trial (starts at 1.0, no spot
    multiplication -- spot-independent so the same code serves single names and
    baskets alike). drift_pct_per_day overrides the fitted historical mean; pass
    None to keep it (DRIFT_MODE == 'historical'). vol_scale rescales the de-meaned
    shock component -- always 1.0 in this file."""
    forecasts = res.forecast(horizon=horizon_days, method="simulation",
                              simulations=n_sims, rng=rng_fn, reindex=False)
    sim_returns_pct = forecasts.simulations.values[0]  # (n_sims, horizon_days)
    fitted_mean = np.asarray(forecasts.mean.values[0])  # (horizon_days,)
    fitted_annualized_drift_pct = float(fitted_mean.mean()) / 100 * TRADING_DAYS_PER_YEAR * 100
    shocks = (sim_returns_pct - fitted_mean) * vol_scale
    mean_component = drift_pct_per_day if drift_pct_per_day is not None else fitted_mean
    sim_returns_pct = shocks + mean_component
    cum_log_return_path = np.cumsum(sim_returns_pct, axis=1) / 100.0
    path = np.exp(cum_log_return_path)
    return {"path": path, "fitted_annualized_drift_pct": fitted_annualized_drift_pct}


def build_correlation_matrix(std_resid_by_ticker, tickers):
    df = pd.DataFrame(std_resid_by_ticker)[tickers].dropna()
    corr = df.corr().to_numpy()
    corr = corr + np.eye(len(tickers)) * 1e-8  # tiny ridge for numerical stability
    return corr, len(df)


def generate_correlated_std_t_shocks(corr, nus, tickers, n_sims, horizon_days, rng):
    """Gaussian-copula shocks: correlated standard normals -> correlated uniforms ->
    each ticker's own Student's t quantile (its own fitted nu), standardized to unit
    variance. Reduces to plain iid Student's t shocks when there's one ticker."""
    L = np.linalg.cholesky(corr)
    n = len(tickers)
    z = rng.standard_normal((n_sims, horizon_days, n)) @ L.T
    u = np.clip(stats.norm.cdf(z), 1e-10, 1 - 1e-10)
    shocks = {}
    for i, tkr in enumerate(tickers):
        nu = nus[tkr]
        std_dev = np.sqrt(nu / (nu - 2))
        shocks[tkr] = stats.t.ppf(u[:, :, i], nu) / std_dev
    return shocks


def garch_basket_monte_carlo(garch_results, shocks, horizon_days, n_sims, drift_by_ticker, vol_scale_by_ticker):
    """Runs each ticker's OWN fitted GARCH variance recursion (via arch's forecast),
    but fed the SAME cross-sectionally correlated shocks, so co-movement across names
    is preserved while each name's own vol dynamics/fat tails stay intact."""
    paths, fitted = {}, {}
    for tkr, res in garch_results.items():
        s = shocks[tkr]

        def rng_fn(size, _s=s):
            return _s

        out = garch_monte_carlo(res, horizon_days, n_sims, drift_by_ticker[tkr],
                                 vol_scale=vol_scale_by_ticker[tkr], rng_fn=rng_fn)
        paths[tkr] = out["path"]
        fitted[tkr] = out["fitted_annualized_drift_pct"]
    return paths, fitted


def ms_basket_monte_carlo(ms_infos, nus, corr, tickers, horizon_days, n_sims, rng, drift_by_ticker, vol_scale_by_ticker):
    """Each ticker's regime evolves independently via its OWN transition matrix
    (a full joint/shared-regime model is out of scope), but the within-day Student's
    t shocks are cross-sectionally correlated via the same Gaussian copula as the
    GARCH engine, so the worst-of payoff still sees realistic co-movement."""
    n = len(tickers)
    L = np.linalg.cholesky(corr)
    k = {t: len(ms_infos[t]["regimes"]) for t in tickers}
    means = {t: np.array([r["mean_pct"] for r in ms_infos[t]["regimes"]]) for t in tickers}
    stds = {t: np.array([r["std_pct"] for r in ms_infos[t]["regimes"]]) for t in tickers}
    trans = {t: ms_infos[t]["transition"] for t in tickers}
    std_dev = {t: np.sqrt(nus[t] / (nus[t] - 2)) for t in tickers}

    regime = {t: rng.choice(k[t], size=n_sims, p=ms_infos[t]["start_probs"]) for t in tickers}
    cum_log_return = {t: np.zeros(n_sims) for t in tickers}
    path_log_return = {t: np.empty((n_sims, horizon_days)) for t in tickers}

    for day in range(horizon_days):
        z = rng.standard_normal((n_sims, n)) @ L.T
        u = np.clip(stats.norm.cdf(z), 1e-10, 1 - 1e-10)
        for i, t in enumerate(tickers):
            next_regime = np.empty(n_sims, dtype=int)
            for j in range(k[t]):
                mask = regime[t] == j
                if mask.any():
                    next_regime[mask] = rng.choice(k[t], size=mask.sum(), p=trans[t][:, j])
            regime[t] = next_regime
            t_draws = stats.t.ppf(u[:, i], nus[t]) / std_dev[t]
            mean_component = drift_by_ticker[t] if drift_by_ticker[t] is not None else means[t][regime[t]]
            shocks_pct = mean_component + stds[t][regime[t]] * t_draws * vol_scale_by_ticker[t]
            cum_log_return[t] += shocks_pct / 100.0
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


def tenor_curve(garch_results, ms_infos, nus, corr, drift_by_ticker, vol_scale_by_ticker, strike_pct,
                 tickers, tenor_months_list, n_sims, rng, strike_observation):
    rows = []
    for months in tenor_months_list:
        h = months * TRADING_DAYS_PER_MONTH

        g_shocks = generate_correlated_std_t_shocks(corr, nus, tickers, n_sims, h, rng)
        g_paths, _ = garch_basket_monte_carlo(garch_results, g_shocks, h, n_sims, drift_by_ticker, vol_scale_by_ticker)
        g_worst = worst_of_index(g_paths, tickers)

        m_paths = ms_basket_monte_carlo(ms_infos, nus, corr, tickers, h, n_sims, rng, drift_by_ticker, vol_scale_by_ticker)
        m_worst = worst_of_index(m_paths, tickers)

        rows.append({"tenor_months": months,
                      "garch_prob": float(np.mean(breach_values(g_worst, strike_observation) < strike_pct)),
                      "ms_prob": float(np.mean(breach_values(m_worst, strike_observation) < strike_pct))})
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
    assert VOL_MODE == "garch", f"This file only supports VOL_MODE == 'garch', got {VOL_MODE!r}"

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

    # --- per-ticker: fetch, diagnostics, GARCH + MS model selection ---
    T = {}
    for tkr in tickers:
        close = fetch_prices(tkr, PERIOD)
        log_returns = np.log(close / close.shift(1)).dropna()
        log_returns_pct = log_returns * 100

        run_diagnostics(log_returns, tkr)
        garch_res, mean_name, vol_name = select_garch_model(log_returns_pct, tkr)
        nu = float(garch_res.params["nu"])
        ms_res, k = select_ms_model(log_returns_pct, tkr)
        ms_info = describe_ms_model(ms_res, k, log_returns_pct, tkr)

        spot = float(close.iloc[-1])
        drift_pct, drift_label = resolve_drift(spot)
        vol_scale, vol_label = resolve_vol_scale()

        T[tkr] = dict(close=close, log_returns=log_returns, log_returns_pct=log_returns_pct,
                      garch_res=garch_res, mean_name=mean_name, vol_name=vol_name, nu=nu,
                      ms_res=ms_res, k=k, ms_info=ms_info,
                      spot=spot, drift_pct=drift_pct, drift_label=drift_label,
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

    # --- correlation matrix (from GARCH standardized residuals, aligned on common dates) ---
    print("=" * 70)
    print("STEP 4: Correlation")
    print("=" * 70)
    std_resid = {tkr: pd.Series(T[tkr]["garch_res"].std_resid, index=T[tkr]["log_returns_pct"].index)
                 for tkr in tickers}
    corr, n_common = build_correlation_matrix(std_resid, tickers)
    if len(tickers) > 1:
        print(f"Estimated from {n_common} common trading days of GARCH standardized residuals:")
        print(pd.DataFrame(corr, index=tickers, columns=tickers).round(2).to_string())
    else:
        print("Single name -- no cross-asset correlation to estimate.")
    print()

    nus = {tkr: T[tkr]["nu"] for tkr in tickers}
    drift_by_ticker = {tkr: T[tkr]["drift_pct"] for tkr in tickers}
    vol_scale_by_ticker = {tkr: T[tkr]["vol_scale"] for tkr in tickers}

    # --- GARCH Monte Carlo (basket-correlated) ---
    print("=" * 70)
    print("STEP 5: GARCH Monte Carlo simulation")
    print("=" * 70)
    g_shocks = generate_correlated_std_t_shocks(corr, nus, tickers, N_SIMS, HORIZON_DAYS, rng)
    g_paths, g_fitted_drift = garch_basket_monte_carlo(
        {tkr: T[tkr]["garch_res"] for tkr in tickers}, g_shocks, HORIZON_DAYS, N_SIMS,
        drift_by_ticker, vol_scale_by_ticker)
    g_worst = worst_of_index(g_paths, tickers)
    prob_garch = float(np.mean(breach_values(g_worst, STRIKE_OBSERVATION) < STRIKE_PCT))
    for tkr in tickers:
        print(f"  [{tkr}] fitted historical drift: {g_fitted_drift[tkr]:+.1f}%/yr  "
              f"-> replaced with: {T[tkr]['drift_label']}")
    print(f"P(worst-of < strike at +{HORIZON_DAYS}d), ignoring autocall  [GARCH-t]:  {prob_garch:.1%}")
    print()

    # --- Markov-Switching Monte Carlo (basket-correlated) ---
    print("=" * 70)
    print("STEP 6: Markov-Switching Monte Carlo simulation")
    print("=" * 70)
    ms_infos = {tkr: T[tkr]["ms_info"] for tkr in tickers}
    m_paths = ms_basket_monte_carlo(ms_infos, nus, corr, tickers, HORIZON_DAYS, N_SIMS, rng,
                                     drift_by_ticker, vol_scale_by_ticker)
    m_worst = worst_of_index(m_paths, tickers)
    prob_ms = float(np.mean(breach_values(m_worst, STRIKE_OBSERVATION) < STRIKE_PCT))
    print(f"P(worst-of < strike at +{HORIZON_DAYS}d), ignoring autocall  [Markov-Switching]:  {prob_ms:.1%}")
    print()

    # --- autocall / redemption analysis, evaluated on the SAME simulated paths above ---
    print("=" * 70)
    print("STEP 7: Autocall / redemption analysis")
    print("=" * 70)
    garch_ac = evaluate_autocall(g_worst, STRIKE_PCT, AUTOCALL_BARRIER_PCT, obs_days, STRIKE_OBSERVATION)
    ms_ac = evaluate_autocall(m_worst, STRIKE_PCT, AUTOCALL_BARRIER_PCT, obs_days, STRIKE_OBSERVATION)

    if AUTOCALL_ENABLED:
        for label, ac in (("GARCH-t", garch_ac), ("Markov-Switching", ms_ac)):
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
              "principal-loss probability equals the terminal breach probability from STEP 5/6.")
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if AUTOCALL_ENABLED:
        print(f"P(principal loss) over the note's life ({NOTE_TENOR_MONTHS}m tenor, "
              f"{AUTOCALL_BARRIER_PCT}% autocall barrier, {AUTOCALL_DELAY_MONTHS}m lockout):")
        print(f"  GARCH-t (volatility clustering):     {garch_ac['principal_loss'].mean():.1%}")
        print(f"  Markov-Switching (regime shifts):    {ms_ac['principal_loss'].mean():.1%}")
        print(f"  (terminal-only breach probability, ignoring autocall, was "
              f"{prob_garch:.1%} GARCH-t / {prob_ms:.1%} MS)")
    else:
        print(f"P(worst-of < strike at +{HORIZON_DAYS}d)  [reverse convertible: principal loss probability]")
        print(f"  GARCH-t (volatility clustering):     {prob_garch:.1%}")
        print(f"  Markov-Switching (regime shifts):    {prob_ms:.1%}")
    print()

    # --- tenor curve (sensitivity view, ignores autocall) ---
    print("=" * 70)
    print("STEP 8: Probability vs tenor (terminal breach only, ignoring autocall)")
    print("=" * 70)
    tenor_df = tenor_curve(
        {tkr: T[tkr]["garch_res"] for tkr in tickers}, ms_infos, nus, corr, drift_by_ticker, vol_scale_by_ticker,
        STRIKE_PCT, tickers, TENOR_GRID_MONTHS, N_SIMS_TENOR, rng, STRIKE_OBSERVATION)
    print(tenor_df.to_string(index=False, formatters={
        "tenor_months": "{:.0f}".format, "garch_prob": "{:.1%}".format, "ms_prob": "{:.1%}".format}))
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

    # GARCH conditional volatility over time, per ticker
    for i, tkr in enumerate(tickers):
        d = T[tkr]
        ann_vol = d["garch_res"].conditional_volatility / 100 * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        axes[0, 1].plot(d["log_returns"].index, ann_vol, color=colors[i % len(colors)], label=tkr)
    axes[0, 1].set_title("GARCH conditional volatility (annualized %)")
    axes[0, 1].set_ylabel("Annualized vol (%)")
    axes[0, 1].legend(fontsize=8)

    # regime probability timeline, per ticker
    for i, tkr in enumerate(tickers):
        hv = T[tkr]["ms_info"]["smoothed_high_vol_prob"]
        axes[0, 2].plot(hv.index, hv.values, color=colors[i % len(colors)], linewidth=0.9, label=tkr)
    axes[0, 2].axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].set_title("P(high-vol regime) over time")
    axes[0, 2].set_ylabel("Probability")
    axes[0, 2].legend(fontsize=8)

    # autocall survival curve: fraction of simulated worst-of paths not yet called
    if AUTOCALL_ENABLED and obs_days:
        g_days, g_surv = zip(*garch_ac["survival"])
        m_days, m_surv = zip(*ms_ac["survival"])
        axes[1, 0].step(g_days, g_surv, where="post", color="tab:blue", label="GARCH-t")
        axes[1, 0].step(m_days, m_surv, where="post", color="tab:red", label="Markov-Switching")
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
    g_pct = np.percentile(g_worst, pct_levels, axis=0)
    m_pct = np.percentile(m_worst, pct_levels, axis=0)
    axes[1, 1].fill_between(forecast_dates, g_pct[0], g_pct[4], color="tab:blue", alpha=0.15)
    axes[1, 1].fill_between(forecast_dates, g_pct[1], g_pct[3], color="tab:blue", alpha=0.3)
    axes[1, 1].plot(forecast_dates, g_pct[2], color="tab:blue", label="GARCH-t median")
    axes[1, 1].fill_between(forecast_dates, m_pct[0], m_pct[4], color="tab:red", alpha=0.15)
    axes[1, 1].fill_between(forecast_dates, m_pct[1], m_pct[3], color="tab:red", alpha=0.3)
    axes[1, 1].plot(forecast_dates, m_pct[2], color="tab:red", label="Markov-Switching median")
    axes[1, 1].axhline(STRIKE_PCT, color="black", linestyle="--", label="Strike")
    if AUTOCALL_ENABLED:
        axes[1, 1].axhline(AUTOCALL_BARRIER_PCT, color="green", linestyle="--", label="Autocall barrier")
    axes[1, 1].set_title("Worst-of index fan chart (5/25/50/75/95th pct)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].tick_params(axis="x", rotation=30)

    # CDF / breach-probability curve, worst-of level per STRIKE_OBSERVATION
    level_grid = np.linspace(30, 130, 150)
    g_curve = breach_probability_curve(breach_values(g_worst, STRIKE_OBSERVATION), level_grid)
    m_curve = breach_probability_curve(breach_values(m_worst, STRIKE_OBSERVATION), level_grid)
    axes[1, 2].plot(level_grid, g_curve, color="tab:blue", label="GARCH-t")
    axes[1, 2].plot(level_grid, m_curve, color="tab:red", label="Markov-Switching")
    axes[1, 2].axvline(STRIKE_PCT, color="black", linestyle="--")
    axes[1, 2].axhline(prob_garch, color="tab:blue", linestyle=":", linewidth=0.8)
    axes[1, 2].axhline(prob_ms, color="tab:red", linestyle=":", linewidth=0.8)
    axes[1, 2].set_title(f"P(worst-of < level) at +{HORIZON_DAYS}d ({STRIKE_OBSERVATION}), ignoring autocall")
    axes[1, 2].set_xlabel("Worst-of index level (par = 100)")
    axes[1, 2].set_ylabel("Probability")
    axes[1, 2].legend(fontsize=8)

    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(tenor_df["tenor_months"], tenor_df["garch_prob"], marker="o", color="tab:blue", label="GARCH-t")
    ax2.plot(tenor_df["tenor_months"], tenor_df["ms_prob"], marker="o", color="tab:red", label="Markov-Switching")
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
                  if drift_by_ticker[tkr] is not None else g_fitted_drift[tkr] for tkr in tickers]
    axes3[0, 0].bar(x - 0.2, [g_fitted_drift[tkr] for tkr in tickers], width=0.4,
                     color="tab:gray", label="Fitted (historical)")
    axes3[0, 0].bar(x + 0.2, used_drift, width=0.4, color="tab:blue", label="Used (resolved)")
    axes3[0, 0].axhline(0, color="black", linewidth=0.8)
    axes3[0, 0].set_xticks(x, tickers)
    axes3[0, 0].set_title(f"Drift: fitted vs. used ({DRIFT_MODE})")
    axes3[0, 0].set_ylabel("Annualized drift (%)")
    axes3[0, 0].legend(fontsize=8)

    # vol: GARCH's own fitted level (used == fitted always in this file, no rescaling)
    fitted_vol = [garch_current_annualized_vol_pct(T[tkr]["garch_res"]) for tkr in tickers]
    axes3[0, 1].bar(x, fitted_vol, width=0.4, color="tab:gray", label="Fitted (GARCH) = used")
    axes3[0, 1].set_xticks(x, tickers)
    axes3[0, 1].set_title("Volatility: GARCH's own fitted level (no rescaling in this file)")
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
        axes3[1, 0].set_title("Correlation (GARCH standardized residuals)")
    else:
        axes3[1, 0].axis("off")
        axes3[1, 0].text(0.5, 0.5, "Single name --\nno correlation to show",
                          ha="center", va="center", transform=axes3[1, 0].transAxes, color="gray")

    # path-outcome funnel: autocalled vs. matured-safe vs. matured-loss, both engines
    categories = ["Autocalled", "Matured, no loss", "Matured, loss"]
    garch_vals = [garch_ac["called"].mean(),
                  garch_ac["matured_uncalled"].mean() - garch_ac["principal_loss"].mean(),
                  garch_ac["principal_loss"].mean()]
    ms_vals = [ms_ac["called"].mean(),
               ms_ac["matured_uncalled"].mean() - ms_ac["principal_loss"].mean(),
               ms_ac["principal_loss"].mean()]
    xc = np.arange(len(categories))
    axes3[1, 1].bar(xc - 0.2, garch_vals, width=0.4, color="tab:blue", label="GARCH-t")
    axes3[1, 1].bar(xc + 0.2, ms_vals, width=0.4, color="tab:red", label="Markov-Switching")
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

    g_matured_terminal = g_worst[garch_ac["matured_uncalled"], -1]
    m_matured_terminal = m_worst[ms_ac["matured_uncalled"], -1]
    ax_hist.hist(g_matured_terminal, bins=60, color="tab:blue", alpha=0.5, label="GARCH-t", density=True)
    ax_hist.hist(m_matured_terminal, bins=60, color="tab:red", alpha=0.5, label="Markov-Switching", density=True)
    ax_hist.axvline(STRIKE_PCT, color="red", linestyle="--")
    ax_hist.set_xlabel("Worst-of terminal level (par = 100)")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(fontsize=8)
    ax_hist.set_xlim(0, 130)

    fig4.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
