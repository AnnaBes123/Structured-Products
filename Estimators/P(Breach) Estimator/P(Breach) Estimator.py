import gc
import glob
import os
import resource
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t as t_dist

from arch import arch_model
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

plt.rcParams["font.family"] = "Arial"
warnings.filterwarnings("ignore", category=UserWarning)

# Fail with a catchable MemoryError instead of a silent, no-diagnostic
# SIGKILL if something runs away - repeated model fitting/simulation in the
# loops below has caused real, hard-to-diagnose memory growth in testing.
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = 4 * 1024 ** 3  # 4 GB
    resource.setrlimit(resource.RLIMIT_AS, (cap, hard if hard != resource.RLIM_INFINITY else cap))
except Exception:
    pass  # not available on all platforms - best effort only

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_BASENAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_PNG = os.path.join(SCRIPT_DIR, SCRIPT_BASENAME + ".png")

# Real-world (physical-measure) probability estimator, not a pricer - see
# README.md for the full rationale, especially why this deliberately isn't
# risk-neutral.

# --- Product terms (same STRIKE/ENTRY_DATE/TENOR convention as the plain
# Fixed Coupon Note.py, one level up in Product MtM/Yield/) ---
#
# PRODUCT_TYPE selects what "breach" means:
#   - "FCN" / "RC":  European, terminal-only condition (same math either way -
#     the plain Fixed Coupon Note and Reverse Convertible in this repo are both
#     a worst-of put struck at STRIKE and only ever looked at, at maturity - no
#     continuous barrier). "P(breach)" = P(worst-of < STRIKE at maturity).
#     BARRIER is ignored.
#   - "BRC": Barrier Reverse Convertible - a genuine down-and-in put, same
#     structure as ../../Product MtM/Yield/Barrier Reverse Convertible/. The
#     put only activates if BARRIER is EVER touched (any date from ENTRY_DATE
#     to maturity, continuously monitored daily); "P(breach)" =
#     P(barrier ever touched AND worst-of < STRIKE at maturity) - a genuinely
#     path-dependent, joint condition, not just a terminal one. Requires
#     0 < BARRIER < STRIKE.
#   - "GENERIC": no product semantics assumed - reports two independent stats
#     off the one STRIKE level, exactly as this file always has: P(breach AT
#     MATURITY) and P(EVER touches STRIKE) along the path. Useful for asking
#     "how likely is this move, really?" without picking a specific structure.
#     BARRIER is ignored.
PRODUCT_TYPE = "GENERIC"

STRIKE = 0.5                  # breach level, as a fraction of each name's own entry level -
                               # the put strike (FCN/RC/BRC) or just "the level" (GENERIC)
BARRIER = None                 # BRC only: down-and-in knock-in level, as a fraction of each
                               # name's own entry level. Must be < STRIKE. Ignored otherwise.
ENTRY_DATE = "2026-09-17"
TENOR = 1

if PRODUCT_TYPE not in ("FCN", "RC", "BRC", "GENERIC"):
    raise ValueError(f"PRODUCT_TYPE must be one of 'FCN', 'RC', 'BRC', 'GENERIC' - got {PRODUCT_TYPE!r}")
if PRODUCT_TYPE == "BRC" and not (BARRIER is not None and 0 < BARRIER < STRIKE):
    raise ValueError(f"PRODUCT_TYPE='BRC' requires 0 < BARRIER < STRIKE (got BARRIER={BARRIER!r}, STRIKE={STRIKE!r})")

PRODUCT_LABELS = {"FCN": "Fixed Coupon Note", "RC": "Reverse Convertible",
                   "BRC": "Barrier Reverse Convertible", "GENERIC": "General Probability Estimator"}

# 1 ticker = single-asset mode. 2+ = basket (worst-of) mode, same convention
# as Multi-RC / Multi-FCN. Try TICKERS = ["AAPL", "JPM", "XOM"].
TICKERS = ["CL=F", "BZ=F"]

N_SIMULATIONS = 10000            # Monte Carlo paths per rolling origin date - a "decent" number
                                 # gives ~1.4% standard error on a 50% probability; see
                                 # MAX_SAFE_SIMULATIONS just below for why this is capped rather
                                 # than left uncapped.
RNG_SEED = 42

# In basket mode, a fresh (n_sims x horizon_days x n_tickers) shock array is
# allocated at EVERY rolling/validation date, so N_SIMULATIONS multiplies
# memory and runtime directly across hundreds of dates - a value that's fine
# single-asset can be unworkable for a basket. Every simulation call below
# clamps to this ceiling regardless of what N_SIMULATIONS above is set to,
# so a large value degrades to "as many sims as is safe" instead of hanging
# or getting killed.
MAX_SAFE_SIMULATIONS = 20000


def _safe_n_sims(requested, basket_mode=False):
    cap = MAX_SAFE_SIMULATIONS // 4 if basket_mode else MAX_SAFE_SIMULATIONS
    if requested > cap:
        print(f"    [n_sims clamped from {requested:,} to {cap:,} - see MAX_SAFE_SIMULATIONS]")
        return cap
    return requested

TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365.25


def _cleanup_stale_validation_pngs():
    """Deletes every 'Validation (<ticker>).png' from a PREVIOUS run before
    this run writes fresh ones - this script is routinely re-run against
    different ad hoc TICKERS (it's a general P(breach) exploration tool, not
    tied to one fixed underlying), and per-ticker filenames would otherwise
    accumulate a stale chart for every ticker ever tried instead of just the
    current TICKERS."""
    pattern = os.path.join(SCRIPT_DIR, f"{SCRIPT_BASENAME} - Validation (*).png")
    for path in glob.glob(pattern):
        os.remove(path)


def _calendar_to_trading_days(calendar_days):
    """Both the rolling and walk-forward simulations advance one FITTED
    trading-day return per simulated step, so a horizon expressed in
    calendar days (as maturity/entry/target dates naturally are) would
    simulate ~365/252 too many daily shocks - this converts to the
    equivalent trading-day count instead."""
    return max(1, round(calendar_days * TRADING_DAYS_PER_YEAR / CALENDAR_DAYS_PER_YEAR))


# Markov-switching (statsmodels MarkovRegression) is off by default: its MLE
# fit computes a numerical Hessian for standard errors, and in testing this
# was consistently where multi-gigabyte memory growth started (every
# GARCH-family-only run has been fast and stable; every run that also fit
# Markov-switching candidates was not). Set True to re-enable once/if that's
# tracked down - the rest of the pipeline (model grid, basket correlation,
# rolling MC, walk-forward validation) works the same either way, just
# choosing among GARCH-family specs only when this is off.
INCLUDE_MARKOV_SWITCHING = False

# Walk-forward validation / "self-learning" - see README for the full design.
FULL_HISTORY_YEARS = 15         # total real history fetched, ending at ENTRY_DATE
VALIDATION_YEARS = 3            # most recent slice of that held out as a test period - kept
                                 # deliberately modest (a "decent," not maximal, validation run):
                                 # each refit/checkpoint fits ~18 GARCH-family models from scratch
                                 # (3 mean specs x 6 vol specs; +2 more if
                                 # INCLUDE_MARKOV_SWITCHING), so the total cost scales directly with
                                 # VALIDATION_YEARS / EVAL_FREQUENCY_DAYS times the size of that grid
REFIT_FREQUENCY_DAYS = 252      # model fully re-estimated this often (~1y)
EVAL_FREQUENCY_DAYS = 42        # prediction checkpoint spacing (~2mo)


def fetch_daily_closes(ticker, start, end, fred_series=None):
    series = None

    try:
        import yfinance as yf
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if data is not None and not data.empty:
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            series = close.dropna()
    except Exception as exc:
        print(f"  yfinance failed for {ticker} ({exc})" + (" ; trying FRED fallback..." if fred_series else ""))

    if (series is None or series.empty) and fred_series:
        try:
            import pandas_datareader.data as web
            data = web.DataReader(fred_series, "fred", start, end)
            series = data[fred_series].dropna()
        except Exception as exc:
            raise RuntimeError(f"Could not fetch {ticker} data from either yfinance or FRED: {exc}")

    if series is None or series.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    return series


def fetch_underlying_name(ticker):
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker


def fetch_multi_asset_path(tickers, start, end):
    fred = "SP500" if tickers == ["^GSPC"] else None
    if len(tickers) == 1:
        series = {tickers[0]: fetch_daily_closes(tickers[0], start, end, fred_series=fred)}
    else:
        series = {t: fetch_daily_closes(t, start, end) for t in tickers}
    df = pd.concat(series, axis=1)
    return df.dropna(how="any")


def log_returns_pct(price_df):
    """Log returns in percent - arch's numerics work better at O(1) than O(0.01)."""
    return (np.log(price_df / price_df.shift(1)).dropna()) * 100.0


# Model selection: grid of ARMA-mean x GARCH-family-variance candidates plus
# Markov-switching, picked by BIC (AIC also reported - see README for the
# AIC/BIC/SBIC and GJR-vs-TGARCH and VAR/VARMA notes).

GARCH_MEAN_SPECS = [
    ("Zero", dict(mean="Zero")),
    ("Constant", dict(mean="Constant")),
    ("AR(1)", dict(mean="AR", lags=1)),
]

GARCH_VOL_SPECS = [
    ("ARCH(1)", dict(vol="ARCH", p=1)),
    ("GARCH(1,1)", dict(vol="GARCH", p=1, o=0, q=1)),
    ("GJR-GARCH(1,1,1)", dict(vol="GARCH", p=1, o=1, q=1)),
    ("EGARCH(1,1,1)", dict(vol="EGARCH", p=1, o=1, q=1)),
    ("APARCH(1,1,1)", dict(vol="APARCH", p=1, o=1, q=1)),
    ("FIGARCH(1,1)", dict(vol="FIGARCH", p=1, q=1)),
]

# Specs whose fitted parameterization the hand-rolled BASKET one-step
# recursion (_garch_one_step_variance) can't represent, so they're only
# offered in single-asset mode: EGARCH's recursion is in log-variance space;
# APARCH's is in |eps|^delta space with an estimated power delta (not just
# eps^2); FIGARCH's is a truncated ARCH(inf) expansion parameterized by
# (phi, d, beta), not (omega, alpha, beta) at all. ARCH(1) needs no entry
# here - it's just GARCH(1,1) with beta forced to 0, which the basket
# recursion already handles (see _garch_one_step_variance's use of .get()).
BASKET_INCOMPATIBLE_VOL_SPECS = {"EGARCH(1,1,1)", "APARCH(1,1,1)", "FIGARCH(1,1)"}

MARKOV_K_REGIMES = [2, 3]


def fit_garch_family_candidates(returns_pct, fit_end, basket_mode=False):
    """Fits every mean x vol combo above (drops BASKET_INCOMPATIBLE_VOL_SPECS
    when basket_mode=True - see its docstring for why each is excluded).
    Estimation uses only data through fit_end - no look-ahead."""
    vol_specs = [s for s in GARCH_VOL_SPECS if not (basket_mode and s[0] in BASKET_INCOMPATIBLE_VOL_SPECS)]
    candidates = []
    for mean_name, mean_kwargs in GARCH_MEAN_SPECS:
        for vol_name, vol_kwargs in vol_specs:
            label = f"{mean_name} + {vol_name} (t)"
            try:
                am = arch_model(returns_pct, dist="t", **mean_kwargs, **vol_kwargs)
                res = am.fit(last_obs=fit_end, disp="off", show_warning=False)
                candidates.append({
                    "label": label, "kind": "garch", "result": am, "fitted": res,
                    "aic": res.aic, "bic": res.bic, "loglik": res.loglikelihood,
                    "nparams": len(res.params),
                    "spec_kwargs": {**mean_kwargs, **vol_kwargs},  # needed to rebuild a
                    # truncated-data model at forecast time - see simulate_garch_forward
                })
            except Exception as exc:
                print(f"    [skip] {label}: failed to fit ({exc})")
    return candidates


def fit_markov_candidates(returns_pct, fit_end):
    """Gaussian, iid-within-regime Markov-switching (switching mean AND
    variance), k=2 and k=3 regimes. See the README for why AR-switching
    and basket-joint regime models were left out of scope."""
    fit_data = returns_pct.loc[:fit_end]
    candidates = []
    for k in MARKOV_K_REGIMES:
        label = f"Markov-Switching ({k} regimes, Gaussian)"
        try:
            mod = MarkovRegression(fit_data, k_regimes=k, trend="c", switching_variance=True)
            res = mod.fit()
            candidates.append({
                "label": label, "kind": "markov", "result": mod, "fitted": res,
                "aic": res.aic, "bic": res.bic, "loglik": res.llf,
                "nparams": len(res.params), "k_regimes": k,
            })
        except Exception as exc:
            print(f"    [skip] {label}: failed to fit ({exc})")
    return candidates


def select_best_model(candidates, underlying_label, verbose=True):
    table = pd.DataFrame([
        {"Model": c["label"], "Params": c["nparams"], "LogLik": c["loglik"], "AIC": c["aic"], "BIC": c["bic"]}
        for c in candidates
    ]).sort_values("BIC").reset_index(drop=True)

    best_bic_label = table.iloc[0]["Model"]
    best_aic_label = table.sort_values("AIC").iloc[0]["Model"]

    if verbose:
        print(f"\nModel comparison for {underlying_label} (sorted by BIC - lower is better):")
        print(table.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
        if best_aic_label != best_bic_label:
            print(f"  Note: AIC would have picked a different model ('{best_aic_label}') than BIC "
                  f"('{best_bic_label}') - BIC's heavier complexity penalty is used as the tiebreaker "
                  f"(see the module docstring for why).")
        else:
            print(f"  AIC and BIC agree: '{best_bic_label}'")

    best = next(c for c in candidates if c["label"] == best_bic_label)
    return best, table


# ---------------------------------------------------------------------------
# SINGLE-ASSET forward simulation
# ---------------------------------------------------------------------------

def simulate_garch_forward(best, returns_pct, origin_date, horizon_days, n_sims=N_SIMULATIONS, seed=RNG_SEED):
    """Native arch-package simulation from a rolling origin, using the
    ALREADY-FITTED parameters (no re-estimation).

    Rebuilds the arch_model on data TRUNCATED to origin_date and calls
    .fix(params) (fixes the already-fitted parameters, no re-estimation)
    rather than calling .forecast(start=origin_date) on the full-length
    model. That distinction matters a lot: arch's forecast(start=X)
    computes simulations for EVERY origin from X through the end of the
    model's own data, not just one - for an early checkpoint with years of
    data still ahead of it, that's thousands of unused origins computed
    and discarded (confirmed as the actual cause of multi-GB memory use
    in testing). Truncating first makes origin_date the LAST point in the
    data, so there is exactly one origin to compute."""
    truncated = returns_pct.loc[:origin_date]
    am = arch_model(truncated, dist="t", **best["spec_kwargs"])
    # Seed the model's OWN fitted Student's-t distribution rather than passing
    # rng= to .forecast() below: arch's rng= replaces the innovation generator
    # entirely with whatever callable it's given, bypassing the fitted
    # distribution's own simulate() (which draws standard_t scaled by the
    # fitted degrees-of-freedom) in favor of that callable's raw output - a
    # standard_normal callable there would silently discard the fitted
    # Student's-t shape and simulate Gaussian innovations instead.
    am.distribution._generator = np.random.default_rng(seed)
    fixed = am.fix(best["fitted"].params)
    fc = fixed.forecast(horizon=horizon_days, method="simulation",
                         simulations=n_sims, reindex=False)
    sim_returns_pct = fc.simulations.values[0]  # shape (n_sims, horizon_days) - the one origin
    cum_log_return = sim_returns_pct.sum(axis=1) / 100.0
    terminal_relative = np.exp(cum_log_return)
    # A cheap, honestly-labeled "ever touched" proxy: the running MINIMUM of
    # the cumulative sum (not of the accumulated minimum of increments,
    # which would be wrong) - computed properly via cumsum then cummin:
    cum_path_pct = np.cumsum(sim_returns_pct, axis=1)
    path_min_relative = np.exp(np.minimum.accumulate(cum_path_pct, axis=1).min(axis=1) / 100.0)
    return terminal_relative, path_min_relative


def simulate_markov_forward(best, returns_pct, origin_date, horizon_days, n_sims=N_SIMULATIONS, seed=RNG_SEED):
    """Hand-rolled forward sim for Markov-switching (statsmodels has no
    built-in simulate()). Regime distribution at origin_date comes from
    re-filtering (not re-estimating) the fixed fitted params through real
    data up to that date - verified to reproduce the original fit's own
    filtered probabilities on the overlapping period."""
    k = best["k_regimes"]
    params = best["fitted"].params

    extended = returns_pct.loc[:origin_date]
    mod_ext = MarkovRegression(extended, k_regimes=k, trend="c", switching_variance=True)
    filtered = mod_ext.filter(params)
    regime_probs_today = filtered.filtered_marginal_probabilities.iloc[-1].values

    T = filtered.regime_transition[:, :, 0] if filtered.regime_transition.shape[2] > 1 else \
        best["fitted"].regime_transition[:, :, 0]
    # T[i, j] = P(next regime = i | current regime = j) - column-stochastic

    const = np.array([params[f"const[{i}]"] for i in range(k)])
    sigma = np.array([np.sqrt(params[f"sigma2[{i}]"]) for i in range(k)])

    rng = np.random.default_rng(seed)
    regime = rng.choice(k, size=n_sims, p=regime_probs_today)

    cum_return_pct = np.zeros(n_sims)
    running_min_pct = np.zeros(n_sims)
    for _ in range(horizon_days):
        u = rng.random(n_sims)
        cum_trans = np.cumsum(T[:, regime], axis=0)  # shape (k, n_sims)
        regime = (u[None, :] > cum_trans).sum(axis=0)
        regime = np.clip(regime, 0, k - 1)

        day_return = const[regime] + sigma[regime] * rng.standard_normal(n_sims)
        cum_return_pct += day_return
        running_min_pct = np.minimum(running_min_pct, cum_return_pct)

    terminal_relative = np.exp(cum_return_pct / 100.0)
    path_min_relative = np.exp(running_min_pct / 100.0)
    return terminal_relative, path_min_relative


# Basket mode: hand-rolled CCC-GARCH-style joint simulator. Each name keeps
# its own best-fit model; a real correlation matrix of standardized shocks
# ties them together (see README - arch's own simulate() can't take
# externally-correlated shocks, hence the hand-rolled one-step recursion
# below, restricted to GARCH/GJR since EGARCH's log-variance recursion
# isn't worth the added implementation risk here).

def _garch_one_step_variance(sigma2_prev, eps_prev, params, has_gamma):
    omega = params["omega"]
    alpha = params["alpha[1]"]
    beta = params.get("beta[1]", 0.0)  # ARCH(1) has no q lag, hence no beta[1] -
    # falls back to 0 rather than KeyError, degrading correctly to the ARCH(1) recursion
    var = omega + alpha * eps_prev ** 2 + beta * sigma2_prev
    if has_gamma:
        gamma = params["gamma[1]"]
        var = var + gamma * (eps_prev ** 2) * (eps_prev < 0)
    return np.maximum(var, 1e-12)


def simulate_basket_garch_forward(best, returns_pct, origin_date, horizon_days, corr_row_shock, n_sims):
    """One-step GARCH/GJR recursion driven by a supplied CORRELATED shock
    stream (already Cholesky-correlated across the basket).

    Truncates to origin_date and re-fixes (not re-estimates) the already-
    selected parameters on that truncated data, same reasoning as
    simulate_garch_forward: arch's last_obs-restricted fit leaves the
    conditional variance/residual NaN exactly AT the last_obs boundary,
    which is every rolling origin_date used here - reading state straight
    off best["fitted"] silently produced NaN paths (and hence a spurious
    0% breach probability) at every origin."""
    params = best["fitted"].params
    has_gamma = "gamma[1]" in params.index
    mean_kind = best["label"].split(" + ")[0]

    truncated = returns_pct.loc[:origin_date]
    am = arch_model(truncated, dist="t", **best["spec_kwargs"])
    fixed = am.fix(params)
    sigma2 = float(fixed.conditional_volatility.iloc[-1] ** 2)
    eps = float(fixed.resid.iloc[-1])

    if mean_kind == "AR(1)":
        mu_const, phi1 = params.get("Const", 0.0), params.get("None[1]", params.get("y[1]", 0.0))
        last_level = float(truncated.iloc[-1])
    else:
        mu_const = params.get("mu", params.get("Const", 0.0))
        phi1 = 0.0
        last_level = 0.0

    cum_return_pct = np.zeros(n_sims)
    running_min_pct = np.zeros(n_sims)
    sigma2_path = np.full(n_sims, sigma2)
    eps_path = np.full(n_sims, eps)
    level_path = np.full(n_sims, last_level)

    for t in range(horizon_days):
        sigma2_path = _garch_one_step_variance(sigma2_path, eps_path, params, has_gamma)
        mu_t = mu_const + phi1 * level_path if mean_kind == "AR(1)" else mu_const
        shock = corr_row_shock[:, t]
        day_return = mu_t + np.sqrt(sigma2_path) * shock
        eps_path = day_return - mu_t
        level_path = day_return

        cum_return_pct += day_return
        running_min_pct = np.minimum(running_min_pct, cum_return_pct)

    terminal_relative = np.exp(cum_return_pct / 100.0)
    path_min_relative = np.exp(running_min_pct / 100.0)
    return terminal_relative, path_min_relative


def _build_basket_shocks(best_models, tickers, L, n_sims, horizon_days, seed):
    """Correlated shocks for the basket loop, one column per ticker.

    Draws correlated standard-normal shocks (via the Cholesky factor L of the
    real shock correlation matrix), then - for every GARCH-family name, which
    was fit with dist="t" - remaps that name's own column through a Gaussian
    copula onto its OWN fitted Student's-t marginal (df=nu, rescaled to unit
    variance, matching arch's own convention). This keeps the correlation
    structure while restoring the fat-tailed marginal the model was actually
    fitted with, instead of leaving every name Gaussian. Markov-switching
    names are left as plain correlated normals: MarkovRegression itself
    assumes Gaussian errors within each regime, so Gaussian shocks there
    already match what was fitted (and simulate_markov_forward draws its own
    shocks independently rather than consuming this array)."""
    rng = np.random.default_rng(seed)
    indep_normal = rng.standard_normal((n_sims, horizon_days, len(tickers)))
    corr_normal = indep_normal @ L.T if L is not None else indep_normal

    shocks = corr_normal.copy()
    for i, ticker in enumerate(tickers):
        best = best_models[ticker]
        if best["kind"] != "garch":
            continue
        nu = float(best["fitted"].params["nu"])
        u = np.clip(norm.cdf(corr_normal[:, :, i]), 1e-12, 1 - 1e-12)
        std_dev = np.sqrt(nu / (nu - 2))
        shocks[:, :, i] = t_dist.ppf(u, df=nu) / std_dev
    return shocks


def estimate_shock_correlation(best_models, returns_pct_df, fit_end):
    """Correlation of each name's OWN standardized shocks (eps_t / sigma_t
    for GARCH-family; (return - regime mean)/regime vol, weighted by
    filtered regime probability, for Markov-switching) over the shared fit
    window - real, estimated, from the same real data every other
    correlation figure in this repo comes from (see Multi-RC's README)."""
    standardized = {}
    for ticker, best in best_models.items():
        if best["kind"] == "garch":
            std_resid = best["fitted"].std_resid.loc[:fit_end]
            standardized[ticker] = std_resid
        else:
            k = best["k_regimes"]
            params = best["fitted"].params
            fit_data = returns_pct_df[ticker].loc[:fit_end]
            mod = MarkovRegression(fit_data, k_regimes=k, trend="c", switching_variance=True)
            filtered = mod.filter(params)
            probs = filtered.filtered_marginal_probabilities.values  # (n, k)
            const = np.array([params[f"const[{i}]"] for i in range(k)])
            sigma = np.array([np.sqrt(params[f"sigma2[{i}]"]) for i in range(k)])
            expected_mean = probs @ const
            expected_sigma = probs @ sigma
            standardized[ticker] = pd.Series((fit_data.values - expected_mean) / expected_sigma, index=fit_data.index)

    std_df = pd.concat(standardized, axis=1).dropna()
    return std_df.corr()


# Rolling probability-of-breach: at each historical date, simulate forward
# to maturity using only information available up to that date, and read
# off the empirical fraction of paths that breach - both at maturity and
# ever (touch) along the path.

def rolling_breach_probabilities(price_df, S0, best_models, returns_pct_df, fit_end, maturity_date,
                                  corr_matrix=None, n_sims=N_SIMULATIONS):
    tickers = list(price_df.columns)
    basket_mode = len(tickers) > 1
    n_sims = _safe_n_sims(n_sims, basket_mode)
    dates = price_df.index
    S0_series = pd.Series(S0, index=tickers)

    # BRC only: barrier knock-in is a one-time, irreversible event - once the
    # REAL (not simulated) worst-of path has ever closed below BARRIER as of
    # some rolling date, every later date treats the put as already active
    # regardless of what the simulated forward path does from there.
    barrier_touched_so_far = None
    if PRODUCT_TYPE == "BRC":
        worst_of_relative_hist = (price_df / S0_series).min(axis=1)
        barrier_touched_so_far = worst_of_relative_hist.cummin() < BARRIER

    terminal_probs, touch_probs = [], []
    L = np.linalg.cholesky(np.asarray(corr_matrix)) if (basket_mode and corr_matrix is not None) else None

    for date in dates:
        # STRIKE is fixed relative to S0 (entry), but every simulation below
        # produces returns relative to TODAY's spot at `date`. relative_today
        # re-bases the simulated outcome back onto S0, so a name that has
        # already moved is judged against its REMAINING distance to STRIKE,
        # not against a fresh full STRIKE move measured from today.
        relative_today = price_df.loc[date] / S0_series
        already_knocked_in = bool(barrier_touched_so_far.loc[date]) if PRODUCT_TYPE == "BRC" else False

        horizon_calendar = (maturity_date - date).days
        if horizon_calendar <= 0:
            worst_today = float(relative_today.min())
            if PRODUCT_TYPE == "BRC":
                touch_probs.append(1.0 if already_knocked_in else 0.0)
                terminal_probs.append(1.0 if (already_knocked_in and worst_today < STRIKE) else 0.0)
            else:
                terminal_probs.append(1.0 if worst_today < STRIKE else 0.0)
                touch_probs.append(1.0 if worst_today < STRIKE else 0.0)
            continue

        horizon = _calendar_to_trading_days(horizon_calendar)

        if not basket_mode:
            ticker = tickers[0]
            best = best_models[ticker]
            if best["kind"] == "garch":
                terminal_rel, path_min_rel = simulate_garch_forward(best, returns_pct_df[ticker], date, horizon, n_sims)
            else:
                terminal_rel, path_min_rel = simulate_markov_forward(best, returns_pct_df[ticker], date, horizon, n_sims)
            worst_terminal = terminal_rel * relative_today[ticker]
            worst_path_min = path_min_rel * relative_today[ticker]
        else:
            worst_terminal = np.ones(n_sims)
            worst_path_min = np.ones(n_sims)
            corr_shocks = _build_basket_shocks(best_models, tickers, L, n_sims, horizon,
                                                seed=RNG_SEED + hash(str(date)) % 10_000)

            for i, ticker in enumerate(tickers):
                best = best_models[ticker]
                if best["kind"] == "garch":
                    terminal_rel, path_min_rel = simulate_basket_garch_forward(
                        best, returns_pct_df[ticker], date, horizon, corr_shocks[:, :, i], n_sims)
                else:
                    terminal_rel, path_min_rel = simulate_markov_forward(
                        best, returns_pct_df[ticker], date, horizon, n_sims,
                        seed=RNG_SEED + hash(str(date) + ticker) % 10_000)
                terminal_from_entry = terminal_rel * relative_today[ticker]
                path_min_from_entry = path_min_rel * relative_today[ticker]
                worst_terminal = np.minimum(worst_terminal, terminal_from_entry)
                worst_path_min = np.minimum(worst_path_min, path_min_from_entry)

        # PRODUCT_TYPE semantics (see the config block up top): BRC's "breach"
        # is a joint, path-dependent condition (barrier ever touched AND
        # terminal < STRIKE); FCN/RC/GENERIC just compare each independently
        # against the one STRIKE level.
        if PRODUCT_TYPE == "BRC":
            touch = np.ones(n_sims, dtype=bool) if already_knocked_in else (worst_path_min < BARRIER)
            breach = touch & (worst_terminal < STRIKE)
        else:
            touch = worst_path_min < STRIKE
            breach = worst_terminal < STRIKE

        terminal_probs.append(float(np.mean(breach)))
        touch_probs.append(float(np.mean(touch)))

    return pd.Series(terminal_probs, index=dates), pd.Series(touch_probs, index=dates)


# ---------------------------------------------------------------------------
# Walk-forward validation: does the predicted probability actually mean
# anything? Fitting well (AIC/BIC) and predicting well (calibrated on
# unseen data) are different claims - this checks the second one. See
# README for the full design (calibration, Brier score, baselines).

def refit_best_model(returns_pct, as_of_date, basket_mode=False, verbose=False):
    """'Self-learning' step: re-runs the full AIC/BIC candidate search on an
    expanding window through as_of_date. Used both for validation refits and
    for the final live model."""
    garch_candidates = fit_garch_family_candidates(returns_pct, as_of_date, basket_mode=basket_mode)
    markov_candidates = fit_markov_candidates(returns_pct, as_of_date) if INCLUDE_MARKOV_SWITCHING else []
    best, _ = select_best_model(garch_candidates + markov_candidates, f"as of {as_of_date.date()}", verbose=verbose)
    return best


def brier_score(predicted, actual):
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    return float(np.mean((predicted - actual) ** 2))


def calibration_table(predicted, actual, n_buckets=5):
    df = pd.DataFrame({"predicted": predicted, "actual": actual})
    n_unique = df["predicted"].nunique()
    df["bucket"] = pd.qcut(df["predicted"], q=min(n_buckets, n_unique), duplicates="drop")
    table = df.groupby("bucket", observed=True).agg(
        n=("actual", "size"), mean_predicted=("predicted", "mean"), empirical_frequency=("actual", "mean")
    ).reset_index()
    return table


def naive_historical_baseline(price_series, as_of_date, horizon_trading_days, strike):
    """Walk-forward-safe naive baseline: the empirical breach frequency among
    all horizon_trading_days-ahead returns observable using ONLY price
    history up to and including as_of_date. This is a forecast that could
    actually have been made AT that checkpoint - unlike a constant equal to
    the mean of the validation set's own outcomes, which requires already
    knowing what happened across the whole held-out period.

    Deliberately terminal-only even under PRODUCT_TYPE='BRC' - a naive
    baseline that also modeled the barrier's path-dependency would need to
    scan every historical window for an interim breach, not just compare
    endpoints; this simpler proxy is a reasonable "how often would the
    terminal condition alone have been breached" floor, not a full BRC
    baseline."""
    history = price_series.loc[:as_of_date]
    fwd_relative = history.pct_change(horizon_trading_days).dropna() + 1.0
    if fwd_relative.empty:
        return float("nan")
    return float((fwd_relative < strike).mean())


def walk_forward_validate(ticker, full_returns_pct, full_price_series, validation_start, validation_end,
                           tenor_years, strike, refit_every=REFIT_FREQUENCY_DAYS, eval_every=EVAL_FREQUENCY_DAYS,
                           n_sims=N_SIMULATIONS):
    """Single-asset only. Walks forward in trading days, evaluating every
    eval_every days and refitting every refit_every days."""
    n_sims = _safe_n_sims(n_sims, basket_mode=False)
    horizon_calendar_days = round(tenor_years * CALENDAR_DAYS_PER_YEAR)
    horizon_trading_days = _calendar_to_trading_days(horizon_calendar_days)  # simulation step
    # count - the model advances one FITTED TRADING-day return per step, so this
    # must not be the calendar-day count used just below to locate target_date.
    trading_dates = full_price_series.index
    checkpoints = trading_dates[(trading_dates >= validation_start) & (trading_dates <= validation_end)][::eval_every]

    current_model, last_refit_date = None, None
    records = []

    for date in checkpoints:
        if last_refit_date is None or (date - last_refit_date).days >= refit_every / 252 * 365.25:
            # Pass the FULL (unsliced) series - fit_garch_family_candidates's
            # own last_obs=date restricts ESTIMATION to pre-date data, but the
            # arch_model object itself needs data extending past date so that
            # later .forecast(start=<any later checkpoint>) calls (before the
            # NEXT refit) have somewhere to originate from.
            current_model = refit_best_model(full_returns_pct, date, verbose=False)
            last_refit_date = date
            print(f"  [refit @ {date.date()}, {len(full_returns_pct.loc[:date]):,} obs] -> {current_model['label']}")
            gc.collect()  # discarded candidate models (arch/statsmodels result objects) each hold
            # real internal state; explicit collection keeps memory bounded across many refits.

        target_pos = trading_dates.searchsorted(date + pd.Timedelta(days=horizon_calendar_days))
        if target_pos >= len(trading_dates):
            continue
        target_date = trading_dates[target_pos]

        if current_model["kind"] == "garch":
            terminal_rel, path_min_rel = simulate_garch_forward(current_model, full_returns_pct, date, horizon_trading_days, n_sims)
        else:
            terminal_rel, path_min_rel = simulate_markov_forward(current_model, full_returns_pct, date, horizon_trading_days, n_sims)

        # Each checkpoint is treated as a fresh note starting at `date` (same
        # convention as actual_relative below) - so for BRC there's no prior
        # "already knocked in" state to carry in from an earlier checkpoint,
        # unlike the live rolling estimate's single continuous note.
        if PRODUCT_TYPE == "BRC":
            predicted_prob = float(np.mean((path_min_rel < BARRIER) & (terminal_rel < strike)))
        else:
            predicted_prob = float(np.mean(terminal_rel < strike))
        naive_predicted = naive_historical_baseline(full_price_series, date, horizon_trading_days, strike)

        actual_relative = float(full_price_series.loc[target_date] / full_price_series.loc[date])
        if PRODUCT_TYPE == "BRC":
            path_between_relative = full_price_series.loc[date:target_date] / full_price_series.loc[date]
            actual_barrier_touched = bool((path_between_relative < BARRIER).any())
            actual_breach = 1.0 if (actual_barrier_touched and actual_relative < strike) else 0.0
        else:
            actual_breach = 1.0 if actual_relative < strike else 0.0

        records.append({"date": date, "target_date": target_date, "predicted": predicted_prob,
                         "naive_predicted": naive_predicted,
                         "actual": actual_breach, "model": current_model["label"]})

    return pd.DataFrame(records)


def plot_validation(results_df, ticker_label, ticker, strike, brier, brier_naive):
    calib = calibration_table(results_df["predicted"], results_df["actual"], n_buckets=5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="lightgrey", linewidth=1, linestyle="dashed", label="Perfect calibration")
    ax.scatter(calib["mean_predicted"], calib["empirical_frequency"], s=calib["n"] * 3, color="darkred",
               alpha=0.8, label="This model (bucket size = # predictions)")
    ax.set_xlabel("Mean Predicted P(breach)")
    ax.set_ylabel("Empirical (Actual) Breach Frequency")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Calibration: predicted probability vs. what actually happened")
    ax.legend(fontsize=8)
    ax.grid(True, color="lightgrey", linewidth=0.4)

    ax = axes[1]
    bars = ax.bar(["This model", "Naive\n(historical rate)"],
                   [brier, brier_naive], color=["darkred", "grey"])
    ax.axhline(0.25, color="lightgrey", linewidth=0.8, linestyle="dotted")
    ax.text(1.4, 0.25, " \"always guess 50%\"\n baseline", fontsize=7, color="grey", va="center")
    ax.set_ylabel("Brier Score (lower = better)")
    ax.set_title("Predictive accuracy vs. naive baselines")
    ax.grid(True, axis="y", color="lightgrey", linewidth=0.4)

    barrier_desc = f", barrier={BARRIER:.0%}" if PRODUCT_TYPE == "BRC" else ""
    fig.suptitle(f"{ticker_label} - Walk-Forward Validation [{PRODUCT_LABELS[PRODUCT_TYPE]}] "
                 f"(strike={strike:.0%}{barrier_desc}, {len(results_df)} out-of-sample predictions)")
    plt.tight_layout()
    safe_ticker = ticker.replace("^", "").replace("/", "-")
    out_path = os.path.join(SCRIPT_DIR, f"{SCRIPT_BASENAME} - Validation ({safe_ticker}).png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Validation chart saved to {out_path}")
    plt.close()


def plot_rolling_probability(price_df, S0_list, tickers, underlying_names, terminal_probs, touch_probs,
                              best_models_label):
    relative = price_df / pd.Series(S0_list, index=price_df.columns)
    worst_of = relative.min(axis=1)
    worst_of_pct = (worst_of - 1) * 100

    fig, ax1 = plt.subplots(figsize=(18, 8))
    asset_colors = ["steelblue", "darkorange", "seagreen", "mediumorchid", "peru", "slategray"]
    for i, ticker in enumerate(tickers):
        rel_i = (price_df[ticker] / S0_list[i] - 1) * 100
        ax1.plot(rel_i.index, rel_i.values, color=asset_colors[i % len(asset_colors)], linewidth=1.0,
                 alpha=0.7, label=underlying_names[i])

    strike_pct = (STRIKE - 1) * 100
    ax1.axhline(strike_pct, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax1.text(0.01, strike_pct, f"Strike: {strike_pct:.1f}%", transform=ax1.get_yaxis_transform(),
              color="dodgerblue", fontsize=9, va="bottom", ha="left")

    if PRODUCT_TYPE == "BRC":
        barrier_pct = (BARRIER - 1) * 100
        ax1.axhline(barrier_pct, color="darkgreen", linewidth=0.8, linestyle="dotted")
        ax1.text(0.01, barrier_pct, f"Barrier: {barrier_pct:.1f}%", transform=ax1.get_yaxis_transform(),
                  color="darkgreen", fontsize=9, va="bottom", ha="left")

    ax1.set_ylabel("Return from Entry (%)")
    ax1.set_xlabel("Date")
    ax1.grid(True, color="lightgrey", linewidth=0.4)

    if PRODUCT_TYPE == "BRC":
        terminal_label = "P(barrier touched AND finishes below strike) - rolling"
        touch_label = "P(barrier EVER touched, i.e. knocked in) - rolling"
    else:
        terminal_label = "P(breach AT MATURITY) - rolling"
        touch_label = "P(EVER touches strike) - rolling"

    ax2 = ax1.twinx()
    l1, = ax2.plot(terminal_probs.index, terminal_probs.values * 100, color="darkred", linewidth=1.8,
                    label=terminal_label)
    l2, = ax2.plot(touch_probs.index, touch_probs.values * 100, color="firebrick", linewidth=1.3,
                    linestyle="dashed", label=touch_label)
    ax2.set_ylabel("Estimated Breach Probability (%)", color="darkred", rotation=270, labelpad=15)
    ax2.tick_params(axis="y", labelcolor="darkred")
    ax2.set_ylim(-2, 102)

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1 + [l1, l2], labels1 + [l1.get_label(), l2.get_label()], loc="upper left", fontsize=9)

    basket_label = " / ".join(underlying_names)
    strike_barrier_desc = f"Strike={STRIKE:.0%}" + (f", Barrier={BARRIER:.0%}" if PRODUCT_TYPE == "BRC" else "")
    ax1.set_title(f"{basket_label} — Real-World Rolling Breach Probability ({best_models_label})\n"
                  f"vs Actual Historical Price Path [{PRODUCT_LABELS[PRODUCT_TYPE]}] ({strike_barrier_desc})")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    _cleanup_stale_validation_pngs()

    underlying_names = [fetch_underlying_name(t) for t in TICKERS]
    basket_mode = len(TICKERS) > 1
    print(f"{'Basket' if basket_mode else 'Underlying'}: "
          f"{', '.join(f'{n} ({t})' for n, t in zip(underlying_names, TICKERS))}")

    entry_ts = pd.Timestamp(ENTRY_DATE)
    maturity_ts = entry_ts + pd.Timedelta(days=round(TENOR * 365.25))
    history_start = entry_ts - pd.Timedelta(days=round(FULL_HISTORY_YEARS * 365.25))
    validation_start = entry_ts - pd.Timedelta(days=round(VALIDATION_YEARS * 365.25))
    validation_end = entry_ts - pd.Timedelta(days=round(TENOR * 365.25))  # strictly before entry - no overlap
    # with the live forecast window; the last checkpoint's own TENOR-years-out
    # outcome lands exactly at ENTRY_DATE, never beyond it.

    print(f"\nFetching {FULL_HISTORY_YEARS}y of history ending {ENTRY_DATE} (covers the initial training")
    print(f"window, the {VALIDATION_YEARS}y held-out walk-forward validation period, and the live")
    print(f"{ENTRY_DATE}-to-maturity backtest window, fetched together for one consistent price series)...")
    full_price_df = fetch_multi_asset_path(TICKERS, history_start, maturity_ts)
    full_returns_df = log_returns_pct(full_price_df)

    last_available_date = full_price_df.index.max()
    note_matured = maturity_ts <= last_available_date
    if entry_ts > last_available_date:
        gap_days = (entry_ts - last_available_date).days
        if gap_days > 10:
            raise RuntimeError(
                f"ENTRY_DATE ({ENTRY_DATE}) is {gap_days} days after the last available real "
                f"trading day ({last_available_date.date()}) - check TICKERS/ENTRY_DATE.")
        print(f"\nNote: ENTRY_DATE ({ENTRY_DATE}) is a weekend/holiday or newer than the most "
              f"recently available close - using {last_available_date.date()} instead.")
        entry_ts = last_available_date
        maturity_ts = entry_ts + pd.Timedelta(days=round(TENOR * 365.25))
        note_matured = maturity_ts <= last_available_date

    path_price_df = full_price_df.loc[entry_ts:min(maturity_ts, last_available_date)]
    if path_price_df.empty:
        raise RuntimeError(f"No trading days found between {ENTRY_DATE} and "
                            f"{min(maturity_ts, last_available_date).date()} for {TICKERS}.")
    S0_list = [float(path_price_df[t].iloc[0]) for t in TICKERS]

    if not note_matured:
        print(f"\nNote: maturity ({maturity_ts.date()}) is AFTER the last available real trading day "
              f"({last_available_date.date()}) - this note hasn't matured yet. The rolling estimate below")
        print(f"runs through {last_available_date.date()} only; there's no 'actual outcome' to compare")
        print(f"against since it hasn't happened.")

    print(f"\n{'#' * 70}\nWALK-FORWARD VALIDATION [{PRODUCT_LABELS[PRODUCT_TYPE]}] (held out "
          f"{validation_start.date()} to {validation_end.date()},\nnever overlapping the live "
          f"{ENTRY_DATE} forecast below)\n{'#' * 70}")
    validation_results = {}
    for ticker, name in zip(TICKERS, underlying_names):
        print(f"\n--- {name} ({ticker}) ---")
        results_df = walk_forward_validate(ticker, full_returns_df[ticker], full_price_df[ticker],
                                            validation_start, validation_end, TENOR, STRIKE)
        validation_results[ticker] = results_df

        b = brier_score(results_df["predicted"], results_df["actual"])
        # Naive baseline: naive_predicted is a PER-CHECKPOINT historical breach
        # frequency computed from only the price history available as of that
        # checkpoint (see naive_historical_baseline) - a forecast actually
        # available at each date, not the mean of the validation set's own
        # (future, at checkpoint time) outcomes.
        b_naive = brier_score(results_df["naive_predicted"], results_df["actual"])

        print(f"\n  {len(results_df)} out-of-sample (predicted, actual) pairs, "
              f"{results_df['actual'].mean():.1%} actually breached")
        print(f"  Brier score - this model:              {b:.4f}")
        print(f"  Brier score - naive (historical rate):  {b_naive:.4f}")
        print(f"  (lower is better; 0.25 = the 'always guess 50%' baseline under class balance)")
        print(f"\n  Calibration table (predicted probability bucket vs. what actually happened):")
        print(calibration_table(results_df["predicted"], results_df["actual"]).to_string(
            index=False, float_format=lambda x: f"{x:.3f}"))

        plot_validation(results_df, f"{name} ({ticker})", ticker, STRIKE, b, b_naive)

    print(f"\n{'#' * 70}\nLIVE MODEL (fit on all data through {ENTRY_DATE} - the last link in the same")
    print(f"walk-forward chain validated above, applied to the actual note)\n{'#' * 70}")
    fit_end = entry_ts
    best_models = {}
    for ticker, name in zip(TICKERS, underlying_names):
        print(f"\n{'=' * 70}\nFitting candidate models for {name} ({ticker})\n{'=' * 70}")
        best = refit_best_model(full_returns_df[ticker], fit_end, basket_mode=basket_mode, verbose=True)
        best_models[ticker] = best
        print(f"  >>> Selected: {best['label']} <<<")
        gc.collect()

    corr_matrix = None
    if basket_mode:
        print(f"\n{'=' * 70}\nEstimating shock correlation across the basket (real, from the same")
        print(f"fit window, standardized by each name's OWN fitted conditional mean/vol)\n{'=' * 70}")
        corr_df = estimate_shock_correlation(best_models, full_returns_df, fit_end)
        print(corr_df.to_string(float_format=lambda x: f"{x:.3f}"))
        corr_matrix = corr_df.loc[TICKERS, TICKERS].values

    print(f"\n{'=' * 70}\nRunning the sequential rolling Monte Carlo (this simulates forward from")
    print(f"EVERY date in the historical path to the fixed maturity date, {N_SIMULATIONS:,} paths each)")
    print(f"{'=' * 70}")
    terminal_probs, touch_probs = rolling_breach_probabilities(
        path_price_df, S0_list, best_models, full_returns_df, fit_end, maturity_ts, corr_matrix)

    if PRODUCT_TYPE == "BRC":
        terminal_desc, touch_desc = "P(barrier touched AND finishes below strike)", "P(barrier EVER touched, i.e. knocked in)"
    else:
        terminal_desc, touch_desc = "P(breach at maturity)", "P(ever touches strike)"

    print(f"\n{'=' * 70}\nRESULTS [{PRODUCT_LABELS[PRODUCT_TYPE]}]\n{'=' * 70}")
    print(f"Entry Date: {ENTRY_DATE}   Maturity Date: {maturity_ts.date()}   Strike: {STRIKE:.0%}"
          + (f"   Barrier: {BARRIER:.0%}" if PRODUCT_TYPE == "BRC" else ""))
    print(f"\nDay-1 estimated {terminal_desc}:  {terminal_probs.iloc[0]:.2%}")
    print(f"Day-1 estimated {touch_desc}:  {touch_probs.iloc[0]:.2%}")
    print(f"\nMost recent rolling estimate ({path_price_df.index[-1].date()}): {terminal_probs.iloc[-1]:.2%}")

    if note_matured:
        actual_relative = path_price_df / pd.Series(S0_list, index=path_price_df.columns)
        actual_worst_of_T = float(actual_relative.iloc[-1].min())
        if PRODUCT_TYPE == "BRC":
            actual_barrier_touched = bool((actual_relative.min(axis=1) < BARRIER).any())
            actually_breached = actual_barrier_touched and actual_worst_of_T < STRIKE
            print(f"What ACTUALLY happened in this historical path: "
                  f"barrier {'was' if actual_barrier_touched else 'was NOT'} touched; "
                  f"worst-of relative performance at maturity = {actual_worst_of_T:.2%} "
                  f"({'BREACHED' if actually_breached else 'did not breach'} - {STRIKE:.0%} strike, "
                  f"{BARRIER:.0%} barrier)")
        else:
            actually_breached = actual_worst_of_T < STRIKE
            print(f"What ACTUALLY happened in this historical path: "
                  f"worst-of relative performance at maturity = {actual_worst_of_T:.2%} "
                  f"({'BREACHED' if actually_breached else 'did not breach'} the {STRIKE:.0%} strike)")
    else:
        print(f"Note matures {maturity_ts.date()} - still {(maturity_ts - last_available_date).days} days out, "
              f"actual outcome not yet known.")

    if len(path_price_df) < 2:
        print(f"\nSkipping the rolling-probability chart - only {len(path_price_df)} real trading day(s) "
              f"between entry and today, not enough to plot a path (the note hasn't started running yet).")
    else:
        plot_rolling_probability(path_price_df, S0_list, TICKERS, underlying_names, terminal_probs, touch_probs,
                                  " / ".join(f"{t}: {best_models[t]['label']}" for t in TICKERS))
