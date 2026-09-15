import gc
import os
import resource
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

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
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# Real-world (physical-measure) probability estimator, not a pricer - see
# README.md for the full rationale, especially why this deliberately isn't
# risk-neutral (risk_neutral_comparison() below quantifies the difference).

# --- Product terms (same STRIKE/ENTRY_DATE/TENOR convention as the plain
# Fixed Coupon Note.py, one level up in Product MtM/Yield/) ---
STRIKE = 0.65                  # breach level, as a fraction of each name's own entry level
ENTRY_DATE = "2026-09-14"
TENOR = 0.503836

# 1 ticker = single-asset mode. 2+ = basket (worst-of) mode, same convention
# as Multi-RC / Multi-FCN. Try TICKERS = ["AAPL", "JPM", "XOM"].
TICKERS = ["COTN.SW", "STMPA.PA"]

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

RISK_FREE_RATE = 0.04           # only used in risk_neutral_comparison()

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
                                 # each refit/checkpoint fits ~11 models from scratch, so the total
                                 # cost scales directly with VALIDATION_YEARS / EVAL_FREQUENCY_DAYS
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
    ("GARCH(1,1)", dict(vol="GARCH", p=1, o=0, q=1)),
    ("GJR-GARCH(1,1,1)", dict(vol="GARCH", p=1, o=1, q=1)),
    ("EGARCH(1,1,1)", dict(vol="EGARCH", p=1, o=1, q=1)),
]

MARKOV_K_REGIMES = [2, 3]


def fit_garch_family_candidates(returns_pct, fit_end, basket_mode=False):
    """Fits every mean x vol combo above (drops EGARCH when basket_mode=True).
    Estimation uses only data through fit_end - no look-ahead."""
    vol_specs = [s for s in GARCH_VOL_SPECS if not (basket_mode and s[0] == "EGARCH(1,1,1)")]
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
    fixed = am.fix(best["fitted"].params)
    fc = fixed.forecast(horizon=horizon_days, method="simulation",
                         simulations=n_sims, reindex=False,
                         rng=np.random.default_rng(seed).standard_normal)
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
    beta = params["beta[1]"]
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

    terminal_probs, touch_probs = [], []
    L = np.linalg.cholesky(np.asarray(corr_matrix)) if (basket_mode and corr_matrix is not None) else None

    for date in dates:
        horizon = (maturity_date - date).days
        if horizon <= 0:
            relative_today = (price_df.loc[date] / S0).values
            worst_today = float(np.min(relative_today))
            terminal_probs.append(1.0 if worst_today < STRIKE else 0.0)
            touch_probs.append(1.0 if worst_today < STRIKE else 0.0)
            continue

        if not basket_mode:
            best = best_models[tickers[0]]
            if best["kind"] == "garch":
                terminal_rel, path_min_rel = simulate_garch_forward(best, returns_pct_df[tickers[0]], date, horizon, n_sims)
            else:
                terminal_rel, path_min_rel = simulate_markov_forward(best, returns_pct_df[tickers[0]], date, horizon, n_sims)
            terminal_probs.append(float(np.mean(terminal_rel < STRIKE)))
            touch_probs.append(float(np.mean(path_min_rel < STRIKE)))
        else:
            worst_terminal = np.ones(n_sims)
            worst_path_min = np.ones(n_sims)
            rng = np.random.default_rng(RNG_SEED + hash(str(date)) % 10_000)
            indep_shocks = rng.standard_normal((n_sims, horizon, len(tickers)))
            corr_shocks = indep_shocks @ L.T if L is not None else indep_shocks

            for i, ticker in enumerate(tickers):
                best = best_models[ticker]
                if best["kind"] == "garch":
                    terminal_rel, path_min_rel = simulate_basket_garch_forward(
                        best, returns_pct_df[ticker], date, horizon, corr_shocks[:, :, i], n_sims)
                else:
                    terminal_rel, path_min_rel = simulate_markov_forward(
                        best, returns_pct_df[ticker], date, horizon, n_sims,
                        seed=RNG_SEED + hash(str(date) + ticker) % 10_000)
                worst_terminal = np.minimum(worst_terminal, terminal_rel)
                worst_path_min = np.minimum(worst_path_min, path_min_rel)

            terminal_probs.append(float(np.mean(worst_terminal < STRIKE)))
            touch_probs.append(float(np.mean(worst_path_min < STRIKE)))

    return pd.Series(terminal_probs, index=dates), pd.Series(touch_probs, index=dates)


def risk_neutral_comparison(S0, sigma_realized, r, q, T, strike=STRIKE):
    """Risk-neutral (drift=r-q) GBM breach probability, closed-form - a
    theoretical contrast number (uses realized vol, not a real market-
    quoted implied vol - this file has no options data), not a real-world
    forecast and not a real market fair-value check."""
    d2 = (np.log(S0 / (strike * S0)) + (r - q - 0.5 * sigma_realized ** 2) * T) / (sigma_realized * np.sqrt(T))
    return norm.cdf(-d2)  # P(S_T < K) under GBM


def risk_neutral_worst_of_comparison(sigmas, corr_matrix, r, T, strike=STRIKE, n_sims=50000, seed=RNG_SEED):
    """Risk-neutral WORST-OF breach probability - the basket generalization
    of risk_neutral_comparison above, using the SAME real correlation
    matrix the real-world estimate uses, so the two numbers are an actual
    apples-to-apples comparison (not one basket figure next to N single-
    name ones). No closed form exists for an N>2 worst-of under GBM (same
    reasoning as Multi-RC/Multi-FCN), so this is a correlated Monte Carlo -
    cheap and exact enough here since there's no GARCH state to propagate,
    just a single terminal draw per path."""
    n = len(sigmas)
    L = np.linalg.cholesky(np.asarray(corr_matrix, dtype=float))
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_sims, n)) @ L.T
    sigmas = np.asarray(sigmas)
    terminal_relative = np.exp((r - 0.5 * sigmas ** 2) * T + sigmas * np.sqrt(T) * z)
    worst_of = terminal_relative.min(axis=1)
    return float(np.mean(worst_of < strike))


def realized_annualized_vol(path):
    log_returns = np.log(path / path.shift(1)).dropna()
    return log_returns.std() * np.sqrt(252)


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


def walk_forward_validate(ticker, full_returns_pct, full_price_series, validation_start, validation_end,
                           tenor_years, strike, refit_every=REFIT_FREQUENCY_DAYS, eval_every=EVAL_FREQUENCY_DAYS,
                           n_sims=N_SIMULATIONS):
    """Single-asset only. Walks forward in trading days, evaluating every
    eval_every days and refitting every refit_every days."""
    n_sims = _safe_n_sims(n_sims, basket_mode=False)
    horizon_days = round(tenor_years * 365.25)
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

        target_pos = trading_dates.searchsorted(date + pd.Timedelta(days=horizon_days))
        if target_pos >= len(trading_dates):
            continue
        target_date = trading_dates[target_pos]

        if current_model["kind"] == "garch":
            terminal_rel, _ = simulate_garch_forward(current_model, full_returns_pct, date, horizon_days, n_sims)
        else:
            terminal_rel, _ = simulate_markov_forward(current_model, full_returns_pct, date, horizon_days, n_sims)
        predicted_prob = float(np.mean(terminal_rel < strike))

        actual_relative = float(full_price_series.loc[target_date] / full_price_series.loc[date])
        actual_breach = 1.0 if actual_relative < strike else 0.0

        records.append({"date": date, "target_date": target_date, "predicted": predicted_prob,
                         "actual": actual_breach, "model": current_model["label"]})

    return pd.DataFrame(records)


def plot_validation(results_df, ticker_label, ticker, strike, brier, brier_naive, brier_rn):
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
    bars = ax.bar(["This model", "Naive\n(historical rate)", "Risk-neutral\n(GBM)"],
                   [brier, brier_naive, brier_rn], color=["darkred", "grey", "steelblue"])
    ax.axhline(0.25, color="lightgrey", linewidth=0.8, linestyle="dotted")
    ax.text(2.4, 0.25, " \"always guess 50%\"\n baseline", fontsize=7, color="grey", va="center")
    ax.set_ylabel("Brier Score (lower = better)")
    ax.set_title("Predictive accuracy vs. naive baselines")
    ax.grid(True, axis="y", color="lightgrey", linewidth=0.4)

    fig.suptitle(f"{ticker_label} - Walk-Forward Validation (strike={strike:.0%}, "
                 f"{len(results_df)} out-of-sample predictions)")
    plt.tight_layout()
    safe_ticker = ticker.replace("^", "").replace("/", "-")
    out_path = os.path.join(SCRIPT_DIR, f"FCN Probability Estimator - Validation ({safe_ticker}).png")
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
    ax1.set_ylabel("Return from Entry (%)")
    ax1.set_xlabel("Date")
    ax1.grid(True, color="lightgrey", linewidth=0.4)

    ax2 = ax1.twinx()
    l1, = ax2.plot(terminal_probs.index, terminal_probs.values * 100, color="darkred", linewidth=1.8,
                    label="P(breach AT MATURITY) - rolling")
    l2, = ax2.plot(touch_probs.index, touch_probs.values * 100, color="firebrick", linewidth=1.3,
                    linestyle="dashed", label="P(EVER touches strike) - rolling")
    ax2.set_ylabel("Estimated Breach Probability (%)", color="darkred", rotation=270, labelpad=15)
    ax2.tick_params(axis="y", labelcolor="darkred")
    ax2.set_ylim(-2, 102)

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1 + [l1, l2], labels1 + [l1.get_label(), l2.get_label()], loc="upper left", fontsize=9)

    basket_label = " / ".join(underlying_names)
    ax1.set_title(f"{basket_label} — Real-World Rolling Breach Probability ({best_models_label})\n"
                  f"vs Actual Historical Price Path (Strike={STRIKE:.0%})")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
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

    print(f"\n{'#' * 70}\nWALK-FORWARD VALIDATION (held out {validation_start.date()} to "
          f"{validation_end.date()},\nnever overlapping the live {ENTRY_DATE} forecast below)\n{'#' * 70}")
    validation_results = {}
    for ticker, name in zip(TICKERS, underlying_names):
        print(f"\n--- {name} ({ticker}) ---")
        results_df = walk_forward_validate(ticker, full_returns_df[ticker], full_price_df[ticker],
                                            validation_start, validation_end, TENOR, STRIKE)
        validation_results[ticker] = results_df

        naive_rate = float((full_price_df[ticker].loc[validation_start:validation_end].pct_change(
            round(TENOR * 365.25)).dropna() < (STRIKE - 1)).mean()) if len(results_df) else float("nan")
        sigma_hist = float(realized_annualized_vol(full_price_df[ticker].loc[:validation_start]))
        rn_prob = risk_neutral_comparison(1.0, sigma_hist, RISK_FREE_RATE, 0.0, TENOR)

        b = brier_score(results_df["predicted"], results_df["actual"])
        b_naive = brier_score(np.full(len(results_df), results_df["actual"].mean()), results_df["actual"])
        b_rn = brier_score(np.full(len(results_df), rn_prob), results_df["actual"])

        print(f"\n  {len(results_df)} out-of-sample (predicted, actual) pairs, "
              f"{results_df['actual'].mean():.1%} actually breached")
        print(f"  Brier score - this model:              {b:.4f}")
        print(f"  Brier score - naive (historical rate):  {b_naive:.4f}")
        print(f"  Brier score - risk-neutral GBM:          {b_rn:.4f}")
        print(f"  (lower is better; 0.25 = the 'always guess 50%' baseline under class balance)")
        print(f"\n  Calibration table (predicted probability bucket vs. what actually happened):")
        print(calibration_table(results_df["predicted"], results_df["actual"]).to_string(
            index=False, float_format=lambda x: f"{x:.3f}"))

        plot_validation(results_df, f"{name} ({ticker})", ticker, STRIKE, b, b_naive, b_rn)

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

    print(f"\n{'=' * 70}\nRESULTS\n{'=' * 70}")
    print(f"Entry Date: {ENTRY_DATE}   Maturity Date: {maturity_ts.date()}   Strike: {STRIKE:.0%}")
    print(f"\nDay-1 estimated P(breach at maturity):  {terminal_probs.iloc[0]:.2%}")
    print(f"Day-1 estimated P(ever touches strike):  {touch_probs.iloc[0]:.2%}")
    print(f"\nMost recent rolling estimate ({path_price_df.index[-1].date()}): {terminal_probs.iloc[-1]:.2%}")

    if note_matured:
        actual_relative = path_price_df / pd.Series(S0_list, index=path_price_df.columns)
        actual_worst_of_T = float(actual_relative.iloc[-1].min())
        actually_breached = actual_worst_of_T < STRIKE
        print(f"What ACTUALLY happened in this historical path: "
              f"worst-of relative performance at maturity = {actual_worst_of_T:.2%} "
              f"({'BREACHED' if actually_breached else 'did not breach'} the {STRIKE:.0%} strike)")
    else:
        print(f"Note matures {maturity_ts.date()} - still {(maturity_ts - last_available_date).days} days out, "
              f"actual outcome not yet known.")

    print(f"\n--- Risk-neutral comparison (the point about Monte Carlo's expected value) ---")
    per_name_sigmas = [float(realized_annualized_vol(full_price_df[t].loc[:fit_end])) for t in TICKERS]
    for ticker, name, sigma_realized in zip(TICKERS, underlying_names, per_name_sigmas):
        rn_prob = risk_neutral_comparison(1.0, sigma_realized, RISK_FREE_RATE, 0.0, TENOR)
        print(f"  {name} alone: risk-neutral (GBM) P(breach) = {rn_prob:.2%}")

    if basket_mode:
        rn_worst_of = risk_neutral_worst_of_comparison(per_name_sigmas, corr_matrix, RISK_FREE_RATE, TENOR)
        print(f"\n  WORST-OF BASKET, apples-to-apples with the real-world estimate above (same real")
        print(f"  correlation matrix, correlated Monte Carlo - no closed form exists for N>2 worst-of):")
        print(f"    Risk-neutral P(breach) = {rn_worst_of:.2%}   vs.   real-world day-1 estimate = {terminal_probs.iloc[0]:.2%}")
    else:
        print(f"\n  vs. this file's real-world day-1 estimate = {terminal_probs.iloc[0]:.2%}")

    print(f"\n  These are answering DIFFERENT questions (no-arbitrage-consistent vs. actual-world-likely) -")
    print(f"  see the module docstring for why they're not expected to match, and shouldn't be used")
    print(f"  interchangeably.")

    if len(path_price_df) < 2:
        print(f"\nSkipping the rolling-probability chart - only {len(path_price_df)} real trading day(s) "
              f"between entry and today, not enough to plot a path (the note hasn't started running yet).")
    else:
        plot_rolling_probability(path_price_df, S0_list, TICKERS, underlying_names, terminal_probs, touch_probs,
                                  " / ".join(f"{t}: {best_models[t]['label']}" for t in TICKERS))
