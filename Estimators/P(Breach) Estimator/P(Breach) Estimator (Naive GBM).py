import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.rcParams["font.family"] = "Arial"
warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_BASENAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_PNG = os.path.join(SCRIPT_DIR, SCRIPT_BASENAME + ".png")

# A deliberately NAIVE real-world Monte Carlo: plain geometric Brownian motion, one
# constant (mu, sigma) per name estimated once from a trailing lookback window, one
# simulation batch for the live ENTRY_DATE-to-maturity note. No GARCH, no fat tails,
# no volatility clustering, no walk-forward "self-learning" refitting, and - unlike
# "P(Breach) Estimator.py" - NOT sequential/rolling: it doesn't re-simulate from
# every historical date, just once, from ENTRY_DATE. This isn't a competing model;
# it's the floor the GARCH-based estimator should be beating. See README.md's
# "Naive GBM benchmark" section and MATHEMATICS.md section 7.
#
# Same PRODUCT_TYPE/STRIKE/BARRIER/TICKERS/ENTRY_DATE/TENOR convention as
# "P(Breach) Estimator.py" (this folder), so the two are directly comparable.
PRODUCT_TYPE = "GENERIC"

STRIKE = 0.5
BARRIER = None
ENTRY_DATE = "2026-09-17"
TENOR = 1

if PRODUCT_TYPE not in ("FCN", "RC", "BRC", "GENERIC"):
    raise ValueError(f"PRODUCT_TYPE must be one of 'FCN', 'RC', 'BRC', 'GENERIC' - got {PRODUCT_TYPE!r}")
if PRODUCT_TYPE == "BRC" and not (BARRIER is not None and 0 < BARRIER < STRIKE):
    raise ValueError(f"PRODUCT_TYPE='BRC' requires 0 < BARRIER < STRIKE (got BARRIER={BARRIER!r}, STRIKE={STRIKE!r})")

PRODUCT_LABELS = {"FCN": "Fixed Coupon Note", "RC": "Reverse Convertible",
                   "BRC": "Barrier Reverse Convertible", "GENERIC": "General Probability Estimator"}

# 1 ticker = single-asset mode. 2+ = basket (worst-of) mode, same convention as
# P(Breach) Estimator.py / Multi-RC / Multi-FCN.
TICKERS = ["CL=F", "BZ=F"]

# Trailing window used to estimate the ONE constant (mu, sigma) pair per name - same
# 2y convention as Product MtM/_common.py's fetch_trailing_realized_vol. Estimated
# once, through ENTRY_DATE, never updated - that's the whole point of "naive."
GBM_LOOKBACK_YEARS = 2

N_SIMULATIONS = 1000000
N_SAMPLE_PATHS_PLOTTED = 200     # how many of the N_SIMULATIONS paths to draw on the fan chart
RNG_SEED = 42
TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365.25


def _calendar_to_trading_days(calendar_days):
    """Same conversion as P(Breach) Estimator.py - one simulated step is one
    trading-day return, so a calendar-day horizon needs converting first."""
    return max(1, round(calendar_days * TRADING_DAYS_PER_YEAR / CALENDAR_DAYS_PER_YEAR))


# --- Data fetching (duplicated from P(Breach) Estimator.py rather than imported -
# every script in this repo is standalone, see CLAUDE.md) ---

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


def log_returns(price_df):
    """Plain natural-log returns, NOT percent-scaled - unlike P(Breach) Estimator.py,
    there's no numerical optimizer here to stabilize (mu/sigma are closed-form sample
    statistics, not MLE), so no x100 scaling is needed."""
    return np.log(price_df / price_df.shift(1)).dropna()


# ---------------------------------------------------------------------------
# GBM parameter estimation: one constant (mu, sigma) per name, estimated once
# ---------------------------------------------------------------------------

def fit_gbm_params(returns_df):
    """Sample mean/std of daily log returns over the lookback window - the simplest
    possible real-world drift/vol estimate. Because these ARE log returns (not
    simple returns), the sample mean already IS the drift of ln(S) directly; no
    separate '-sigma^2/2' convexity correction is needed anywhere else in this file
    (see MATHEMATICS.md section 7 for why)."""
    mu = returns_df.mean()
    sigma = returns_df.std(ddof=1)
    return mu, sigma


def estimate_correlation(returns_df):
    """Real Pearson correlation of the plain log returns - directly usable to
    correlate Gaussian shocks under GBM with no copula step, unlike
    P(Breach) Estimator.py's Student's-t basket case (see MATHEMATICS.md section 7)."""
    return returns_df.corr()


# ---------------------------------------------------------------------------
# Forward simulation: ONE batch, from ENTRY_DATE straight to maturity
# ---------------------------------------------------------------------------

def simulate_gbm_forward(mu, sigma, tickers, horizon_days, n_sims, corr_matrix=None, seed=RNG_SEED):
    """iid Gaussian daily log-return shocks (Cholesky-correlated across names in
    basket mode) driven by the ONE fixed (mu, sigma) per name - no volatility
    clustering, no fat tails: exactly the two things GARCH + Student's t buy you
    over this file. Returns the FULL relative price path, shape
    (n_sims, horizon_days, n_tickers), already relative to S0 (the simulation
    origin IS ENTRY_DATE, so - unlike the rolling estimator - no re-basing is
    needed). Callers derive terminal/path-min/worst-of from this one array rather
    than re-simulating - see the __main__ block, which also reuses it for the
    fan chart."""
    rng = np.random.default_rng(seed)
    n_tickers = len(tickers)
    z = rng.standard_normal((n_sims, horizon_days, n_tickers))
    if corr_matrix is not None:
        L = np.linalg.cholesky(np.asarray(corr_matrix))
        z = z @ L.T

    mu_arr = mu[tickers].values[None, None, :]
    sigma_arr = sigma[tickers].values[None, None, :]
    log_ret = mu_arr + sigma_arr * z                       # (n_sims, horizon_days, n_tickers)
    cum_log = np.cumsum(log_ret, axis=1)
    return np.exp(cum_log)


# ---------------------------------------------------------------------------
# Closed-form benchmarks (single-asset only - see MATHEMATICS.md section 7 for
# why there's no simple closed form for a correlated worst-of basket)
# ---------------------------------------------------------------------------

def terminal_closed_form(level_relative, mu, sigma, horizon_days):
    """Exact lognormal terminal CDF under this file's own constant-(mu,sigma) GBM:
    P(S_T/S0 < level_relative). A sanity check on the simulation/RNG plumbing (the
    simulator's terminal draw IS this lognormal by construction, so any material gap
    here is a bug, not sampling noise), not a model-error check."""
    T = horizon_days
    z = (np.log(level_relative) - mu * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(z))


def barrier_touch_closed_form(level_relative, mu, sigma, horizon_days):
    """Exact CONTINUOUS-monitoring first-passage probability (reflection principle)
    for an arithmetic Brownian motion with drift, X_t = mu*t + sigma*W_t, hitting
    b = ln(level_relative) <= 0 by time T:
        P(min_{0<=t<=T} X_t <= b) = Φ((b-muT)/(sigma√T)) + exp(2*mu*b/sigma²)*Φ((b+muT)/(sigma√T))
    A verification benchmark only - this file's own simulation is DAILY-discretized
    (one shock per trading day), so a genuine, small discretization gap against this
    continuous benchmark is expected and not itself a bug (same role as Barrier
    Reverse Convertible's CRR-vs-AnalyticBarrierEngine check)."""
    b = np.log(level_relative)
    if b >= 0:
        return 1.0
    T = horizon_days
    d1 = (b - mu * T) / (sigma * np.sqrt(T))
    d2 = (b + mu * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1) + np.exp(2 * mu * b / sigma ** 2) * norm.cdf(d2))


# ---------------------------------------------------------------------------
# Plot: the real path vs. a fan of sample simulated paths, plus a terminal histogram
# ---------------------------------------------------------------------------

def plot_gbm_estimate(path_price_df, S0_list, tickers, underlying_names, sim_terminal_relative,
                       sim_path_relative_worst, mu, sigma, p_breach, p_touch, terminal_cf, touch_cf):
    S0_series = pd.Series(S0_list, index=tickers)
    actual_worst_of_pct = ((path_price_df / S0_series).min(axis=1) - 1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    horizon = sim_path_relative_worst.shape[1]
    plotted = min(N_SAMPLE_PATHS_PLOTTED, sim_path_relative_worst.shape[0])
    sample_idx = np.random.default_rng(RNG_SEED).choice(sim_path_relative_worst.shape[0], size=plotted, replace=False)
    x_days = np.arange(1, horizon + 1)
    for i in sample_idx:
        ax.plot(x_days, (sim_path_relative_worst[i] - 1) * 100, color="steelblue", linewidth=0.5, alpha=0.15)
    ax.plot([], [], color="steelblue", linewidth=1.5, label=f"{plotted} of {N_SIMULATIONS:,} simulated worst-of paths (naive GBM)")
    ax.plot(np.arange(1, len(actual_worst_of_pct) + 1), actual_worst_of_pct.values, color="firebrick",
             linewidth=1.8, label="Actual historical worst-of path")

    strike_pct = (STRIKE - 1) * 100
    ax.axhline(strike_pct, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.text(0.01, strike_pct, f"Strike: {strike_pct:.1f}%", transform=ax.get_yaxis_transform(),
            color="dodgerblue", fontsize=9, va="bottom", ha="left")
    if PRODUCT_TYPE == "BRC":
        barrier_pct = (BARRIER - 1) * 100
        ax.axhline(barrier_pct, color="darkgreen", linewidth=0.8, linestyle="dotted")
        ax.text(0.01, barrier_pct, f"Barrier: {barrier_pct:.1f}%", transform=ax.get_yaxis_transform(),
                color="darkgreen", fontsize=9, va="bottom", ha="left")

    ax.set_xlabel("Trading Days from Entry")
    ax.set_ylabel("Return from Entry (%)")
    ax.set_title("Naive GBM simulated paths vs. actual historical path")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, color="lightgrey", linewidth=0.4)

    ax = axes[1]
    terminal_pct = (sim_terminal_relative - 1) * 100
    ax.hist(terminal_pct, bins=80, color="steelblue", alpha=0.75)
    ax.axvline(strike_pct, color="dodgerblue", linewidth=1.2, linestyle="dotted", label=f"Strike ({strike_pct:.1f}%)")
    if PRODUCT_TYPE == "BRC":
        ax.axvline(barrier_pct, color="darkgreen", linewidth=1.2, linestyle="dotted", label=f"Barrier ({barrier_pct:.1f}%)")
    ax.set_xlabel("Simulated Terminal Worst-of Return (%)")
    ax.set_ylabel("Simulated Paths")
    ax.set_title("Distribution of simulated terminal outcomes")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", color="lightgrey", linewidth=0.4)

    if PRODUCT_TYPE == "BRC":
        stats_text = (f"P(barrier touched AND finishes below strike) = {p_breach:.2%}\n"
                       f"P(barrier EVER touched) = {p_touch:.2%}  (closed-form: {touch_cf:.2%})")
    else:
        stats_text = (f"P(breach at maturity) = {p_breach:.2%}  (closed-form: {terminal_cf:.2%})\n"
                       f"P(ever touches strike) = {p_touch:.2%}  (closed-form: {touch_cf:.2%})")
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=10)

    basket_label = " / ".join(underlying_names)
    mu_sigma_desc = ", ".join(f"{t}: μ={mu[t]*TRADING_DAYS_PER_YEAR:.1%}/yr, σ={sigma[t]*np.sqrt(TRADING_DAYS_PER_YEAR):.1%}/yr"
                               for t in tickers)
    fig.suptitle(f"{basket_label} — Naive Monte Carlo GBM Estimate [{PRODUCT_LABELS[PRODUCT_TYPE]}]\n{mu_sigma_desc}")
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
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
    lookback_start = entry_ts - pd.Timedelta(days=round(GBM_LOOKBACK_YEARS * 365.25))

    print(f"\nFetching {GBM_LOOKBACK_YEARS}y trailing history ending {ENTRY_DATE} (to fit the constant")
    print(f"GBM mu/sigma) plus the live {ENTRY_DATE}-to-maturity path (fetched together for one")
    print(f"consistent price series)...")
    full_price_df = fetch_multi_asset_path(TICKERS, lookback_start, maturity_ts)

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

    lookback_returns = log_returns(full_price_df.loc[:entry_ts])
    mu, sigma = fit_gbm_params(lookback_returns)
    print(f"\nGBM parameters (from {lookback_returns.index[0].date()} to {entry_ts.date()}, "
          f"{len(lookback_returns):,} trading days):")
    for t in TICKERS:
        print(f"  {t}: mu={mu[t]*TRADING_DAYS_PER_YEAR:.2%}/yr (drift of ln S), "
              f"sigma={sigma[t]*np.sqrt(TRADING_DAYS_PER_YEAR):.2%}/yr")

    corr_matrix = None
    if basket_mode:
        corr_df = estimate_correlation(lookback_returns)
        print(f"\nReturn correlation over the same lookback window:")
        print(corr_df.to_string(float_format=lambda x: f"{x:.3f}"))
        corr_matrix = corr_df.loc[TICKERS, TICKERS].values

    path_price_df = full_price_df.loc[entry_ts:min(maturity_ts, last_available_date)]
    if path_price_df.empty:
        raise RuntimeError(f"No trading days found between {ENTRY_DATE} and "
                            f"{min(maturity_ts, last_available_date).date()} for {TICKERS}.")
    S0_list = [float(path_price_df[t].iloc[0]) for t in TICKERS]

    if not note_matured:
        print(f"\nNote: maturity ({maturity_ts.date()}) is AFTER the last available real trading day "
              f"({last_available_date.date()}) - this note hasn't matured yet.")

    horizon_calendar = (maturity_ts - entry_ts).days
    horizon_days = _calendar_to_trading_days(horizon_calendar)
    print(f"\n{'=' * 70}\nRunning ONE Monte Carlo batch ({N_SIMULATIONS:,} paths, {horizon_days} trading "
          f"days) from {entry_ts.date()} straight to maturity - not rolling/sequential\n{'=' * 70}")
    # Full relative price path, shape (n_sims, horizon_days, n_tickers) - reused below
    # both for the breach/touch stats AND the fan chart, rather than re-simulating.
    sim_relative_path = simulate_gbm_forward(mu, sigma, TICKERS, horizon_days, N_SIMULATIONS, corr_matrix)
    worst_of_path = sim_relative_path.min(axis=2)           # (n_sims, horizon_days) - worst-of at every step
    worst_terminal = worst_of_path[:, -1]
    worst_path_min = worst_of_path.min(axis=1)

    if PRODUCT_TYPE == "BRC":
        touch = worst_path_min < BARRIER
        breach = touch & (worst_terminal < STRIKE)
    else:
        touch = worst_path_min < STRIKE
        breach = worst_terminal < STRIKE
    p_breach = float(np.mean(breach))
    p_touch = float(np.mean(touch))

    terminal_cf = touch_cf = float("nan")
    if not basket_mode:
        mu0, sigma0 = float(mu.iloc[0]), float(sigma.iloc[0])
        terminal_cf = terminal_closed_form(STRIKE, mu0, sigma0, horizon_days)
        touch_cf = barrier_touch_closed_form(BARRIER if PRODUCT_TYPE == "BRC" else STRIKE, mu0, sigma0, horizon_days)
        print(f"\nClosed-form verification (single-asset only - continuous-monitoring GBM, no")
        print(f"discretization, see MATHEMATICS.md section 7):")
        print(f"  Terminal lognormal CDF:         {terminal_cf:.2%}  (simulation sanity check)")
        print(f"  Reflection-principle touch prob: {touch_cf:.2%}  (expect a small gap vs. simulation -")
        print(f"    this file's simulation is daily-discretized, the closed form is continuous)")
    else:
        print(f"\nNo closed-form check in basket mode - no simple closed form for a correlated")
        print(f"worst-of hitting probability (see MATHEMATICS.md section 7).")

    if PRODUCT_TYPE == "BRC":
        terminal_desc, touch_desc = "P(barrier touched AND finishes below strike)", "P(barrier EVER touched)"
    else:
        terminal_desc, touch_desc = "P(breach at maturity)", "P(ever touches strike)"

    print(f"\n{'=' * 70}\nRESULTS [{PRODUCT_LABELS[PRODUCT_TYPE]}] - Naive GBM\n{'=' * 70}")
    print(f"Entry Date: {entry_ts.date()}   Maturity Date: {maturity_ts.date()}   Strike: {STRIKE:.0%}"
          + (f"   Barrier: {BARRIER:.0%}" if PRODUCT_TYPE == "BRC" else ""))
    print(f"{terminal_desc}: {p_breach:.2%}")
    print(f"{touch_desc}: {p_touch:.2%}")

    if note_matured:
        actual_relative = path_price_df / pd.Series(S0_list, index=path_price_df.columns)
        actual_worst_of_T = float(actual_relative.iloc[-1].min())
        if PRODUCT_TYPE == "BRC":
            actual_barrier_touched = bool((actual_relative.min(axis=1) < BARRIER).any())
            actually_breached = actual_barrier_touched and actual_worst_of_T < STRIKE
            print(f"\nWhat ACTUALLY happened: barrier {'was' if actual_barrier_touched else 'was NOT'} touched; "
                  f"worst-of at maturity = {actual_worst_of_T:.2%} "
                  f"({'BREACHED' if actually_breached else 'did not breach'})")
        else:
            actually_breached = actual_worst_of_T < STRIKE
            print(f"\nWhat ACTUALLY happened: worst-of relative performance at maturity = "
                  f"{actual_worst_of_T:.2%} ({'BREACHED' if actually_breached else 'did not breach'} "
                  f"the {STRIKE:.0%} strike)")
    else:
        print(f"\nNote matures {maturity_ts.date()} - still {(maturity_ts - last_available_date).days} "
              f"days out, actual outcome not yet known.")

    if len(path_price_df) < 2:
        print(f"\nSkipping the chart - only {len(path_price_df)} real trading day(s) between entry and "
              f"today, not enough to plot a path.")
    else:
        plot_gbm_estimate(path_price_df, S0_list, TICKERS, underlying_names, worst_terminal,
                           worst_of_path, mu, sigma, p_breach, p_touch, terminal_cf, touch_cf)
