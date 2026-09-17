import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql

plt.rcParams["font.family"] = "Arial"

import sys

_PRODUCT_MTM_ROOT = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(_PRODUCT_MTM_ROOT) != "Product MtM":
    _PRODUCT_MTM_ROOT = os.path.dirname(_PRODUCT_MTM_ROOT)
if _PRODUCT_MTM_ROOT not in sys.path:
    sys.path.insert(0, _PRODUCT_MTM_ROOT)

from _common import (
    fetch_daily_closes,
    fetch_dividend_yield,
    fetch_underlying_name,
    _quantlib_process,
    zcb_price_and_greeks,
    fetch_risk_free_rate,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Product terms (same base terms as "Multi-RC.py",
# Product MtM/Yield/Reverse Convertible/Multi-RC/, plus the autocall
# feature from the single-name "Fixed Coupon Note (Autocallable).py",
# one folder up) ---
STRIKE = 0.90                  # worst-of put strike, as a fraction of each name's OWN entry level
TRIGGER = 1.00                 # autocall level, as a fraction of each name's OWN entry level -
                                # ALL names must be at/above this on an observation date for the
                                # note to call (worst-of >= TRIGGER means every single name is)
OBS_PER_YEAR = 4                # quarterly observation dates
GS_CDS_SPREAD = 0.002675      # Goldman Sachs 1y CDS, 26.75 bps (Investing.com) - issuer credit spread, ZCB leg only - tenor-matched to TENOR=1, not the 5y CDS an earlier version of this repo used

ENTRY_DATE = "2025-01-02"
TENOR = 1
RISK_FREE_RATE = fetch_risk_free_rate(ENTRY_DATE)  # 1Y Treasury CMT (FRED DGS1) as of ENTRY_DATE - real, historical; option leg pricing and MC risk-neutral drift

TICKERS = ["AAPL", "JPM", "XOM"]           # same default basket as Multi-RC, for direct comparison
CORRELATION_LOOKBACK_YEARS = 2

N_MC_PATHS = 100000
MC_SEED = 42

SPOT_SCENARIO_RANGE = np.arange(0.60, 1.41, 0.05)  # 60% to 140% of S0, in 5pt steps


def fetch_multi_asset_path(tickers, start, end):
    """Daily close series for every basket name, aligned on shared trading
    dates (inner join) - see Multi-RC.py for the full rationale."""
    series = {ticker: fetch_daily_closes(ticker, start, end) for ticker in tickers}
    df = pd.concat(series, axis=1)
    df = df.dropna(how="any")
    if df.index[-1] < end - pd.Timedelta(days=10):
        raise RuntimeError(
            f"Requested window {start.date()} to {end.date()} extends past the last available "
            f"trading day ({df.index[-1].date()}) - this script models a COMPLETED historical "
            f"window, not a live in-progress note. Pick an ENTRY_DATE/TENOR combination that ends "
            f"on or before today."
        )
    return df


def estimate_vols_and_correlation(calibration_price_df):
    """Realized annualized vol per name and the full correlation matrix,
    both from real trailing daily-close history - see Multi-RC.py and its
    README for the full rationale (real data, no look-ahead)."""
    log_returns = np.log(calibration_price_df / calibration_price_df.shift(1)).dropna()
    vols = log_returns.std() * np.sqrt(252)
    corr = log_returns.corr()
    return vols.to_dict(), corr


def black_scholes_put(S, K, T, r, sigma, q=0.0):
    """Plain European put via QuantLib's AnalyticEuropeanEngine - used only
    in verify_against_closed_form, the deepest (N=1, trigger unreachable)
    reduction check."""
    if T <= 0:
        return {"price": max(K - S, 0.0)}

    today = ql.Date(1, 1, 2000)
    ql.Settings.instance().evaluationDate = today
    process = _quantlib_process(S, r, q, sigma, today)

    days = max(int(round(T * 365)), 1)
    exercise = ql.EuropeanExercise(today + ql.Period(days, ql.Days))
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return {"price": option.NPV()}


def worst_of_put_price(relative_spots, K, T, r, sigmas, qs, corr_matrix, n_paths=N_MC_PATHS, seed=MC_SEED):
    """Terminal-only worst-of put via QuantLib's basket-option machinery,
    same as Multi-RC's own pricer. Used only as a verification benchmark
    below, not as the live autocall pricer - see MATHEMATICS.md."""
    n = len(relative_spots)
    today = ql.Date(1, 1, 2000)
    ql.Settings.instance().evaluationDate = today

    processes = [_quantlib_process(relative_spots[i], r, qs[i], sigmas[i], today) for i in range(n)]

    corr = ql.Matrix(n, n)
    for i in range(n):
        for j in range(n):
            corr[i][j] = float(corr_matrix[i][j])

    process_array = ql.StochasticProcessArray(processes, corr)
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    basket_payoff = ql.MinBasketPayoff(payoff)

    days = max(int(round(T * 365)), 1)
    exercise = ql.EuropeanExercise(today + ql.Period(days, ql.Days))
    option = ql.BasketOption(basket_payoff, exercise)

    engine = ql.MCEuropeanBasketEngine(process_array, "PseudoRandom", timeStepsPerYear=1,
                                        requiredSamples=n_paths, seed=seed)
    option.setPricingEngine(engine)
    return option.NPV()


# REPLICATION: Multi-RC's worst-of ZCB-minus-put + a worst-of autocall
# feature. No closed form and no QuantLib basket-autocall engine exist for
# this, so the live pricer is a hand-rolled correlated multi-step Monte
# Carlo. See README.md for the dispersion discussion and MATHEMATICS.md for
# the full derivation.

def get_observation_dates(entry_date, tenor=TENOR, obs_per_year=OBS_PER_YEAR):
    """Quarterly (or otherwise) calendar dates from entry to maturity. The
    LAST one is maturity itself - not an autocall check (ordinary
    maturity settlement already returns full principal whenever the
    worst-of level is at or above the strike, trigger included)."""
    start = pd.Timestamp(entry_date)
    n_obs = int(round(tenor * obs_per_year))
    return [start + pd.Timedelta(days=round(tenor * 365.25 * i / n_obs)) for i in range(1, n_obs + 1)]


def map_to_trading_days(calendar_dates, path_index):
    trading_dates = []
    for cal_date in calendar_dates:
        pos = min(path_index.searchsorted(cal_date), len(path_index) - 1)
        trading_dates.append(path_index[pos])

    deduped = []
    for d in trading_dates:
        if not deduped or d != deduped[-1]:
            deduped.append(d)
    return deduped


def determine_actual_call_date(S0_list, price_df, observation_dates, trigger=TRIGGER):
    """First historical date the WORST-OF relative performance closes
    at/above the trigger (every name simultaneously), or None."""
    relative = price_df / pd.Series(S0_list, index=price_df.columns)
    worst_of = relative.min(axis=1)
    for obs_date in observation_dates[:-1]:
        if worst_of[obs_date] >= trigger:
            return obs_date
    return None


def simulate_forward_price(relative_spots_today, valuation_date, maturity_date, future_call_obs_dates,
                            sigmas, qs, corr_matrix, r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD,
                            strike=STRIKE, trigger=TRIGGER, n_paths=N_MC_PATHS, seed=MC_SEED):
    """Correlated multi-asset, multi-date Monte Carlo forward valuation
    (Cholesky-correlated GBM, worst-of-across-names at each observation
    date). See MATHEMATICS.md section 2 for the full construction."""
    n_assets = len(relative_spots_today)
    T_remaining = (maturity_date - valuation_date).days / 365.25
    if T_remaining <= 0:
        worst_of = min(relative_spots_today)
        put_payoff = max(strike - worst_of, 0.0)
        return {"price": 1.0 - (1.0 / strike) * put_payoff, "prob_called": 0.0}

    grid_dates = list(future_call_obs_dates) + [maturity_date]
    grid_years = np.array([(d - valuation_date).days / 365.25 for d in grid_dates])
    dt = np.diff(np.concatenate([[0.0], grid_years]))
    dt = np.clip(dt, 1e-8, None)
    n_steps = len(dt)

    L = np.linalg.cholesky(np.asarray(corr_matrix, dtype=float))
    rng = np.random.default_rng(seed)
    Z_indep = rng.standard_normal((n_paths, n_steps, n_assets))
    Z_corr = Z_indep @ L.T  # correlate ACROSS ASSETS within each time step; steps stay independent

    log_increments = np.empty((n_paths, n_steps, n_assets))
    for i in range(n_assets):
        drift_i = (r - qs[i] - 0.5 * sigmas[i] ** 2) * dt
        diffusion_i = sigmas[i] * np.sqrt(dt) * Z_corr[:, :, i]
        log_increments[:, :, i] = drift_i[None, :] + diffusion_i

    log_levels = np.log(relative_spots_today)[None, None, :] + np.cumsum(log_increments, axis=1)
    levels = np.exp(log_levels)  # shape (n_paths, n_steps, n_assets)
    worst_of_over_time = levels.min(axis=2)  # shape (n_paths, n_steps) - worst-of at each grid date

    n_call_obs = len(future_call_obs_dates)
    if n_call_obs > 0:
        triggered = worst_of_over_time[:, :n_call_obs] >= trigger
        any_triggered = triggered.any(axis=1)
        first_call_idx = np.argmax(triggered, axis=1)
    else:
        any_triggered = np.zeros(n_paths, dtype=bool)
        first_call_idx = np.zeros(n_paths, dtype=int)

    worst_of_T = worst_of_over_time[:, -1]
    call_time = grid_years[first_call_idx]
    settle_time = np.where(any_triggered, call_time, grid_years[-1])

    principal_pv = 1.0 * np.exp(-(r + credit_spread) * settle_time)

    put_quantity = 1.0 / strike
    put_payoff = np.where(any_triggered, 0.0, -put_quantity * np.maximum(strike - worst_of_T, 0.0))
    put_pv = put_payoff * np.exp(-r * grid_years[-1])

    price = float(np.mean(principal_pv + put_pv))
    return {"price": price, "prob_called": float(np.mean(any_triggered))}


def verify_against_closed_form(sigmas, qs, corr_matrix, T, r=RISK_FREE_RATE, n_paths=N_MC_PATHS):
    """Two nested reduction checks (N=1 and N=3, trigger unreachable)
    against independent closed-form/QuantLib benchmarks - see
    MATHEMATICS.md section 3."""
    maturity_date = pd.Timestamp(ENTRY_DATE) + pd.Timedelta(days=round(T * 365.25))

    single_relative = [1.0]
    mc_single = simulate_forward_price(single_relative, pd.Timestamp(ENTRY_DATE), maturity_date,
                                        future_call_obs_dates=[], sigmas=sigmas[:1], qs=qs[:1],
                                        corr_matrix=np.array([[1.0]]), r=r, trigger=10.0, n_paths=n_paths)["price"]
    zcb_single = zcb_price_and_greeks(1.0, T, r + GS_CDS_SPREAD)
    vanilla_single = black_scholes_put(1.0, STRIKE, T, r, sigmas[0], qs[0])["price"]
    closed_form_single = zcb_single["price"] - (1.0 / STRIKE) * vanilla_single
    n1_rel_diff = abs(mc_single - closed_form_single) / closed_form_single

    n = len(sigmas)
    all_relative = [1.0] * n
    mc_multi = simulate_forward_price(all_relative, pd.Timestamp(ENTRY_DATE), maturity_date,
                                       future_call_obs_dates=[], sigmas=sigmas, qs=qs,
                                       corr_matrix=corr_matrix, r=r, trigger=10.0, n_paths=n_paths)["price"]
    zcb_multi = zcb_price_and_greeks(1.0, T, r + GS_CDS_SPREAD)
    worst_of_put = worst_of_put_price(all_relative, STRIKE, T, r, sigmas, qs, corr_matrix,
                                       n_paths=200_000, seed=MC_SEED)
    closed_form_multi = zcb_multi["price"] - (1.0 / STRIKE) * worst_of_put
    n3_rel_diff = abs(mc_multi - closed_form_multi) / closed_form_multi

    return {
        "mc_single": mc_single, "closed_form_single": closed_form_single, "n1_rel_diff": n1_rel_diff,
        "mc_multi": mc_multi, "closed_form_multi": closed_form_multi, "n3_rel_diff": n3_rel_diff,
    }


def autocallable_running_return(S0_list, price_df, actual_call_date):
    """Payoff If Settled Today (Relative to Par), worst-of - see README
    "Reading the charts"."""
    relative = price_df / pd.Series(S0_list, index=price_df.columns)
    worst_of = relative.min(axis=1)
    running = pd.Series(0.0, index=price_df.index)
    below_strike = worst_of < STRIKE
    running[below_strike] = worst_of[below_strike] / STRIKE - 1.0

    if actual_call_date is not None:
        running[price_df.index >= actual_call_date] = 0.0

    return running, worst_of


def autocallable_mtm_price_series(S0_list, price_df, sigmas, qs, corr_matrix, observation_dates,
                                   actual_call_date, r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD,
                                   n_paths=N_MC_PATHS):
    """Model value of the LIVE note only - series ends at actual_call_date
    (NaN after) so the plotted line stops at settlement. See README
    "Reading the charts" and autocall_settlement_proceeds_series below."""
    maturity_date = price_df.index[-1]
    relative_path = price_df / pd.Series(S0_list, index=price_df.columns)

    prices = []
    for date, row in relative_path.iterrows():
        if actual_call_date is not None and date > actual_call_date:
            prices.append(np.nan)  # note has settled - nothing left to mark, line ends
            continue
        if actual_call_date is not None and date == actual_call_date:
            prices.append(1.0)  # settlement value itself (par, as modeled here - excludes coupons)
            continue

        future_obs = [d for d in observation_dates[:-1] if d > date]
        result = simulate_forward_price(row.values.tolist(), date, maturity_date, future_obs,
                                         sigmas, qs, corr_matrix, r, credit_spread, n_paths=n_paths)
        prices.append(result["price"])
    return pd.Series(prices, index=price_df.index)


def autocall_settlement_proceeds_series(price_df, actual_call_date):
    """Cash proceeds after an actual autocall, held flat (par=1.0, no
    reinvestment, excl. coupons) - see README "Reading the charts"."""
    proceeds = pd.Series(np.nan, index=price_df.index)
    if actual_call_date is not None:
        proceeds[price_df.index >= actual_call_date] = 1.0
    return proceeds


def finite_difference_greeks(relative_spots, valuation_date, maturity_date, future_call_obs_dates,
                              sigmas, qs, corr_matrix, r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD,
                              n_paths=N_MC_PATHS, seed=MC_SEED):
    """Bump-and-reprice Greeks - one Delta/Vega per name (risk is a
    vector), common random numbers (MC_SEED) so finite differences
    aren't swamped by independent MC noise."""
    n = len(relative_spots)
    bump_S, bump_sigma, bump_r = 0.01, 0.005, 0.0001

    def price_at(spots=relative_spots, sigs=sigmas, rate=r, val_date=valuation_date, obs_dates=future_call_obs_dates):
        return simulate_forward_price(spots, val_date, maturity_date, obs_dates, sigs, qs, corr_matrix,
                                       rate, credit_spread, n_paths=n_paths, seed=seed)["price"]

    price_mid = price_at()

    deltas = []
    for i in range(n):
        up = list(relative_spots); up[i] += bump_S
        down = list(relative_spots); down[i] -= bump_S
        deltas.append((price_at(spots=up) - price_at(spots=down)) / (2 * bump_S))

    vegas = []
    for i in range(n):
        sig_up = list(sigmas); sig_up[i] += bump_sigma
        sig_down = list(sigmas); sig_down[i] -= bump_sigma
        vegas.append((price_at(sigs=sig_up) - price_at(sigs=sig_down)) / (2 * bump_sigma))

    rho = (price_at(rate=r + bump_r) - price_at(rate=r - bump_r)) / (2 * bump_r)

    eps_T_days = min(1, max((maturity_date - valuation_date).days - 1, 0))
    if eps_T_days > 0:
        later_date = valuation_date + pd.Timedelta(days=eps_T_days)
        shifted_obs = [d for d in future_call_obs_dates if d > later_date]
        price_theta = price_at(val_date=later_date, obs_dates=shifted_obs)
        theta = (price_theta - price_mid) / (eps_T_days / 365.25)
    else:
        theta = 0.0

    return {"price": price_mid, "deltas": deltas, "vegas": vegas, "rho": rho, "theta": theta}


def plot_path(price_df, S0_list, tickers, underlying_names, observation_dates, actual_call_date,
              sigmas, qs, corr_matrix, greeks=None):
    index_return_pct = (price_df / pd.Series(S0_list, index=price_df.columns) - 1) * 100
    tracker, worst_of = autocallable_running_return(S0_list, price_df, actual_call_date)
    tracker_pct = tracker * 100

    _, ax = plt.subplots(figsize=(18, 8))
    asset_colors = ["steelblue", "darkorange", "seagreen", "mediumorchid", "peru", "slategray"]
    for i, ticker in enumerate(tickers):
        ax.plot(index_return_pct.index, index_return_pct[ticker].values, color=asset_colors[i % len(asset_colors)],
                linewidth=1.1, alpha=0.85, label=underlying_names[i])

    ax.plot(tracker_pct.index, tracker_pct.values, color="indianred", linewidth=1.8,
            linestyle="dashed", label="Multi-FCN (Autocallable) Payoff If Settled Today (Relative to Par) (worst-of)")

    ax.grid(True, which="major", color="lightgrey", linewidth=0.6)
    ax.axhline(0, color="lightgrey", linewidth=0.8)
    strike_pct = (STRIKE - 1) * 100
    ax.axhline(strike_pct, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.text(0.01, strike_pct, f"Strike: {strike_pct:.1f}%", transform=ax.get_yaxis_transform(),
            color="dodgerblue", fontsize=9, va="bottom", ha="left")

    trigger_pct = (TRIGGER - 1) * 100
    ax.axhline(trigger_pct, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.text(0.01, trigger_pct, f"Autocall Trigger: {trigger_pct:.1f}%", transform=ax.get_yaxis_transform(),
            color="dodgerblue", fontsize=9, va="bottom", ha="left")

    for obs_date in observation_dates[:-1]:
        ax.axvline(obs_date, color="mediumseagreen", linewidth=1, linestyle="dashed")

    if actual_call_date is not None:
        ax.axvline(actual_call_date, color="darkgreen", linewidth=1.2, linestyle="dashed")
        ax.text(actual_call_date, 0.98, "  Autocalled", transform=ax.get_xaxis_transform(),
                color="darkgreen", fontsize=9, va="top", ha="left")

    ax.set_xlabel("Date")
    ax.set_ylabel("Return from Entry (%)")
    lines, labels = ax.get_legend_handles_labels()
    ax.margins(x=0, y=0.05)

    if greeks is not None:
        delta_lines = "\n".join(f"  {t}: {d:.3f}" for t, d in zip(tickers, greeks["deltas"]))
        vega_lines = "\n".join(f"  {t}: {v:.4f}" for t, v in zip(tickers, greeks["vegas"]))
        greeks_text = (
            "Greeks at Inception:\n"
            f"Δ (Delta) per name:\n{delta_lines}\n"
            f"ν (Vega) per name (per 1% vol):\n{vega_lines}\n"
            f"ρ (Rho): {greeks['rho'] * 0.01:.4f} per 1% change in the 1Y Treasury rate\n"
            f"θ (Theta): {greeks['theta']:.4f} per year"
        )
        ax.text(1.16, 0.5, greeks_text, transform=ax.transAxes,
                fontsize=9, fontweight="light", color="black", va="center", ha="left")

    combined_min = min(index_return_pct.min().min(), tracker_pct.min(), strike_pct)
    combined_max = max(index_return_pct.max().max(), tracker_pct.max(), trigger_pct)

    mtm_price = autocallable_mtm_price_series(S0_list, price_df, sigmas, qs, corr_matrix,
                                               observation_dates, actual_call_date)
    mtm_gain_over_par_pct = (mtm_price - 1.0) * 100

    ax2 = ax.twinx()
    mtm_line, = ax2.plot(
        mtm_gain_over_par_pct.index, mtm_gain_over_par_pct.values, color="darkred", linewidth=1.8,
        linestyle="solid", label="Multi-FCN (Autocallable) — Model Value of Redemption Component (% of Par, excl. coupons, Monte Carlo)")
    ax2.set_ylabel("Redemption Component Value vs. Par (%)", color="darkred", rotation=270, labelpad=10)
    ax2.tick_params(axis="y", labelcolor="darkred")

    if actual_call_date is not None:
        proceeds = autocall_settlement_proceeds_series(price_df, actual_call_date)
        proceeds_gain_over_par_pct = (proceeds - 1.0) * 100
        proceeds_line, = ax2.plot(
            proceeds_gain_over_par_pct.index, proceeds_gain_over_par_pct.values, color="gray",
            linewidth=1.5, linestyle="dotted",
            label="Cash Proceeds After Autocall (par received, held flat, no reinvestment, excl. coupons)")
        lines.append(proceeds_line)
        labels.append(proceeds_line.get_label())

    combined_min = min(combined_min, mtm_gain_over_par_pct.min())
    combined_max = max(combined_max, mtm_gain_over_par_pct.max())
    pad = (combined_max - combined_min) * 0.05
    ax.set_ylim(combined_min - pad, combined_max + pad)
    ax2.set_ylim(combined_min - pad, combined_max + pad)

    lines.append(mtm_line)
    labels.append(mtm_line.get_label())

    basket_label = " / ".join(underlying_names)
    end_date = index_return_pct.index[-1]
    ax.set_title(f"Worst-of Basket ({basket_label}) Price Return Path from {price_df.index[0].date()} to {end_date.date()} \n"
                 f"vs Model Value of Redemption Component (Excludes Coupons) — Multi-FCN (Autocallable)")
    ax.legend(lines, labels, loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    underlying_names = [fetch_underlying_name(t) for t in TICKERS]
    print(f"Basket: {', '.join(f'{n} ({t})' for n, t in zip(underlying_names, TICKERS))}")

    entry_ts = pd.Timestamp(ENTRY_DATE)
    maturity_ts = entry_ts + pd.Timedelta(days=round(TENOR * 365.25))
    calibration_start = entry_ts - pd.Timedelta(days=round(CORRELATION_LOOKBACK_YEARS * 365.25))

    print(f"\nFetching {CORRELATION_LOOKBACK_YEARS}y TRAILING history (ending {ENTRY_DATE}) to estimate")
    print(f"vol and correlation from real data - no look-ahead into the backtest window...")
    calibration_df = fetch_multi_asset_path(TICKERS, calibration_start, entry_ts)
    vols_dict, corr_df = estimate_vols_and_correlation(calibration_df)
    sigmas = [vols_dict[t] for t in TICKERS]
    corr_matrix = corr_df.loc[TICKERS, TICKERS].values

    print(f"\nRealized annualized vol (from the {CORRELATION_LOOKBACK_YEARS}y trailing window):")
    for t, n, s in zip(TICKERS, underlying_names, sigmas):
        print(f"  {n} ({t}): {s:.2%}")
    print(f"\nRealized correlation matrix (from the same trailing window):")
    print(corr_df.loc[TICKERS, TICKERS].to_string(float_format=lambda x: f"{x:.3f}"))

    dividend_yields = [fetch_dividend_yield(t) for t in TICKERS]
    print(f"\nDividend yields (flat, continuous):")
    for t, n, q in zip(TICKERS, underlying_names, dividend_yields):
        print(f"  {n} ({t}): {q:.2%}")

    print(f"\nFetching {ENTRY_DATE} (entry) to +{TENOR}y (maturity) backtest path...")
    price_df = fetch_multi_asset_path(TICKERS, entry_ts, maturity_ts)
    S0_list = [float(price_df[t].iloc[0]) for t in TICKERS]
    S_T_list = [float(price_df[t].iloc[-1]) for t in TICKERS]

    observation_calendar_dates = get_observation_dates(ENTRY_DATE, TENOR, OBS_PER_YEAR)
    observation_dates = map_to_trading_days(observation_calendar_dates, price_df.index)
    actual_call_date = determine_actual_call_date(S0_list, price_df, observation_dates)

    print(f"\nAutocall schedule (quarterly, trigger={TRIGGER:.0%} - ALL names must be at/above it):")
    relative_at_obs = price_df / pd.Series(S0_list, index=price_df.columns)
    for i, obs_date in enumerate(observation_dates, start=1):
        worst = relative_at_obs.loc[obs_date].min() if obs_date in price_df.index else None
        tag = " (maturity, not an autocall check)" if i == len(observation_dates) else ""
        print(f"  Obs {i}: {obs_date.date()}  -  worst-of relative performance {worst:.2%}{tag}")

    if actual_call_date is not None:
        print(f"\n>>> Note ACTUALLY AUTOCALLED on {actual_call_date.date()} in this historical path (every")
        print(f"name closed at or above the {TRIGGER:.0%} trigger). Principal (as modeled here) is returned")
        print(f"early; no further downside exposure after this date. A real note would also pay accrued")
        print(f"coupons to that date - this script does not price or include coupons. <<<")
    else:
        print(f"\n>>> Note never autocalled in this historical path; ran to full maturity like Multi-RC. <<<")

    print(f"\nVerifying the hand-rolled correlated multi-step engine (autocall trigger set unreachable):")
    check = verify_against_closed_form(sigmas, dividend_yields, corr_matrix, TENOR)
    print(f"  N=1 reduction: MC={check['mc_single']:.5f}  closed-form single-name={check['closed_form_single']:.5f}"
          f"  rel diff={check['n1_rel_diff']:.3%}  "
          f"({'PASS' if check['n1_rel_diff'] < 0.01 else 'FAIL - investigate before trusting results'})")
    print(f"  N=3 reduction: MC={check['mc_multi']:.5f}  Multi-RC's own worst-of engine={check['closed_form_multi']:.5f}"
          f"  rel diff={check['n3_rel_diff']:.3%}  "
          f"({'PASS' if check['n3_rel_diff'] < 0.02 else 'FAIL - investigate before trusting results'})")

    tracker, worst_of = autocallable_running_return(S0_list, price_df, actual_call_date)
    note_return = tracker.iloc[-1]

    summary_rows = {
        "Entry Date": price_df.index[0].date(),
        "Maturity Date": price_df.index[-1].date(),
        "Strike": f"{STRIKE:.0%}", "Trigger": f"{TRIGGER:.0%}",
        "Actual Autocall Date": actual_call_date.date() if actual_call_date is not None else "Never (ran to maturity)",
    }
    for t, n, s0, st in zip(TICKERS, underlying_names, S0_list, S_T_list):
        summary_rows[f"{n} ({t}) Return"] = f"{st / s0 - 1:+.2%}"
    summary_rows["Multi-FCN (Autocallable) — Redemption Payoff (vs. Par, excl. coupons)"] = f"{note_return:.2%}"

    print("\n" + pd.Series(summary_rows).to_string())

    future_obs_at_entry = observation_dates[:-1]
    greeks = finite_difference_greeks([1.0] * len(TICKERS), entry_ts, maturity_ts, future_obs_at_entry,
                                       sigmas, dividend_yields, corr_matrix)
    fair_value_pct_of_par = greeks["price"]

    mc_entry = simulate_forward_price([1.0] * len(TICKERS), entry_ts, maturity_ts, future_obs_at_entry,
                                       sigmas, dividend_yields, corr_matrix)

    print(f"\nMulti-FCN (Autocallable) — model value of redemption component at inception (excludes coupons)")
    print(f"(Monte Carlo, {N_MC_PATHS:,} paths, quarterly worst-of autocall obs at {TRIGGER:.0%} trigger,")
    print(f"ZCB discounted at the 1Y Treasury rate {RISK_FREE_RATE:.2%} + GS CDS {GS_CDS_SPREAD:.2%}, worst-of put at the 1Y Treasury rate")
    print(f"alone, T={TENOR}y, strike={STRIKE:.0%}):")
    print(f"  {fair_value_pct_of_par:.2%} of par ({fair_value_pct_of_par - 1:+.2%} vs. par)")
    print(f"  P(called before maturity): {mc_entry['prob_called']:.2%}")

    for t, n, d, v in zip(TICKERS, underlying_names, greeks["deltas"], greeks["vegas"]):
        print(f"  Delta ({n}): {d:.3f}   Vega ({n}, per 1% vol): {v * 0.01:.4f}")
    print(f"  Rho: {greeks['rho'] * 0.01:.4f}  (per 1% change in the 1Y Treasury rate)")
    print(f"  Theta: {greeks['theta']:.4f} per year / {greeks['theta'] / 365:.5f} per day")

    no_autocall_price = simulate_forward_price([1.0] * len(TICKERS), entry_ts, maturity_ts, [],
                                                sigmas, dividend_yields, corr_matrix, trigger=10.0)["price"]
    print(f"\nFor comparison, the SAME worst-of basket with NO autocall feature (Multi-RC, in effect)")
    print(f"would be worth: {no_autocall_price:.2%} of par. The autocallable version above should be")
    print(f"HIGHER: early redemption at par is optionality that can only help the noteholder relative")
    print(f"to waiting out the full term - a real, computed consequence of adding the feature, not an")
    print(f"assumed one.")

    plot_path(price_df, S0_list, TICKERS, underlying_names, observation_dates, actual_call_date,
              sigmas, dividend_yields, corr_matrix, greeks=greeks)
