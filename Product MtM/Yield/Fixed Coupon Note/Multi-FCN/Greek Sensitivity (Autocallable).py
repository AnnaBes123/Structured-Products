import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    fetch_risk_free_rate,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Same product terms as "Multi-FCN (Autocallable).py" in this folder ---
STRIKE = 0.90
TRIGGER = 1.00
OBS_PER_YEAR = 4
GS_CDS_SPREAD = 0.002675       # Goldman Sachs 1y CDS, 26.75 bps (Investing.com) - tenor-matched to TENOR=1

ENTRY_DATE = "2025-01-02"
TENOR = 1
RISK_FREE_RATE = fetch_risk_free_rate(ENTRY_DATE)  # 1Y Treasury CMT (FRED DGS1) as of ENTRY_DATE - real, historical; option leg pricing and MC risk-neutral drift

TICKERS = ["AAPL", "JPM", "XOM"]
CORRELATION_LOOKBACK_YEARS = 2

N_MC_PATHS = 50000
MC_SEED = 42

# T, strike, trigger, the funding curve, and each name's (trailing-window)
# vol/correlation/dividend yield are fixed throughout - the only thing that
# varies below is a COMMON multiplicative shock applied to every name at
# once, same convention as Multi-RC's own Greek Sensitivity.py.
SPOT_SCENARIO_RANGE = np.arange(0.60, 1.41, 0.05)  # 60% to 140% of S0, in 5pt steps


def fetch_multi_asset_path(tickers, start, end):
    series = {ticker: fetch_daily_closes(ticker, start, end) for ticker in tickers}
    df = pd.concat(series, axis=1)
    return df.dropna(how="any")


def estimate_vols_and_correlation(calibration_price_df):
    log_returns = np.log(calibration_price_df / calibration_price_df.shift(1)).dropna()
    vols = log_returns.std() * np.sqrt(252)
    corr = log_returns.corr()
    return vols.to_dict(), corr


def get_observation_dates(entry_date, tenor=TENOR, obs_per_year=OBS_PER_YEAR):
    start = pd.Timestamp(entry_date)
    n_obs = int(round(tenor * obs_per_year))
    return [start + pd.Timedelta(days=round(tenor * 365.25 * i / n_obs)) for i in range(1, n_obs + 1)]


def simulate_forward_price(relative_spots_today, valuation_date, maturity_date, future_call_obs_dates,
                            sigmas, qs, corr_matrix, r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD,
                            strike=STRIKE, trigger=TRIGGER, n_paths=N_MC_PATHS, seed=MC_SEED):
    """Hand-rolled correlated multi-step Monte Carlo - see "Multi-FCN
    (Autocallable).py" in this folder for the full rationale (no QuantLib
    basket-autocallable engine exists, so this generalizes the single-name
    Autocallable FCN's own hand-rolled MC engine with an added, Cholesky-
    correlated asset axis and a worst-of reduction at each observation
    date)."""
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
    Z_corr = Z_indep @ L.T

    log_increments = np.empty((n_paths, n_steps, n_assets))
    for i in range(n_assets):
        drift_i = (r - qs[i] - 0.5 * sigmas[i] ** 2) * dt
        diffusion_i = sigmas[i] * np.sqrt(dt) * Z_corr[:, :, i]
        log_increments[:, :, i] = drift_i[None, :] + diffusion_i

    log_levels = np.log(relative_spots_today)[None, None, :] + np.cumsum(log_increments, axis=1)
    levels = np.exp(log_levels)
    worst_of_over_time = levels.min(axis=2)

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


# GREEKS LADDER: x-axis is a COMMON multiplicative shock applied to every
# name at once (day-1 observation dates); T, strike, trigger, funding curve
# and each name's vol/correlation/dividend yield are held fixed. Delta/Vega
# per name, Rho a single shared line, Theta excluded. See README.

def greek_sensitivity_table(valuation_date, maturity_date, future_call_obs_dates, sigmas, qs, corr_matrix,
                             tickers, r=RISK_FREE_RATE, spot_multiples=SPOT_SCENARIO_RANGE,
                             n_paths=N_MC_PATHS, seed=MC_SEED):
    bump_S, bump_sigma, bump_r = 0.01, 0.005, 0.0001
    n = len(tickers)
    rows = []
    for multiple in spot_multiples:
        relative_spots = [multiple] * n
        base = simulate_forward_price(relative_spots, valuation_date, maturity_date, future_call_obs_dates,
                                       sigmas, qs, corr_matrix, r, n_paths=n_paths, seed=seed)["price"]

        deltas = []
        for i in range(n):
            up = list(relative_spots); up[i] += bump_S
            down = list(relative_spots); down[i] -= bump_S
            price_up = simulate_forward_price(up, valuation_date, maturity_date, future_call_obs_dates,
                                               sigmas, qs, corr_matrix, r, n_paths=n_paths, seed=seed)["price"]
            price_down = simulate_forward_price(down, valuation_date, maturity_date, future_call_obs_dates,
                                                 sigmas, qs, corr_matrix, r, n_paths=n_paths, seed=seed)["price"]
            deltas.append((price_up - price_down) / (2 * bump_S))

        vegas = []
        for i in range(n):
            sig_up = list(sigmas); sig_up[i] += bump_sigma
            sig_down = list(sigmas); sig_down[i] -= bump_sigma
            price_up = simulate_forward_price(relative_spots, valuation_date, maturity_date, future_call_obs_dates,
                                               sig_up, qs, corr_matrix, r, n_paths=n_paths, seed=seed)["price"]
            price_down = simulate_forward_price(relative_spots, valuation_date, maturity_date, future_call_obs_dates,
                                                 sig_down, qs, corr_matrix, r, n_paths=n_paths, seed=seed)["price"]
            vegas.append((price_up - price_down) / (2 * bump_sigma))

        price_up_r = simulate_forward_price(relative_spots, valuation_date, maturity_date, future_call_obs_dates,
                                             sigmas, qs, corr_matrix, r + bump_r, n_paths=n_paths, seed=seed)["price"]
        price_down_r = simulate_forward_price(relative_spots, valuation_date, maturity_date, future_call_obs_dates,
                                               sigmas, qs, corr_matrix, r - bump_r, n_paths=n_paths, seed=seed)["price"]
        rho = (price_up_r - price_down_r) / (2 * bump_r)

        row = {"Spot (% of S0, all names)": multiple * 100, "Price (% of Par)": base * 100, "Rho (per 1% rate)": rho * 0.01}
        for i, ticker in enumerate(tickers):
            row[f"Delta ({ticker})"] = deltas[i]
            row[f"Vega ({ticker}, per 1% vol)"] = vegas[i] * 0.01
        rows.append(row)
    return pd.DataFrame(rows)


def plot_greek_sensitivity(table, tickers, T, r, underlying_names):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    spot = table["Spot (% of S0, all names)"]
    asset_colors = ["steelblue", "darkorange", "seagreen", "mediumorchid", "peru", "slategray"]

    ax = axes[0, 0]
    ax.plot(spot, table["Price (% of Par)"], color="firebrick", linewidth=1.8, marker="o", markersize=3)
    ax.axvline(100, color="lightgrey", linewidth=0.8, linestyle="dashed")
    ax.axvline(STRIKE * 100, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.axvline(TRIGGER * 100, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.set_title("Price (% of Par)")
    ax.set_xlabel("Spot (% of S0, all names shocked together)")
    ax.set_ylabel("Model Value of Redemption Component (% of Par)")
    ax.grid(True, color="lightgrey", linewidth=0.4)

    ax = axes[0, 1]
    for i, (ticker, name) in enumerate(zip(tickers, underlying_names)):
        ax.plot(spot, table[f"Delta ({ticker})"], color=asset_colors[i % len(asset_colors)], linewidth=1.8,
                marker="o", markersize=3, label=name)
    ax.axvline(100, color="lightgrey", linewidth=0.8, linestyle="dashed")
    ax.axvline(STRIKE * 100, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.axvline(TRIGGER * 100, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.axhline(0, color="lightgrey", linewidth=0.6)
    ax.set_title("Delta (per name)")
    ax.set_xlabel("Spot (% of S0, all names shocked together)")
    ax.set_ylabel("Delta")
    ax.legend(fontsize=8)
    ax.grid(True, color="lightgrey", linewidth=0.4)

    ax = axes[1, 0]
    for i, (ticker, name) in enumerate(zip(tickers, underlying_names)):
        ax.plot(spot, table[f"Vega ({ticker}, per 1% vol)"], color=asset_colors[i % len(asset_colors)], linewidth=1.8,
                marker="o", markersize=3, label=name)
    ax.axvline(100, color="lightgrey", linewidth=0.8, linestyle="dashed")
    ax.axvline(STRIKE * 100, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.axvline(TRIGGER * 100, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.axhline(0, color="lightgrey", linewidth=0.6)
    ax.set_title("Vega (per name, per 1% change in that name's vol)")
    ax.set_xlabel("Spot (% of S0, all names shocked together)")
    ax.set_ylabel("Vega")
    ax.legend(fontsize=8)
    ax.grid(True, color="lightgrey", linewidth=0.4)

    ax = axes[1, 1]
    ax.plot(spot, table["Rho (per 1% rate)"], color="brown", linewidth=1.8, marker="o", markersize=3)
    ax.axvline(100, color="lightgrey", linewidth=0.8, linestyle="dashed")
    ax.axvline(STRIKE * 100, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.axvline(TRIGGER * 100, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.axhline(0, color="lightgrey", linewidth=0.6)
    ax.set_title("Rho (per 1% change in the 1Y Treasury rate)")
    ax.set_xlabel("Spot (% of S0, all names shocked together)")
    ax.set_ylabel("Rho")
    ax.grid(True, color="lightgrey", linewidth=0.4)

    basket_label = " / ".join(underlying_names)
    fig.suptitle(
        f"Worst-of Basket ({basket_label}) Multi-FCN (Autocallable) - Greeks Ladder\n"
        f"(T={T}y, strike={STRIKE:.0%}, trigger={TRIGGER:.0%}, r={r:.2%} fixed throughout; x-axis is a "
        f"COMMON shock applied to every name in the basket at once)"
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    print("Greeks Ladder - Multi-FCN (Autocallable)")
    print("=" * 60)
    print("This is NOT a new backtest and does not use the historical path.")
    print("It reuses the same entry conditions (and the same trailing-window")
    print("vol/correlation estimate) as the main backtest script, then holds")
    print("T, strike, trigger, funding curve, vol and correlation fixed and")
    print("varies ONLY a COMMON shock applied to every name in the basket at")
    print("once. Delta and Vega are reported per name. Greeks are Monte Carlo")
    print("finite differences (common random numbers), not closed-form, since")
    print("the autocall feature has no simple algebraic formula for a basket.\n")

    underlying_names = [fetch_underlying_name(t) for t in TICKERS]
    print(f"Basket: {', '.join(f'{n} ({t})' for n, t in zip(underlying_names, TICKERS))}")

    entry_ts = pd.Timestamp(ENTRY_DATE)
    maturity_ts = entry_ts + pd.Timedelta(days=round(TENOR * 365.25))
    calibration_start = entry_ts - pd.Timedelta(days=round(CORRELATION_LOOKBACK_YEARS * 365.25))

    print(f"\nFetching {CORRELATION_LOOKBACK_YEARS}y trailing history (ending {ENTRY_DATE}) for vol/correlation...")
    calibration_df = fetch_multi_asset_path(TICKERS, calibration_start, entry_ts)
    vols_dict, corr_df = estimate_vols_and_correlation(calibration_df)
    sigmas = [vols_dict[t] for t in TICKERS]
    corr_matrix = corr_df.loc[TICKERS, TICKERS].values

    dividend_yields = [fetch_dividend_yield(t) for t in TICKERS]

    observation_dates = get_observation_dates(ENTRY_DATE, TENOR, OBS_PER_YEAR)
    future_call_obs_dates = observation_dates[:-1]

    print(f"\nFixed throughout (only the common spot shock varies below):")
    for t, n, s, q in zip(TICKERS, underlying_names, sigmas, dividend_yields):
        print(f"  {n} ({t}): vol={s:.2%}, dividend yield={q:.2%}")
    print(f"  Strike:            {STRIKE:.0%}")
    print(f"  Autocall trigger:  {TRIGGER:.0%}, checked quarterly ({len(future_call_obs_dates)} obs dates)")
    print(f"  Tenor (T):         {TENOR} year(s)")
    print(f"  1Y Treasury (FRED DGS1): {RISK_FREE_RATE:.2%}")

    sensitivity = greek_sensitivity_table(entry_ts, maturity_ts, future_call_obs_dates, sigmas, dividend_yields,
                                           corr_matrix, TICKERS)
    print(f"\nGreeks Ladder (Monte Carlo finite differences, common random numbers):")
    print(sensitivity.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    plot_greek_sensitivity(sensitivity, TICKERS, TENOR, RISK_FREE_RATE, underlying_names)
