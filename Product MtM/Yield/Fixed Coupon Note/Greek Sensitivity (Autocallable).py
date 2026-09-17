import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

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
    interpolate_implied_vol,
    fetch_trailing_realized_vol,
    fetch_risk_free_rate,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Same product terms as "Fixed Coupon Note (Autocallable).py" in this folder ---
STRIKE = 0.90
GS_CDS_SPREAD = 0.002675       # Goldman Sachs 1y CDS, 26.75 bps (Investing.com) - tenor-matched to TENOR=1
TRIGGER = 1.00
OBS_PER_YEAR = 4

ENTRY_DATE = "2025-01-01"
TENOR = 1
RISK_FREE_RATE = fetch_risk_free_rate(ENTRY_DATE)  # 1Y Treasury CMT (FRED DGS1) as of ENTRY_DATE - real, historical; option leg pricing and MC risk-neutral drift

TICKER = "MCD"
SPX_FRED_SERIES = "SP500"    # FRED fallback if yfinance fails - valid ONLY when TICKER
                             # is literally "^GSPC"; never used as a stand-in for a
                             # single-name stock's own price

SPX_VOL_TERM_STRUCTURE_TICKERS = {"^VIX": 30, "^VIX3M": 93, "^VIX6M": 182}  # SPX-only proxy;
                                                                            # only used when TICKER is an index (see below)
VIX_FRED_SERIES = "VIXCLS"

N_MC_PATHS = 50000
MC_SEED = 42

# T (TENOR), strike, trigger and the funding curve are fixed constants
# throughout this whole analysis - the only thing that varies below is
# spot. Vol is also held at its inception value for every row.
SPOT_SCENARIO_RANGE = np.arange(0.60, 1.41, 0.05)  # 60% to 140% of S0, in 5pt steps


def fetch_index_path(entry_date, years=TENOR):
    start = pd.Timestamp(entry_date)
    end = start + pd.Timedelta(days=round(years * 365.25))
    return fetch_daily_closes(TICKER, start, end, fred_series=SPX_FRED_SERIES if TICKER == "^GSPC" else None)


def fetch_spx_vol_term_structure(entry_date, years=TENOR):
    start = pd.Timestamp(entry_date)
    end = start + pd.Timedelta(days=round(years * 365.25))

    term_structure = {}
    for ticker, tenor_days in SPX_VOL_TERM_STRUCTURE_TICKERS.items():
        fred_series = VIX_FRED_SERIES if ticker == "^VIX" else None
        try:
            series = fetch_daily_closes(ticker, start, end, fred_series=fred_series)
            term_structure[tenor_days / 365.25] = series / 100.0
        except Exception as exc:
            print(f"  Could not fetch {ticker} ({exc}); dropping it from the vol term structure")

    if not term_structure:
        raise RuntimeError("Could not fetch any SPX implied vol data (VIX/VIX3M/VIX6M)")

    return term_structure


def autocall_digital_strip_price(S0, obs_times, r, sigma, trigger, Q, q=0.0):
    """Closed-form autocall value as a strip of digital calls, weighted by
    each date's EXACT joint first-passage probability. Full derivation in
    ../MATHEMATICS.md section 5.3."""
    obs_times = np.asarray(obs_times, dtype=float)
    n = len(obs_times)
    if n == 0:
        return {"total_price": 0.0, "total_prob_called": 0.0, "per_date": []}

    K = trigger * S0
    d2 = (np.log(S0 / K) + (r - q - 0.5 * sigma ** 2) * obs_times) / (sigma * np.sqrt(obs_times))
    k = -d2

    # corr(X_i, X_j) for the standardized Brownian value at each observation date
    R = np.sqrt(np.minimum.outer(obs_times, obs_times) / np.maximum.outer(obs_times, obs_times))

    not_called_before = 1.0  # P(not called before the 1st observation) is trivially 1
    not_called_upto = 1.0
    per_date = []
    total_price = 0.0
    for i in range(n):
        cov_upto = R[:i + 1, :i + 1]
        not_called_upto = multivariate_normal(mean=np.zeros(i + 1), cov=cov_upto).cdf(k[:i + 1])

        prob_exact = not_called_before - not_called_upto
        price_i = Q * np.exp(-r * obs_times[i]) * prob_exact
        total_price += price_i
        per_date.append({"T": obs_times[i], "prob_exact": prob_exact, "price": price_i})

        not_called_before = not_called_upto

    return {"total_price": total_price, "total_prob_called": 1.0 - not_called_upto, "per_date": per_date}


def get_observation_dates(entry_date, tenor=TENOR, obs_per_year=OBS_PER_YEAR):
    start = pd.Timestamp(entry_date)
    n_obs = int(round(tenor * obs_per_year))
    return [start + pd.Timedelta(days=round(tenor * 365.25 * i / n_obs)) for i in range(1, n_obs + 1)]


def simulate_forward_price(S_t, S0, valuation_date, maturity_date, future_call_obs_dates,
                            sigma, r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD,
                            strike=STRIKE, trigger=TRIGGER, n_paths=N_MC_PATHS, seed=MC_SEED, q=0.0):
    T_remaining = (maturity_date - valuation_date).days / 365.25
    if T_remaining <= 0:
        put_quantity = 1.0 / strike
        put_payoff = -put_quantity * max(strike * S0 - S_t, 0.0)
        return {"price": S0 + put_payoff, "prob_called": 0.0}

    grid_dates = list(future_call_obs_dates) + [maturity_date]
    grid_years = np.array([(d - valuation_date).days / 365.25 for d in grid_dates])
    dt = np.diff(np.concatenate([[0.0], grid_years]))
    dt = np.clip(dt, 1e-8, None)

    rng = np.random.default_rng(seed)
    n_steps = len(dt)
    z = rng.standard_normal((n_paths, n_steps))

    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    log_levels = np.log(S_t) + np.cumsum(drift + diffusion, axis=1)
    levels = np.exp(log_levels)

    n_call_obs = len(future_call_obs_dates)
    if n_call_obs > 0:
        triggered = levels[:, :n_call_obs] >= trigger * S0
        any_triggered = triggered.any(axis=1)
        first_call_idx = np.argmax(triggered, axis=1)
    else:
        any_triggered = np.zeros(n_paths, dtype=bool)
        first_call_idx = np.zeros(n_paths, dtype=int)

    S_T = levels[:, -1]
    call_time = grid_years[first_call_idx]
    settle_time = np.where(any_triggered, call_time, grid_years[-1])

    principal_pv = S0 * np.exp(-(r + credit_spread) * settle_time)

    put_quantity = 1.0 / strike
    put_payoff = np.where(any_triggered, 0.0, -put_quantity * np.maximum(strike * S0 - S_T, 0.0))
    put_pv = put_payoff * np.exp(-r * grid_years[-1])

    price = float(np.mean(principal_pv + put_pv))
    return {"price": price, "prob_called": float(np.mean(any_triggered))}


# GREEKS LADDER: T, strike, trigger, funding curve and vol pinned at entry;
# only spot varies. Bump-and-reprice on the MC engine with common random
# numbers (MC_SEED). Theta excluded (T never moves). See README.

def finite_difference_greeks(S_t, S0, valuation_date, maturity_date, future_call_obs_dates, sigma, r, credit_spread, q=0.0):
    eps_S = 0.005 * S0
    eps_sigma = 0.001
    eps_r = 0.0001

    def price_at(S=S_t, sig=sigma, rate=r):
        return simulate_forward_price(S, S0, valuation_date, maturity_date, future_call_obs_dates,
                                       sig, rate, credit_spread, q=q)["price"]

    price_mid = price_at()
    delta = (price_at(S=S_t + eps_S) - price_at(S=S_t - eps_S)) / (2 * eps_S)
    vega = (price_at(sig=sigma + eps_sigma) - price_at(sig=sigma - eps_sigma)) / (2 * eps_sigma)
    rho = (price_at(rate=r + eps_r) - price_at(rate=r - eps_r)) / (2 * eps_r)

    return {"price": price_mid, "delta": delta, "vega": vega, "rho": rho}


def greek_sensitivity_table(S0, entry_date, maturity_date, future_call_obs_dates, sigma,
                             r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD, spot_multiples=SPOT_SCENARIO_RANGE, q=0.0):
    rows = []
    for multiple in spot_multiples:
        S = multiple * S0
        g = finite_difference_greeks(S, S0, entry_date, maturity_date, future_call_obs_dates, sigma, r, credit_spread, q=q)
        rows.append({
            "Spot (% of S0)": multiple * 100,
            "Price (% of Par)": g["price"] / S0 * 100,
            "Delta": g["delta"],
            "Vega (per 1% vol)": g["vega"] / S0 * 0.01,
            "Rho (per 1% rate)": g["rho"] / S0 * 0.01,
        })
    return pd.DataFrame(rows)


def plot_greek_sensitivity(table, S0, T, sigma, r, underlying_name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    spot = table["Spot (% of S0)"]

    panels = [
        ("Price (% of Par)", "Model Value of Redemption Component (% of Par)", "firebrick"),
        ("Delta", "Delta", "darkred"),
        ("Vega (per 1% vol)", "Vega (per 1% change in vol)", "indianred"),
        ("Rho (per 1% rate)", "Rho (per 1% change in the 1Y Treasury rate)", "brown"),
    ]

    for ax, (col, ylabel, color) in zip(axes.flat, panels):
        ax.plot(spot, table[col], color=color, linewidth=1.8, marker="o", markersize=3)
        ax.axvline(100, color="lightgrey", linewidth=0.8, linestyle="dashed")
        ax.axvline(STRIKE * 100, color="dodgerblue", linewidth=0.8, linestyle="dotted")
        ax.axvline(TRIGGER * 100, color="dodgerblue", linewidth=0.8, linestyle="dotted")
        ax.axhline(0, color="lightgrey", linewidth=0.6)
        ax.grid(True, color="lightgrey", linewidth=0.4)
        ax.set_xlabel("Spot (% of S0)")
        ax.set_ylabel(ylabel)
        ax.set_title(col)

    fig.suptitle(
        f"{underlying_name} Autocallable Fixed Coupon Note - Greeks Ladder (T, strike, trigger fixed, spot varies)\n"
        f"Fixed throughout: T={T}y, vol={sigma:.2%}, 1Y Treasury={r:.2%}, CDS={GS_CDS_SPREAD:.2%}, "
        f"strike={STRIKE:.0%}, trigger={TRIGGER:.0%} of S0={S0:,.2f} (Monte Carlo, {N_MC_PATHS:,} paths)"
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    print("Greeks Ladder - Autocallable Fixed Coupon Note")
    print("=" * 60)
    print("This is NOT a new backtest and does not use the historical path.")
    print("It reuses the same entry conditions as the main backtest script,")
    print("then holds T, strike, trigger, funding curve and vol all fixed and")
    print("varies ONLY spot, so each Greek's value at a given spot level tells")
    print("you directly how much MTM moves for a 1-unit change in that")
    print("variable at that spot. Greeks are Monte Carlo finite differences")
    print("(common random numbers), not closed-form, since the autocall")
    print("feature has no simple algebraic formula. Theta is excluded since")
    print("T never varies in this analysis.\n")

    underlying_name = fetch_underlying_name(TICKER)
    print(f"Fetching {underlying_name} entry level for {ENTRY_DATE}...")
    path = fetch_index_path(ENTRY_DATE)
    S0 = float(path.iloc[0])
    entry_date = path.index[0]
    maturity_date = entry_date + pd.Timedelta(days=round(TENOR * 365.25))

    dividend_yield = fetch_dividend_yield(TICKER)

    observation_dates = get_observation_dates(ENTRY_DATE, TENOR, OBS_PER_YEAR)
    future_call_obs_dates = observation_dates[:-1]  # exclude maturity - see main script for rationale

    if TICKER.startswith("^"):
        print(f"Fetching SPX implied vol term structure for {ENTRY_DATE}...")
        raw_term_structure = fetch_spx_vol_term_structure(ENTRY_DATE)
        vols_at_entry = {tenor: series.iloc[0] for tenor, series in raw_term_structure.items()}
        entry_vol = interpolate_implied_vol(vols_at_entry, TENOR)
        vol_source_desc = f"the VIX/VIX3M/VIX6M term structure on {ENTRY_DATE}"
    else:
        print(f"{TICKER} is a single name - using its own trailing 2y realized volatility instead of")
        print(f"the SPX VIX/VIX3M/VIX6M proxy:")
        entry_vol = fetch_trailing_realized_vol(TICKER, ENTRY_DATE)
        vol_source_desc = f"{TICKER}'s own trailing 2y realized vol (not VIX-derived)"
        print(f"  {entry_vol:.2%}")

    print(f"\nFixed throughout (only spot varies below):")
    print(f"  Underlying:      {underlying_name} ({TICKER})")
    print(f"  Entry Date:      {ENTRY_DATE}")
    print(f"  S0 (spot):       {S0:,.2f}")
    print(f"  Strike:          {STRIKE:.0%} of S0")
    print(f"  Autocall trigger:{TRIGGER:.0%} of S0, checked quarterly ({len(future_call_obs_dates)} obs dates)")
    print(f"  Tenor (T):       {TENOR} year(s)")
    print(f"  Vol (sigma):     {entry_vol:.2%}  (from {vol_source_desc})")
    print(f"  1Y Treasury (FRED DGS1): {RISK_FREE_RATE:.2%}")
    print(f"  GS CDS spread:   {GS_CDS_SPREAD:.2%}")
    print(f"  Dividend yield:  {dividend_yield:.2%}  (flat, continuous - 0% if an index)")

    obs_years_at_entry = [(d - entry_date).days / 365.25 for d in future_call_obs_dates]
    mc_entry = simulate_forward_price(S0, S0, entry_date, maturity_date, future_call_obs_dates, entry_vol, q=dividend_yield)
    strip = autocall_digital_strip_price(S0, obs_years_at_entry, RISK_FREE_RATE, entry_vol, TRIGGER,
                                          Q=TRIGGER * S0, q=dividend_yield)

    print(f"\nClosed-form check on the autocall feature (short digital calls, trigger={TRIGGER:.0%}):")
    print(f"  Each date's digital call c=Q*e^(-rT)*N(d2) is weighted by the EXACT first-passage")
    print(f"  probability of being called then (not earlier) - same joint-normal correlation")
    print(f"  structure as the main backtest script and Autocall Density.")
    for obs_date, leg in zip(future_call_obs_dates, strip["per_date"]):
        print(f"  Obs {obs_date.date()} (T={leg['T']:.2f}y): "
              f"P(called exactly here)={leg['prob_exact']:.2%}, digital call PV={leg['price']:,.2f}")
    print(f"  Closed-form autocall strip PV: {strip['total_price']:,.2f}")
    print(f"  Closed-form P(called before maturity): {strip['total_prob_called']:.2%}")
    print(f"  Monte Carlo P(called before maturity):  {mc_entry['prob_called']:.2%}")

    sensitivity = greek_sensitivity_table(S0, entry_date, maturity_date, future_call_obs_dates, entry_vol, q=dividend_yield)
    print(f"\nGreeks Ladder (Monte Carlo finite differences, common random numbers):")
    print(sensitivity.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    plot_greek_sensitivity(sensitivity, S0, TENOR, entry_vol, RISK_FREE_RATE, underlying_name)
