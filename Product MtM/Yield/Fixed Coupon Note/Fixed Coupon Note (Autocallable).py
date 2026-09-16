import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql
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
    realized_annualized_vol,
    max_drawdown,
    _quantlib_process,
    zcb_price_and_greeks,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Product terms (same base terms as the plain "Fixed Coupon Note.py",
# plus the autocall feature) ---
STRIKE = 0.9                 # short put strike, as a fraction of S0
RISK_FREE_RATE = 0.04         # SOFR proxy - option leg pricing and MC risk-neutral drift
GS_CDS_SPREAD = 0.005308      # Goldman Sachs 5y CDS - issuer credit spread, principal leg only

TRIGGER = 1.00                # autocall level, as a fraction of S0 (100%)
OBS_PER_YEAR = 4              # quarterly observation dates

ENTRY_DATE = "2025-01-01"
TENOR = 1

TICKER = "PFE"
FRED_SERIES = "SP500"

VOL_TERM_STRUCTURE_TICKERS = {"^VIX": 30, "^VIX3M": 93, "^VIX6M": 182}
VIX_FRED_SERIES = "VIXCLS"

N_MC_PATHS = 50000
MC_SEED = 42

SPOT_SCENARIO_RANGE = np.arange(0.60, 1.41, 0.05)  # 60% to 140% of S0, in 5pt steps


def fetch_index_path(entry_date, years=TENOR):
    start = pd.Timestamp(entry_date)
    end = start + pd.Timedelta(days=round(years * 365.25))
    path = fetch_daily_closes(TICKER, start, end, fred_series=FRED_SERIES)
    if path.index[-1] < end - pd.Timedelta(days=10):
        raise RuntimeError(
            f"Requested window {start.date()} to {end.date()} extends past the last available "
            f"trading day ({path.index[-1].date()}) - this script models a COMPLETED historical "
            f"window, not a live in-progress note. Pick an ENTRY_DATE/TENOR combination that ends "
            f"on or before today."
        )
    return path


def fetch_vol_term_structure(entry_date, years=TENOR):
    start = pd.Timestamp(entry_date)
    end = start + pd.Timedelta(days=round(years * 365.25))

    term_structure = {}
    for ticker, tenor_days in VOL_TERM_STRUCTURE_TICKERS.items():
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
    each date's EXACT joint first-passage probability (not the marginal
    N(d2), which double-counts). Full derivation in MATHEMATICS.md."""
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


def black_scholes_put(S, K, T, r, sigma, q=0.0):
    """Plain European put via QuantLib's AnalyticEuropeanEngine (dividend
    yield q is a native input). Used only for verify_against_closed_form -
    the actual autocall pricing is the Monte Carlo engine below."""
    if T <= 0:
        intrinsic = max(K - S, 0.0)
        delta = -1.0 if S < K else 0.0
        return {"price": intrinsic, "delta": delta, "vega": 0.0, "rho": 0.0, "theta": 0.0}

    today = ql.Date(1, 1, 2000)
    ql.Settings.instance().evaluationDate = today
    process = _quantlib_process(S, r, q, sigma, today)

    days = max(int(round(T * 365)), 1)
    exercise = ql.EuropeanExercise(today + ql.Period(days, ql.Days))
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    return {
        "price": option.NPV(), "delta": option.delta(), "vega": option.vega(),
        "rho": option.rho(), "theta": option.theta(),
    }


# REPLICATION: ZCB(Principal) - Short Put(1/Strike) - autocall feature (short
# call-like knockout at each quarterly observation date). See README.md for
# the product framing and MATHEMATICS.md for the full derivation, the Monte
# Carlo engine, and the closed-form verification checks.

def get_observation_dates(entry_date, tenor=TENOR, obs_per_year=OBS_PER_YEAR):
    """Quarterly observation dates; the last is maturity itself, not an
    autocall check (see determine_actual_call_date)."""
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


def determine_actual_call_date(path, S0, observation_dates, trigger=TRIGGER):
    """First historical observation date the index closes at/above the
    trigger, or None if it never does."""
    for obs_date in observation_dates[:-1]:
        if path[obs_date] >= trigger * S0:
            return obs_date
    return None


def simulate_forward_price(S_t, S0, valuation_date, maturity_date, future_call_obs_dates,
                            sigma, r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD,
                            strike=STRIKE, trigger=TRIGGER, n_paths=N_MC_PATHS, seed=MC_SEED, q=0.0):
    """Monte Carlo forward valuation under the risk-neutral measure. See
    MATHEMATICS.md section 5.2 for the simulation grid and the two-leg
    discounting split (principal always paid, put only if never called)."""
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
    levels = np.exp(log_levels)  # shape (n_paths, n_steps); last column = maturity level

    n_call_obs = len(future_call_obs_dates)
    if n_call_obs > 0:
        triggered = levels[:, :n_call_obs] >= trigger * S0
        any_triggered = triggered.any(axis=1)
        first_call_idx = np.argmax(triggered, axis=1)  # first True; index 0 (unused) where none
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


def finite_difference_greeks(S_t, S0, valuation_date, maturity_date, future_call_obs_dates, sigma, r, credit_spread, q=0.0):
    """Bump-and-reprice Greeks with common random numbers (same MC_SEED)
    so finite differences aren't swamped by independent MC noise."""
    eps_S = 0.005 * S0
    eps_sigma = 0.001
    eps_r = 0.0001

    def price_at(S=S_t, sig=sigma, rate=r, val_date=valuation_date, obs_dates=future_call_obs_dates):
        return simulate_forward_price(S, S0, val_date, maturity_date, obs_dates, sig, rate, credit_spread, q=q)["price"]

    price_mid = price_at()
    delta = (price_at(S=S_t + eps_S) - price_at(S=S_t - eps_S)) / (2 * eps_S)
    vega = (price_at(sig=sigma + eps_sigma) - price_at(sig=sigma - eps_sigma)) / (2 * eps_sigma)
    rho = (price_at(rate=r + eps_r) - price_at(rate=r - eps_r)) / (2 * eps_r)

    eps_T_days = min(1, max((maturity_date - valuation_date).days - 1, 0))
    if eps_T_days > 0:
        later_date = valuation_date + pd.Timedelta(days=eps_T_days)
        shifted_obs = [d for d in future_call_obs_dates if d > later_date]
        price_theta = price_at(val_date=later_date, obs_dates=shifted_obs)
        theta = (price_theta - price_mid) / (eps_T_days / 365.25)
    else:
        theta = 0.0

    return {"price": price_mid, "delta": delta, "vega": vega, "rho": rho, "theta": theta}


def verify_against_closed_form(S0, T, sigma, r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD, q=0.0):
    """With the trigger pushed unreachable, the MC engine must reduce to
    the plain FCN's closed-form price - see MATHEMATICS.md section 5.4."""
    maturity_date = pd.Timestamp(ENTRY_DATE) + pd.Timedelta(days=round(T * 365.25))
    mc_price = simulate_forward_price(
        S0, S0, pd.Timestamp(ENTRY_DATE), maturity_date, future_call_obs_dates=[],
        sigma=sigma, r=r, credit_spread=credit_spread, trigger=10.0, q=q,  # unreachable
    )["price"]

    zcb = zcb_price_and_greeks(S0, T, r + credit_spread)
    put = black_scholes_put(S0, STRIKE * S0, T, r, sigma, q)
    closed_form_price = zcb["price"] - (1.0 / STRIKE) * put["price"]

    rel_diff = abs(mc_price - closed_form_price) / closed_form_price
    return {
        "mc_price": mc_price, "closed_form_price": closed_form_price, "rel_diff": rel_diff,
    }


def autocallable_running_return(S0, path, actual_call_date):
    """Redemption Payoff (Relative to Par) - see README "Reading the
    charts" for what this tracker does and doesn't represent."""
    strike_level = STRIKE * S0
    below_strike = path < strike_level
    running = pd.Series(0.0, index=path.index)
    running[below_strike] = path[below_strike] / strike_level - 1.0

    if actual_call_date is not None:
        running[path.index >= actual_call_date] = 0.0

    return running


def autocallable_mtm_price_series(S0, path, vol_term_structure, observation_dates, actual_call_date,
                                   r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD, q=0.0):
    """Model value of the LIVE note only - series ends at actual_call_date
    (NaN after) so the plotted line stops at settlement. See README
    "Reading the charts" and autocall_settlement_proceeds_series below."""
    maturity_date = path.index[-1]
    prices = []
    for date, level in path.items():
        if actual_call_date is not None and date > actual_call_date:
            prices.append(np.nan)  # note has settled - nothing left to mark, line ends
            continue
        if actual_call_date is not None and date == actual_call_date:
            prices.append(S0)  # settlement value itself (par, as modeled here - excludes coupons)
            continue

        T_remaining = (maturity_date - date).days / 365.25
        vols_today = {tenor: series[date] for tenor, series in vol_term_structure.items()}
        sigma = interpolate_implied_vol(vols_today, max(T_remaining, 0.0))
        future_obs = [d for d in observation_dates[:-1] if d > date]
        price = simulate_forward_price(level, S0, date, maturity_date, future_obs, sigma, r, credit_spread, q=q)["price"]
        prices.append(price)

    return pd.Series(prices, index=path.index)


def autocall_settlement_proceeds_series(S0, path, actual_call_date):
    """Cash proceeds after an actual autocall, held flat (no
    reinvestment, excl. coupons) - see README "Reading the charts"."""
    proceeds = pd.Series(np.nan, index=path.index)
    if actual_call_date is not None:
        proceeds[path.index >= actual_call_date] = S0
    return proceeds


def plot_path(path, S0, observation_dates, actual_call_date, underlying_name, mtm_vol_term_structure=None,
              greeks=None, q=0.0):
    index_return_pct = (path / S0 - 1) * 100
    note_return_pct = autocallable_running_return(S0, path, actual_call_date) * 100

    _, ax = plt.subplots(figsize=(18, 8))
    ax.plot(index_return_pct.index, index_return_pct.values, color="firebrick", linewidth=1.5,
            label=underlying_name)
    ax.tick_params(axis="y", labelcolor="firebrick")
    ax.plot(note_return_pct.index, note_return_pct.values, color="indianred", linewidth=1.5,
            linestyle="dashed", label="Autocallable FCN Redemption Payoff (Relative to Par)")

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
    ax.set_ylabel("Return from Entry (%)", color="firebrick")
    lines, labels = ax.get_legend_handles_labels()
    ax.margins(x=0, y=0.05)

    if greeks is not None:
        greeks_text = (
            "Greeks at Inception:\n"
            f"Δ (Delta): {greeks['delta']:.2f}\n"
            f"ν (Vega):  {greeks['vega'] / S0 * 0.01:.4f} per 1% change in vol\n"
            f"ρ (Rho):   {greeks['rho'] / S0 * 0.01:.4f} per 1% change in SOFR\n"
            f"θ (Theta): {greeks['theta'] / S0:.4f} per year"
        )
        ax.text(1.06, 0.5, greeks_text, transform=ax.transAxes,
                fontsize=12, fontweight="light", color="black", va="center", ha="left")

    if mtm_vol_term_structure is not None:
        mtm_price = autocallable_mtm_price_series(S0, path, mtm_vol_term_structure, observation_dates,
                                                    actual_call_date, q=q)
        mtm_gain_over_par_pct = (mtm_price / S0 - 1) * 100

        ax2 = ax.twinx()
        mtm_line, = ax2.plot(
            mtm_gain_over_par_pct.index, mtm_gain_over_par_pct.values, color="darkred", linewidth=1.5,
            linestyle="solid", label="Autocallable FCN — Model Value of Redemption Component (% of Par, excl. coupons, Monte Carlo)")
        ax2.set_ylabel("Redemption Component Value vs. Par (%)", color="darkred", rotation=270, labelpad=10)
        ax2.tick_params(axis="y", labelcolor="darkred")

        if actual_call_date is not None:
            proceeds = autocall_settlement_proceeds_series(S0, path, actual_call_date)
            proceeds_gain_over_par_pct = (proceeds / S0 - 1) * 100
            proceeds_line, = ax2.plot(
                proceeds_gain_over_par_pct.index, proceeds_gain_over_par_pct.values, color="gray",
                linewidth=1.5, linestyle="dotted",
                label="Cash Proceeds After Autocall (par received, held flat, no reinvestment, excl. coupons)")
            lines.append(proceeds_line)
            labels.append(proceeds_line.get_label())

        combined_min = min(index_return_pct.min(), note_return_pct.min(), mtm_gain_over_par_pct.min(), strike_pct)
        combined_max = max(index_return_pct.max(), note_return_pct.max(), mtm_gain_over_par_pct.max(), trigger_pct)
        pad = (combined_max - combined_min) * 0.05
        ax.set_ylim(combined_min - pad, combined_max + pad)
        ax2.set_ylim(combined_min - pad, combined_max + pad)

        lines.append(mtm_line)
        labels.append(mtm_line.get_label())

    end_date = index_return_pct.index[-1]
    ax.set_title(f"{underlying_name} Price Return Path from {path.index[0].date()} to {end_date.date()} \n"
                 f"vs Model Value of Redemption Component (Excludes Coupons) — Autocallable Fixed Coupon Note")
    ax.legend(lines, labels, loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    underlying_name = fetch_underlying_name(TICKER)
    print(f"Fetching {underlying_name} path from {ENTRY_DATE} (entry) to +{TENOR}y (maturity)...")
    path = fetch_index_path(ENTRY_DATE)

    S0 = float(path.iloc[0])
    S_T = float(path.iloc[-1])
    vol = realized_annualized_vol(path)

    dividend_yield = fetch_dividend_yield(TICKER)
    print(f"Dividend yield for {underlying_name} ({TICKER}): {dividend_yield:.2%} (flat, continuous - 0% if an index)")

    observation_calendar_dates = get_observation_dates(ENTRY_DATE, TENOR, OBS_PER_YEAR)
    observation_dates = map_to_trading_days(observation_calendar_dates, path.index)
    actual_call_date = determine_actual_call_date(path, S0, observation_dates)
    note_return = autocallable_running_return(S0, path, actual_call_date).iloc[-1]

    print(f"\nAutocall schedule (quarterly, trigger={TRIGGER:.0%} of S0={S0:,.2f}):")
    for i, obs_date in enumerate(observation_dates, start=1):
        level = path.reindex([obs_date]).iloc[0] if obs_date in path.index else None
        tag = " (maturity, not an autocall check)" if i == len(observation_dates) else ""
        print(f"  Obs {i}: {obs_date.date()}  -  index level {level:,.2f} "
              f"({level / S0 - 1:+.2%} vs. S0){tag}")

    if actual_call_date is not None:
        print(f"\n>>> Note ACTUALLY AUTOCALLED on {actual_call_date.date()} in this historical path "
              f"(index closed at or above the {TRIGGER:.0%} trigger). Principal (as modeled here) is "
              f"returned early; no further downside exposure after this date. A real note would also "
              f"pay accrued coupons to that date - this script does not price or include coupons. <<<")
    else:
        print(f"\n>>> Note never autocalled in this historical path; ran to full maturity like the "
              f"plain Fixed Coupon Note. <<<")

    summary = pd.Series({
        "Entry Date": path.index[0].date(),
        "Maturity Date": path.index[-1].date(),
        "S0": f"{S0:,.2f}",
        "S_T": f"{S_T:,.2f}",
        "Strike Level": f"{STRIKE * S0:,.2f}",
        "Autocall Trigger Level": f"{TRIGGER * S0:,.2f}",
        "Actual Autocall Date": actual_call_date.date() if actual_call_date is not None else "Never (ran to maturity)",
        f"{underlying_name} Return": f"{S_T / S0 - 1:.2%}",
        "Autocallable FCN — Redemption Payoff (vs. Par, excl. coupons)": f"{note_return:.2%}",
        "Realized Vol (ann.)": f"{vol:.2%}",
        "Max Drawdown": f"{max_drawdown(path):.2%}",
    })

    print("\n" + summary.to_string())

    if TICKER.startswith("^"):
        print(f"\nFetching SPX implied vol term structure (VIX/VIX3M/VIX6M) for the same window...")
        raw_term_structure = fetch_vol_term_structure(ENTRY_DATE)
        vol_term_structure = {
            tenor: series.reindex(path.index).ffill().bfill()
            for tenor, series in raw_term_structure.items()
        }
        vols_at_entry = {tenor: series.iloc[0] for tenor, series in vol_term_structure.items()}
        entry_vol = interpolate_implied_vol(vols_at_entry, TENOR)
        vol_source_desc = f"interpolated from the {path.index[0].date()} VIX/VIX3M/VIX6M term structure"
    else:
        print(f"\n{TICKER} is a single name - no free historical implied-vol source exists for it, so")
        print(f"using its own trailing 2y REALIZED volatility instead of the SPX VIX/VIX3M/VIX6M proxy")
        print(f"(held flat for the note's life, same technique as the DCI/Multi-FCN/Multi-RC products):")
        entry_vol = fetch_trailing_realized_vol(TICKER, ENTRY_DATE)
        vol_term_structure = {TENOR: pd.Series(entry_vol, index=path.index)}
        vol_source_desc = f"{TICKER}'s own trailing 2y realized vol (not VIX-derived)"
        print(f"  {entry_vol:.2%}")

    print(f"\nVerifying the Monte Carlo engine against the plain FCN's closed-form price")
    print(f"(autocall trigger set unreachable - should collapse to ZCB - put_quantity*put):")
    check = verify_against_closed_form(S0, TENOR, entry_vol, q=dividend_yield)
    print(f"  Monte Carlo price:   {check['mc_price']:,.2f}")
    print(f"  Closed-form price:   {check['closed_form_price']:,.2f}")
    print(f"  Relative difference: {check['rel_diff']:.3%}  "
          f"({'PASS' if check['rel_diff'] < 0.01 else 'FAIL - investigate before trusting results'})")

    maturity_date = path.index[-1]
    future_obs_at_entry = [d for d in observation_dates[:-1] if d > path.index[0]]
    greeks = finite_difference_greeks(S0, S0, path.index[0], maturity_date, future_obs_at_entry,
                                       entry_vol, RISK_FREE_RATE, GS_CDS_SPREAD, q=dividend_yield)
    fair_value_pct_of_par = greeks["price"] / S0

    print(f"\nAutocallable Fixed Coupon Note — model value of redemption component at inception (excludes coupons)")
    print(f"(Monte Carlo, {N_MC_PATHS:,} paths, quarterly autocall obs at {TRIGGER:.0%} trigger,")
    print(f"ZCB leg discounted at SOFR {RISK_FREE_RATE:.2%} + GS CDS {GS_CDS_SPREAD:.2%}, put leg at SOFR")
    print(f"alone (dividend yield {dividend_yield:.2%}), T={TENOR}y, vol={entry_vol:.2%}")
    print(f"{vol_source_desc}):")
    print(f"  {fair_value_pct_of_par:.2%} of par ({fair_value_pct_of_par - 1:+.2%} vs. par)")

    mc_entry = simulate_forward_price(S0, S0, path.index[0], maturity_date, future_obs_at_entry,
                                       entry_vol, RISK_FREE_RATE, GS_CDS_SPREAD, q=dividend_yield)

    obs_years_at_entry = [(d - path.index[0]).days / 365.25 for d in future_obs_at_entry]

    print(f"\nClosed-form check on the autocall feature (short digital calls, trigger={TRIGGER:.0%}):")
    print(f"  Each date's digital call c=Q*e^(-rT)*N(d2) is weighted by the EXACT first-passage")
    print(f"  probability of being called then (not earlier), using the same joint-normal")
    print(f"  correlation structure (rho_ij = sqrt(min/max of t_i,t_j)) as Autocall Density.")
    strip = autocall_digital_strip_price(S0, obs_years_at_entry, RISK_FREE_RATE, entry_vol, TRIGGER,
                                          Q=TRIGGER * S0, q=dividend_yield)
    for obs_date, leg in zip(future_obs_at_entry, strip["per_date"]):
        print(f"  Obs {obs_date.date()} (T={leg['T']:.2f}y): "
              f"P(called exactly here)={leg['prob_exact']:.2%}, digital call PV={leg['price']:,.2f}")
    print(f"  Closed-form autocall strip PV: {strip['total_price']:,.2f}")
    print(f"  Closed-form P(called before maturity): {strip['total_prob_called']:.2%}")
    print(f"  Monte Carlo P(called before maturity):  {mc_entry['prob_called']:.2%}")

    print(f"\nAutocallable Fixed Coupon Note Greeks at inception (finite-difference, common random numbers):")
    print(f"  Delta: {greeks['delta']:.2f}")
    print(f"  Vega:  {greeks['vega'] / S0 * 0.01:.4f}  (per 1% change in vol)")
    print(f"  Rho:   {greeks['rho'] / S0 * 0.01:.4f}  (per 1% change in SOFR)")
    print(f"  Theta: {greeks['theta'] / S0:.4f} per year / {greeks['theta'] / S0 / 365:.5f} per day")

    print(f"\nModel Value of Redemption Component vs. Par at Inception (excludes coupons):")
    print(f"  Price: {greeks['price']:,.2f}  vs. Par (S0): {S0:,.2f}  ({fair_value_pct_of_par:.2%} of par)")

    plot_path(path, S0, observation_dates, actual_call_date, underlying_name,
              mtm_vol_term_structure=vol_term_structure, greeks=greeks, q=dividend_yield)
