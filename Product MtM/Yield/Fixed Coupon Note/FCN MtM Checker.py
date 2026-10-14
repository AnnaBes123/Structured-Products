"""Indicative model mark of a still-LIVE autocallable FCN, from its real term
sheet dates. Companion to "Fixed Coupon Note (Autocallable).py" (completed
historical windows only), whose Monte Carlo engine it reuses. Redemption
component only - coupons are not priced. See README.md "Marking a live FCN"
and DEVLOG.md."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"

_PRODUCT_MTM_ROOT = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(_PRODUCT_MTM_ROOT) != "Product MtM":
    _PRODUCT_MTM_ROOT = os.path.dirname(_PRODUCT_MTM_ROOT)
if _PRODUCT_MTM_ROOT not in sys.path:
    sys.path.insert(0, _PRODUCT_MTM_ROOT)

from _common import (
    fetch_daily_closes,
    fetch_dividend_yield,
    fetch_underlying_name,
    fetch_risk_free_rate_series,
    zcb_price_and_greeks,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# Importing the sibling runs its module-level constants (one FRED call) but not its __main__ block.
_spec = importlib.util.spec_from_file_location(
    "fcn_autocallable", os.path.join(SCRIPT_DIR, "Fixed Coupon Note (Autocallable).py"))
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)

# --- Term sheet ---
TICKER = "EWY"
STRIKE = 0.65                         # short put strike, as a fraction of the initial fixing
TRIGGER = 1.00                        # autocall level, as a fraction of the initial fixing
TRADE_DATE = "2026-05-23"
INITIAL_FIXING = 178.0                # term-sheet level; None = last close on/before TRADE_DATE
AUTOCALL_OBS_DATES = ["2026-08-19"]   # autocall checks only - the final fixing is not one
FINAL_FIXING_DATE = "2026-11-19"      # put fixes here; redemption follows a few days later
ISSUER_CDS_SPREAD = 0.002675          # issuer credit spread, principal leg only (GS 1y CDS default)

VALUATION_DATE = None                 # None = latest available close
VOL_LOOKBACK_YEARS = 2                # trailing realized vol window, re-estimated on each valuation date
IMPLIED_VOL_OVERRIDE = 0.6825         # e.g. 0.55 = a quoted implied vol for the valuation-date mark; history keeps realized vol

N_MC_PATHS = 100000
MC_SEED = 42


def trailing_realized_vol(history, date, lookback_years=VOL_LOOKBACK_YEARS):
    window = history.loc[date - pd.Timedelta(days=round(lookback_years * 365.25)):date]
    log_returns = np.log(window / window.shift(1)).dropna()
    return float(log_returns.std() * np.sqrt(252))


def past_observations(path, S0, obs_dates, trigger=TRIGGER):
    """(calendar date, trading day used, close, called?) for each autocall
    observation that has already happened."""
    rows = []
    for cal_date in obs_dates:
        if cal_date > path.index[-1]:
            continue
        trading_day = path.index[path.index.searchsorted(cal_date)]
        level = float(path[trading_day])
        rows.append((cal_date, trading_day, level, level >= trigger * S0))
    return rows


def model_value(S, S0, date, final_date, future_obs, sigma, r, q, n_paths=N_MC_PATHS):
    return engine.simulate_forward_price(
        S, S0, date, final_date, future_obs, sigma, r, ISSUER_CDS_SPREAD,
        strike=STRIKE, trigger=TRIGGER, n_paths=n_paths, seed=MC_SEED, q=q)


def closed_form_value(S, S0, T, sigma, r, q):
    """ZCB - (1/strike) * put: exact once no autocall observation remains."""
    zcb = zcb_price_and_greeks(S0, T, r + ISSUER_CDS_SPREAD)["price"]
    put = engine.black_scholes_put(S, STRIKE * S0, T, r, sigma, q)["price"]
    return zcb - put / STRIKE


def mtm_series(path, S0, vols, rates, obs_dates, final_date, call_date, q):
    """Model value on each date the note was live; par on the call date, NaN after."""
    values = []
    for date, level in path.items():
        if call_date is not None and date >= call_date:
            values.append(S0 if date == call_date else np.nan)
            continue
        future_obs = [d for d in obs_dates if d > date]
        values.append(model_value(level, S0, date, final_date, future_obs, vols[date], rates[date], q)["price"])
    return pd.Series(values, index=path.index)


def bumped_greeks(S, S0, date, final_date, future_obs, sigma, r, q):
    """Bump-and-reprice with common random numbers (same seed)."""
    eps_S, eps_sigma = 0.005 * S0, 0.001

    def price(spot=S, sig=sigma):
        return model_value(spot, S0, date, final_date, future_obs, sig, r, q)["price"]

    return {
        "delta": (price(spot=S + eps_S) - price(spot=S - eps_S)) / (2 * eps_S),
        "vega": (price(sig=sigma + eps_sigma) - price(sig=sigma - eps_sigma)) / (2 * eps_sigma),
    }


def plot_mark(path, S0, mtm, obs_dates, final_date, call_date, underlying_name):
    underlying_pct = (path / S0 - 1) * 100
    mtm_pct = (mtm / S0 - 1) * 100

    _, ax = plt.subplots(figsize=(16, 10))
    ax.plot(underlying_pct.index, underlying_pct.values, color="indianred", linewidth=1.2,
            label=f"{underlying_name} vs. Initial Fixing")
    ax.plot(mtm_pct.index, mtm_pct.values, color="darkred", linewidth=2.2,
            label="Autocallable FCN — Model Value of Redemption Component (% of Par, excl. coupons, Monte Carlo)")

    ax.grid(True, which="major", color="lightgrey", linewidth=0.6)
    for level, name in (((STRIKE - 1) * 100, "Strike"), ((TRIGGER - 1) * 100, "Autocall Trigger")):
        ax.axhline(level, color="dodgerblue", linewidth=0.8, linestyle="dotted")
        ax.text(0.01, level, f"{name}: {level:.1f}%", transform=ax.get_yaxis_transform(),
                color="dodgerblue", fontsize=9, va="bottom", ha="left")

    for obs_date in obs_dates:
        ax.axvline(obs_date, color="mediumseagreen", linewidth=1, linestyle="dashed")
        ax.text(obs_date, 0.98, "  Autocall Obs", transform=ax.get_xaxis_transform(),
                color="mediumseagreen", fontsize=9, va="top", ha="left")
    ax.axvline(final_date, color="dimgray", linewidth=1, linestyle="dashed")
    ax.text(final_date, 0.98, "Final Fixing  ", transform=ax.get_xaxis_transform(),
            color="dimgray", fontsize=9, va="top", ha="right")
    if call_date is not None:
        ax.axvline(call_date, color="darkgreen", linewidth=1.2, linestyle="dashed")
        ax.text(call_date, 0.93, "  Autocalled", transform=ax.get_xaxis_transform(),
                color="darkgreen", fontsize=9, va="top", ha="left")

    last_date = mtm.dropna().index[-1]
    ax.annotate(f"  {mtm_pct[last_date]:+.2f}% vs. par\n  ({last_date.date()})", (last_date, mtm_pct[last_date]),
                color="darkred", fontsize=10, va="center", ha="left")

    ax.set_xlim(path.index[0], final_date + pd.Timedelta(days=5))
    ax.set_xlabel("Date")
    ax.set_ylabel("% vs. Initial Fixing / Par")
    ax.set_title(f"Live Autocallable Fixed Coupon Note on {underlying_name} — Indicative Model Mark as of {path.index[-1].date()}\n"
                 f"Redemption Component Only (Excludes Coupons), Strike {STRIKE:.0%}, Trigger {TRIGGER:.0%}, "
                 f"Initial Fixing {S0:,.2f}")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.06), fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    trade_date = pd.Timestamp(TRADE_DATE)
    final_date = pd.Timestamp(FINAL_FIXING_DATE)
    obs_dates = sorted(pd.Timestamp(d) for d in AUTOCALL_OBS_DATES)
    valuation_cap = pd.Timestamp(VALUATION_DATE) if VALUATION_DATE else pd.Timestamp.today().normalize()
    valuation_cap = min(valuation_cap, final_date)

    underlying_name = fetch_underlying_name(TICKER)
    print(f"Fetching {underlying_name} ({TICKER}) closes up to {valuation_cap.date()}...")
    history = fetch_daily_closes(
        TICKER, trade_date - pd.Timedelta(days=round(VOL_LOOKBACK_YEARS * 365.25) + 10),
        valuation_cap + pd.Timedelta(days=1))
    start_day = history.index[history.index.searchsorted(trade_date, side="right") - 1]
    path = history.loc[start_day:]
    if len(path) < 2:
        raise RuntimeError(f"No {TICKER} closes after TRADE_DATE {trade_date.date()} yet - nothing to mark.")

    S0 = float(INITIAL_FIXING) if INITIAL_FIXING is not None else float(path.iloc[0])
    fixing_source = "pinned term-sheet level" if INITIAL_FIXING is not None else f"{start_day.date()} close"
    q = fetch_dividend_yield(TICKER)
    rates = fetch_risk_free_rate_series(path.index[0], path.index[-1]).reindex(path.index, method="ffill").bfill()
    vols = pd.Series({d: trailing_realized_vol(history, d) for d in path.index})
    realized_vol_today = float(vols.iloc[-1])
    if IMPLIED_VOL_OVERRIDE is not None:
        vols.iloc[-1] = float(IMPLIED_VOL_OVERRIDE)

    observed = past_observations(path, S0, obs_dates)
    call_date = next((day for _, day, _, called in observed if called), None)
    # Past observations are compared on the trading day actually used; future ones stay calendar dates.
    obs_for_pricing = [day for _, day, _, _ in observed] + [d for d in obs_dates if d > path.index[-1]]

    val_date = path.index[-1]
    S_t = float(path.iloc[-1])
    print(f"\nTerms: strike {STRIKE:.0%} ({STRIKE * S0:,.2f}), autocall trigger {TRIGGER:.0%} ({TRIGGER * S0:,.2f}), "
          f"initial fixing {S0:,.2f} ({fixing_source}), final fixing {final_date.date()}")

    print("\nAutocall observations:")
    for cal_date, day, level, called in observed:
        print(f"  {cal_date.date()} (close of {day.date()}): {level:,.2f} = {level / S0:.2%} of fixing  ->  "
              f"{'AUTOCALLED' if called else 'not called'}")
    for d in obs_dates:
        if d > val_date:
            print(f"  {d.date()}: upcoming ({(d - val_date).days} days)")

    mtm = mtm_series(path, S0, vols, rates, obs_for_pricing, final_date, call_date, q)

    if call_date is not None:
        print(f"\n>>> Note AUTOCALLED on {call_date.date()} - redeemed at par (plus coupons, not modeled). "
              f"Nothing left to mark. <<<")
    else:
        sigma, r = float(vols[val_date]), float(rates[val_date])
        T = (final_date - val_date).days / 365.25
        future_obs = [d for d in obs_for_pricing if d > val_date]
        mark = model_value(S_t, S0, val_date, final_date, future_obs, sigma, r, q)
        greeks = bumped_greeks(S_t, S0, val_date, final_date, future_obs, sigma, r, q)

        print(f"\nIndicative model mark as of {val_date.date()} close (redemption component only, excludes coupons):")
        print(f"  {TICKER} close:            {S_t:,.2f}  ({S_t / S0:.2%} of fixing)")
        print(f"  Distance to trigger:    {S_t / (TRIGGER * S0) - 1:+.2%}")
        print(f"  Distance to strike:     {S_t / (STRIKE * S0) - 1:+.2%}")
        print(f"  Time to final fixing:   {(final_date - val_date).days} days ({T:.3f}y)")
        vol_source = (f"IMPLIED_VOL_OVERRIDE; trailing {VOL_LOOKBACK_YEARS}y realized is {realized_vol_today:.2%}"
                      if IMPLIED_VOL_OVERRIDE is not None else f"trailing {VOL_LOOKBACK_YEARS}y realized, not implied")
        print(f"  Inputs: vol {sigma:.2%} ({vol_source}), 1Y Treasury {r:.2%}, "
              f"issuer spread {ISSUER_CDS_SPREAD:.2%}, dividend yield {q:.2%}")
        print(f"  Model value:            {mark['price'] / S0:.2%} of par ({mark['price'] / S0 - 1:+.2%} vs. par)")
        print(f"  Delta:                  {greeks['delta'] * 0.01 * S_t / S0:+.3%} of par per +1% move in {TICKER}")
        print(f"  Vega:                   {greeks['vega'] / S0 * 0.01:+.3%} of par per +1 vol point")
        if future_obs:
            print(f"  Risk-neutral P(autocalled at a remaining observation): {mark['prob_called']:.2%} "
                  f"(pricing-measure, not a real-world forecast)")
        else:
            closed = closed_form_value(S_t, S0, T, sigma, r, q)
            rel_diff = abs(mark["price"] - closed) / closed
            print(f"  No autocall observation remains, so the note is exactly ZCB - (1/strike) * put:")
            print(f"    closed form {closed / S0:.2%} of par vs. Monte Carlo {mark['price'] / S0:.2%}  "
                  f"(diff {rel_diff:.3%}, {'PASS' if rel_diff < 0.01 else 'FAIL - investigate'})")

    plot_mark(path, S0, mtm, obs_dates, final_date, call_date, underlying_name)
