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
    interpolate_implied_vol,
    fetch_trailing_realized_vol,
    _quantlib_process,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Same product terms as "Outperfomance Certificate.py" in this folder ---
STRIKE = 1.00
OUTPERFORMANCE_PARTICIPATION = 2.00
RISK_FREE_RATE = 0.04

ENTRY_DATE = "2025-01-02"
TENOR = 1

TICKER = "NVDA"
FRED_SERIES = "SP500"

VOL_TERM_STRUCTURE_TICKERS = {"^VIX": 30, "^VIX3M": 93, "^VIX6M": 182}
VIX_FRED_SERIES = "VIXCLS"

# T, strike, vol and rate fixed; only spot varies below.
SPOT_SCENARIO_RANGE = np.arange(0.60, 1.41, 0.05)  # 60% to 140% of S0, in 5pt steps


def fetch_index_path(entry_date, years=TENOR):
    start = pd.Timestamp(entry_date)
    end = start + pd.DateOffset(years=years)
    return fetch_daily_closes(TICKER, start, end, fred_series=FRED_SERIES)


def fetch_vol_term_structure(entry_date, years=TENOR):
    start = pd.Timestamp(entry_date)
    end = start + pd.DateOffset(years=years)

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


def black_scholes_call(S, K, T, r, sigma, q=0.0):
    """Plain European call via QuantLib's AnalyticEuropeanEngine (dividend
    yield q is a native input)."""
    if T <= 0:
        intrinsic = max(S - K, 0.0)
        delta = 1.0 if S > K else 0.0
        return {"price": intrinsic, "delta": delta, "vega": 0.0, "rho": 0.0, "theta": 0.0}

    today = ql.Date(1, 1, 2000)
    ql.Settings.instance().evaluationDate = today
    process = _quantlib_process(S, r, q, sigma, today)

    days = max(int(round(T * 365)), 1)
    exercise = ql.EuropeanExercise(today + ql.Period(days, ql.Days))
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    return {
        "price": option.NPV(), "delta": option.delta(), "vega": option.vega(),
        "rho": option.rho(), "theta": option.theta(),
    }


def lepo_price_and_greeks(S, T, q=0.0):
    """LEPO (Low Exercise Price Option): the risk-neutral PV of receiving
    one share at maturity, S*e^(-qT) - equals spot exactly only when q=0.
    See the main certificate file / README for the full rationale."""
    discount_q = np.exp(-q * T)
    return {
        "price": S * discount_q, "delta": discount_q, "vega": 0.0, "rho": 0.0,
        "theta": q * S * discount_q,
    }


def outperformance_certificate_mtm(S, S0, T_remaining, sigma, r=RISK_FREE_RATE, q=0.0):
    zero_strike_leg = lepo_price_and_greeks(S, T_remaining, q)
    atm_leg = black_scholes_call(S, STRIKE * S0, T_remaining, r, sigma, q)
    atm_quantity = OUTPERFORMANCE_PARTICIPATION - 1.0

    return {
        "price": zero_strike_leg["price"] + atm_quantity * atm_leg["price"],
        "delta": zero_strike_leg["delta"] + atm_quantity * atm_leg["delta"],
        "vega": zero_strike_leg["vega"] + atm_quantity * atm_leg["vega"],
        "rho": zero_strike_leg["rho"] + atm_quantity * atm_leg["rho"],
        "theta": zero_strike_leg["theta"] + atm_quantity * atm_leg["theta"],
    }


# GREEKS LADDER: T, strike, vol and rate fixed; only spot varies below.

def greek_sensitivity_table(S0, T, sigma, r=RISK_FREE_RATE, spot_multiples=SPOT_SCENARIO_RANGE, q=0.0):
    rows = []
    for multiple in spot_multiples:
        S = multiple * S0
        g = outperformance_certificate_mtm(S, S0, T, sigma, r, q=q)
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
        ("Price (% of Par)", "Model Value (% of Par)", "firebrick"),
        ("Delta", "Delta", "darkred"),
        ("Vega (per 1% vol)", "Vega (per 1% change in vol)", "indianred"),
        ("Rho (per 1% rate)", "Rho (per 1% change in rates)", "brown"),
    ]

    for ax, (col, ylabel, color) in zip(axes.flat, panels):
        ax.plot(spot, table[col], color=color, linewidth=1.8, marker="o", markersize=3)
        ax.axvline(100, color="lightgrey", linewidth=0.8, linestyle="dashed")
        ax.axhline(0, color="lightgrey", linewidth=0.6)
        ax.grid(True, color="lightgrey", linewidth=0.4)
        ax.set_xlabel("Spot (% of S0)")
        ax.set_ylabel(ylabel)
        ax.set_title(col)

    fig.suptitle(
        f"{underlying_name} Outperformance Certificate - Greeks Ladder (T and strike fixed, spot varies)\n"
        f"Fixed throughout: T={T}y, vol={sigma:.2%}, r={r:.2%}, strike={STRIKE:.0%} of S0={S0:,.2f}, "
        f"participation={OUTPERFORMANCE_PARTICIPATION:.0%}"
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    print("Greeks Ladder - Outperformance Certificate")
    print("=" * 60)
    print("This is NOT a new backtest and does not use the historical path.")
    print("It reuses the same entry conditions as the main backtest script,")
    print("then holds T, strike, vol and rate all fixed and varies ONLY spot,")
    print("so each Greek's value at a given spot level tells you directly how")
    print("much MTM moves for a 1-unit change in that variable at that spot")
    print("(e.g. the Vega row at spot=80% is what a 1% vol move is worth if")
    print("the index has already dropped 20%).\n")

    underlying_name = fetch_underlying_name(TICKER)
    print(f"Fetching {underlying_name} entry level for {ENTRY_DATE}...")
    path = fetch_index_path(ENTRY_DATE)
    S0 = float(path.iloc[0])

    dividend_yield = fetch_dividend_yield(TICKER)

    if TICKER.startswith("^"):
        print(f"Fetching SPX implied vol term structure for {ENTRY_DATE}...")
        raw_term_structure = fetch_vol_term_structure(ENTRY_DATE)
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
    print(f"  Tenor (T):       {TENOR} year(s)")
    print(f"  Vol (sigma):     {entry_vol:.2%}  (from {vol_source_desc})")
    print(f"  Risk-free rate:  {RISK_FREE_RATE:.2%}")
    print(f"  Dividend yield:  {dividend_yield:.2%}  (flat, continuous - 0% if an index)")

    sensitivity = greek_sensitivity_table(S0, TENOR, entry_vol, RISK_FREE_RATE, q=dividend_yield)
    print(f"\nGreeks Ladder (QuantLib closed-form Black-Scholes-Merton - exact, not finite-difference,")
    print(f"since the outperformance certificate is a sum of plain calls with a")
    print(f"known analytic derivative):")
    print(sensitivity.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    plot_greek_sensitivity(sensitivity, S0, TENOR, entry_vol, RISK_FREE_RATE, underlying_name)
