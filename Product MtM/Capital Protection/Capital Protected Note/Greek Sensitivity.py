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
    zcb_price_and_greeks,
    fetch_risk_free_rate,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Reference product terms for this Greeks ladder - NOT synced to
# "Capital Protected Note.py" in this folder. The main script's TICKER is
# meant to be changed freely to backtest different names; this ladder
# intentionally stays on the generic reference terms below (see README) so
# it profiles the templated note, not whatever one-off name the main
# script currently points at. ---
STRIKE = 1.00
PARTICIPATION_RATE = 0.80
GS_CDS_SPREAD = 0.002675       # Goldman Sachs 1y CDS, 26.75 bps (Investing.com) - tenor-matched to TENOR=1

ENTRY_DATE = "2025-01-02"
TENOR = 1
RISK_FREE_RATE = fetch_risk_free_rate(ENTRY_DATE)  # 1Y Treasury CMT (FRED DGS1) as of ENTRY_DATE - real, historical; used for BOTH the ZCB leg and the option leg

TICKER = "^GSPC"
SPX_FRED_SERIES = "SP500"    # FRED fallback if yfinance fails - valid ONLY when TICKER
                             # is literally "^GSPC"; never used as a stand-in for a
                             # single-name stock's own price

SPX_VOL_TERM_STRUCTURE_TICKERS = {"^VIX": 30, "^VIX3M": 93, "^VIX6M": 182}  # SPX-only proxy;
                                                                            # only used when TICKER is an index (see below)
VIX_FRED_SERIES = "VIXCLS"

# T, strike, participation rate, funding curve and vol fixed; only spot
# varies below.
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


def black_scholes_call(S, K, T, r, sigma, q=0.0):
    """Plain European call via QuantLib's AnalyticEuropeanEngine (dividend
    yield q is a native input). No barrier on this product, so this
    closed-form engine gives the entire option leg's price AND Greeks
    directly - no finite-difference bump-and-reprice needed anywhere in
    this file."""
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


def capital_protected_note_price(S, S0, T_remaining, sigma, r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD, q=0.0):
    zcb = zcb_price_and_greeks(S0, T_remaining, r + credit_spread)
    call = black_scholes_call(S, STRIKE * S0, T_remaining, r, sigma, q)

    return {
        "price": zcb["price"] + PARTICIPATION_RATE * call["price"],
        "delta": PARTICIPATION_RATE * call["delta"],
        "vega": PARTICIPATION_RATE * call["vega"],
        "rho": zcb["rho"] + PARTICIPATION_RATE * call["rho"],
        "theta": zcb["theta"] + PARTICIPATION_RATE * call["theta"],
    }


# GREEKS LADDER: T, strike, participation rate, funding curve and vol
# fixed; only spot varies. All four Greeks are exact closed-form (no
# barrier, so no finite differences needed).

def greek_sensitivity_table(S0, T, sigma, r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD,
                             spot_multiples=SPOT_SCENARIO_RANGE, q=0.0):
    rows = []
    for multiple in spot_multiples:
        S = multiple * S0
        g = capital_protected_note_price(S, S0, T, sigma, r, credit_spread, q)
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
        ("Rho (per 1% rate)", "Rho (per 1% change in the 1Y Treasury rate)", "brown"),
    ]

    for ax, (col, ylabel, color) in zip(axes.flat, panels):
        ax.plot(spot, table[col], color=color, linewidth=1.8, marker="o", markersize=3)
        ax.axvline(100, color="lightgrey", linewidth=0.8, linestyle="dashed")
        ax.axvline(STRIKE * 100, color="dodgerblue", linewidth=0.8, linestyle="dotted")
        ax.axhline(0, color="lightgrey", linewidth=0.6)
        ax.grid(True, color="lightgrey", linewidth=0.4)
        ax.set_xlabel("Spot (% of S0)")
        ax.set_ylabel(ylabel)
        ax.set_title(col)

    fig.suptitle(
        f"{underlying_name} Capital Protected Note - Greeks Ladder (T, strike, participation fixed, spot varies)\n"
        f"Fixed throughout: T={T}y, vol={sigma:.2%}, 1Y Treasury={r:.2%}, CDS={GS_CDS_SPREAD:.2%}, "
        f"strike={STRIKE:.0%}, participation={PARTICIPATION_RATE:.0%} of S0={S0:,.2f}"
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    print("Greeks Ladder - Capital Protected Note")
    print("=" * 60)
    print("This is NOT a new backtest and does not use the historical path.")
    print("It uses this ladder's own reference entry conditions (see the header")
    print("comment above), not whatever ticker/terms the main backtest script")
    print("currently has set,")
    print("then holds T, strike, participation rate, funding curve and vol")
    print("all fixed and varies ONLY spot, so each Greek's value at a given")
    print("spot level tells you directly how much MTM moves for a 1-unit")
    print("change in that variable at that spot. All four Greeks are exact")
    print("closed-form - no barrier feature, no finite differences needed.\n")

    underlying_name = fetch_underlying_name(TICKER)
    print(f"Fetching {underlying_name} entry level for {ENTRY_DATE}...")
    path = fetch_index_path(ENTRY_DATE)
    S0 = float(path.iloc[0])

    dividend_yield = fetch_dividend_yield(TICKER)

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
    print(f"  Participation:   {PARTICIPATION_RATE:.0%}")
    print(f"  Tenor (T):       {TENOR} year(s)")
    print(f"  Vol (sigma):     {entry_vol:.2%}  (from {vol_source_desc})")
    print(f"  1Y Treasury (FRED DGS1): {RISK_FREE_RATE:.2%}")
    print(f"  GS CDS spread:   {GS_CDS_SPREAD:.2%}")
    print(f"  Dividend yield:  {dividend_yield:.2%}  (flat, continuous - 0% if an index)")

    sensitivity = greek_sensitivity_table(S0, TENOR, entry_vol, q=dividend_yield)
    print(f"\nGreeks Ladder (closed-form - exact, not finite-difference):")
    print(sensitivity.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    plot_greek_sensitivity(sensitivity, S0, TENOR, entry_vol, RISK_FREE_RATE, underlying_name)
