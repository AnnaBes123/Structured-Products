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
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Reference product terms for this Greeks ladder - NOT synced to
# "Bullish Sharkfin.py" in this folder. The main script's STRIKE/BARRIER/
# TICKER are meant to be changed freely to backtest different names; this
# ladder intentionally stays on the generic reference terms below (see
# README) so it profiles the templated note, not whatever one-off name the
# main script currently points at. ---
STRIKE = 1.00
BARRIER = 1.20
REBATE = 0.0
RISK_FREE_RATE = 0.04
GS_CDS_SPREAD = 0.005308

ENTRY_DATE = "2025-01-02"
TENOR = 1

TICKER = "^GSPC"
FRED_SERIES = "SP500"

VOL_TERM_STRUCTURE_TICKERS = {"^VIX": 30, "^VIX3M": 93, "^VIX6M": 182}
VIX_FRED_SERIES = "VIXCLS"

# This ladder spans 60%-140% of S0, which crosses BARRIER - both the
# un-breached and breached regimes are covered (see breached branch below).
SPOT_SCENARIO_RANGE = np.arange(0.60, 1.41, 0.05)  # 60% to 140% of S0, in 5pt steps


def fetch_index_path(entry_date, years=TENOR):
    start = pd.Timestamp(entry_date)
    end = start + pd.Timedelta(days=round(years * 365.25))
    return fetch_daily_closes(TICKER, start, end, fred_series=FRED_SERIES)


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


def up_and_out_call_crr(S, K, H, T, r, sigma, steps, rebate=REBATE, q=0.0):
    """CRR up-and-out call, assumes NOT already breached - see
    ../../Yield/Barrier Reverse Convertible/MATHEMATICS.md."""
    if S >= H:
        return rebate
    if T <= 0:
        return max(S - K, 0.0)

    today = ql.Date(1, 1, 2000)
    ql.Settings.instance().evaluationDate = today
    process = _quantlib_process(S, r, q, sigma, today)

    days = max(int(round(T * 365)), 1)
    exercise = ql.EuropeanExercise(today + ql.Period(days, ql.Days))
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    option = ql.BarrierOption(ql.Barrier.UpOut, H, rebate, payoff, exercise)

    n = max(int(round(steps)), 2)
    option.setPricingEngine(ql.BinomialCRRBarrierEngine(process, n, n))
    return option.NPV()


def bullish_sharkfin_price(S, S0, T_remaining, sigma, breached, r=RISK_FREE_RATE,
                            credit_spread=GS_CDS_SPREAD, steps=None, q=0.0):
    """Once breached=True, the up-and-out call has permanently knocked out
    and contributes 0 to the note's forward value - capital returned at
    100% (the ZCB alone) from then on."""
    zcb = zcb_price_and_greeks(S0, T_remaining, r + credit_spread)

    if breached:
        call_price = 0.0
    else:
        K = STRIKE * S0
        H = BARRIER * S0
        n = steps if steps is not None else max(round(T_remaining * 252), 1)
        call_price = up_and_out_call_crr(S, K, H, T_remaining, r, sigma, n, REBATE, q)

    return zcb["price"] + call_price


# GREEKS LADDER: T, strike, barrier and funding curve fixed; only spot
# varies. Bump-and-reprice finite differences, wider bumps than the
# closed-form products - see this folder's README for the bump-size scan.

def finite_difference_greeks_at(S, S0, T, sigma, breached, r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD,
                                 steps=None, q=0.0):
    bump_S, bump_sigma, bump_r = S0 * 0.02, 0.02, 0.0001
    n = steps if steps is not None else max(round(T * 252), 1)

    price = bullish_sharkfin_price(S, S0, T, sigma, breached, r, credit_spread, steps=n, q=q)

    price_up_S = bullish_sharkfin_price(S + bump_S, S0, T, sigma, breached, r, credit_spread, steps=n, q=q)
    price_down_S = bullish_sharkfin_price(S - bump_S, S0, T, sigma, breached, r, credit_spread, steps=n, q=q)
    delta = (price_up_S - price_down_S) / (2 * bump_S)

    price_up_sigma = bullish_sharkfin_price(S, S0, T, sigma + bump_sigma, breached, r, credit_spread, steps=n, q=q)
    price_down_sigma = bullish_sharkfin_price(S, S0, T, sigma - bump_sigma, breached, r, credit_spread, steps=n, q=q)
    vega = (price_up_sigma - price_down_sigma) / (2 * bump_sigma)

    price_up_r = bullish_sharkfin_price(S, S0, T, sigma, breached, r + bump_r, credit_spread, steps=n, q=q)
    price_down_r = bullish_sharkfin_price(S, S0, T, sigma, breached, r - bump_r, credit_spread, steps=n, q=q)
    rho = (price_up_r - price_down_r) / (2 * bump_r)

    return {"price": price, "delta": delta, "vega": vega, "rho": rho}


def greek_sensitivity_table(S0, T, sigma, r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD,
                             spot_multiples=SPOT_SCENARIO_RANGE, q=0.0):
    rows = []
    for multiple in spot_multiples:
        S = multiple * S0
        breached = multiple >= BARRIER  # a spot scenario AT/ABOVE the barrier is already knocked out
        g = finite_difference_greeks_at(S, S0, T, sigma, breached, r, credit_spread, q=q)
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
        ("Rho (per 1% rate)", "Rho (per 1% change in SOFR)", "brown"),
    ]

    for ax, (col, ylabel, color) in zip(axes.flat, panels):
        ax.plot(spot, table[col], color=color, linewidth=1.8, marker="o", markersize=3)
        ax.axvline(100, color="lightgrey", linewidth=0.8, linestyle="dashed")
        ax.axvline(STRIKE * 100, color="dodgerblue", linewidth=0.8, linestyle="dotted")
        ax.axvline(BARRIER * 100, color="darkgreen", linewidth=0.8, linestyle="dotted")
        ax.axhline(0, color="lightgrey", linewidth=0.6)
        ax.grid(True, color="lightgrey", linewidth=0.4)
        ax.set_xlabel("Spot (% of S0)")
        ax.set_ylabel(ylabel)
        ax.set_title(col)

    fig.suptitle(
        f"{underlying_name} Bullish Sharkfin - Greeks Ladder (T, strike, barrier fixed, spot varies)\n"
        f"Fixed throughout: T={T}y, vol={sigma:.2%}, SOFR={r:.2%}, CDS={GS_CDS_SPREAD:.2%}, "
        f"strike={STRIKE:.0%}, barrier={BARRIER:.0%} of S0={S0:,.2f}"
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    print("Greeks Ladder - Bullish Sharkfin")
    print("=" * 60)
    print("This is NOT a new backtest and does not use the historical path.")
    print("It uses this ladder's own reference entry conditions (see the header")
    print("comment above), not whatever ticker/terms the main backtest script")
    print("currently has set,")
    print("then holds T, strike, barrier, funding curve and vol all fixed and")
    print("varies ONLY spot, so each Greek's value at a given spot level tells")
    print("you directly how much MTM moves for a 1-unit change in that")
    print("variable at that spot. Greeks are finite differences (bump and")
    print("reprice), not further differentiation of the barrier pricing,")
    print("since barrier Greeks can be discontinuous right at the barrier.\n")

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
    print(f"  Barrier:         {BARRIER:.0%} of S0 (knock-out, CRR lattice, checked once per trading day)")
    print(f"  Tenor (T):       {TENOR} year(s)")
    print(f"  Vol (sigma):     {entry_vol:.2%}  (from {vol_source_desc})")
    print(f"  SOFR proxy:      {RISK_FREE_RATE:.2%}")
    print(f"  GS CDS spread:   {GS_CDS_SPREAD:.2%}")
    print(f"  Dividend yield:  {dividend_yield:.2%}  (flat, continuous - 0% if an index)")

    sensitivity = greek_sensitivity_table(S0, TENOR, entry_vol, q=dividend_yield)
    print(f"\nGreeks Ladder (finite differences, bump-and-reprice):")
    print(sensitivity.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    plot_greek_sensitivity(sensitivity, S0, TENOR, entry_vol, RISK_FREE_RATE, underlying_name)
