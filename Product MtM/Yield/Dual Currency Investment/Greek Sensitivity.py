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
    realized_annualized_vol,
    _quantlib_process,
    zcb_price_and_greeks,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Same product terms as "DCI.py" in this folder ---
STRIKE = 0.95
DEPOSIT_CCY = "USD"
ALT_CCY = "EUR"
DEPOSIT_RATE = 0.04
ALT_RATE = 0.02
ISSUER_CDS_SPREAD = 0.005308

ENTRY_DATE = "2025-01-02"
TENOR = 0.25

# T, strike, rates and vol fixed throughout; only spot varies below.
SPOT_SCENARIO_RANGE = np.arange(0.80, 1.21, 0.02)  # 80% to 120% of S0, in 2pt steps


def fetch_fx_path(deposit_ccy, alt_ccy, entry_date, years=TENOR):
    """S = DEPOSIT amount per 1 ALT unit. See DCI.py in this folder for the
    Yahoo Finance ticker-convention verification."""
    ticker = f"{alt_ccy}{deposit_ccy}=X"
    start = pd.Timestamp(entry_date)
    end = start + pd.Timedelta(days=round(years * 365.25))

    try:
        import yfinance as yf
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if data is None or data.empty:
            raise RuntimeError("empty response")
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        series = close.dropna()
    except Exception as exc:
        raise RuntimeError(
            f"Could not fetch {ticker} from yfinance ({exc}) - check that {deposit_ccy}/{alt_ccy} "
            f"is a valid, listed FX pair.")

    if series.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    return series


def black_scholes_put(S, K, T, r, sigma, q=0.0):
    """Plain European put via QuantLib's AnalyticEuropeanEngine. With q set
    to the ALT currency's own short rate, this process IS Garman-Kohlhagen -
    see DCI.py in this folder."""
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


def dci_mtm(S, S0, T_remaining, sigma, r=DEPOSIT_RATE, alt_rate=ALT_RATE, credit_spread=ISSUER_CDS_SPREAD):
    zcb = zcb_price_and_greeks(S0, T_remaining, r + credit_spread)
    put = black_scholes_put(S, STRIKE * S0, T_remaining, r, sigma, alt_rate)
    put_quantity = 1.0 / STRIKE

    return {
        "price": zcb["price"] - put_quantity * put["price"],
        "delta": -put_quantity * put["delta"],
        "vega": -put_quantity * put["vega"],
        "rho": zcb["rho"] - put_quantity * put["rho"],
        "theta": zcb["theta"] - put_quantity * put["theta"],
    }


# GREEKS LADDER: T, strike and rates fixed; only spot varies. Theta excluded
# (T never moves). See README.

def greek_sensitivity_table(S0, T, sigma, r=DEPOSIT_RATE, alt_rate=ALT_RATE, spot_multiples=SPOT_SCENARIO_RANGE):
    rows = []
    for multiple in spot_multiples:
        S = multiple * S0
        g = dci_mtm(S, S0, T, sigma, r, alt_rate)
        rows.append({
            "Spot (% of S0)": multiple * 100,
            "Price (% of Par)": g["price"] / S0 * 100,
            "Delta": g["delta"],
            "Vega (per 1% vol)": g["vega"] / S0 * 0.01,
            "Rho (per 1% rate)": g["rho"] / S0 * 0.01,
        })
    return pd.DataFrame(rows)


def plot_greek_sensitivity(table, S0, T, sigma, r, pair_label):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    spot = table["Spot (% of S0)"]

    panels = [
        ("Price (% of Par)", "Model Value (% of Par)", "firebrick"),
        ("Delta", "Delta", "darkred"),
        ("Vega (per 1% vol)", "Vega (per 1% change in vol)", "indianred"),
        ("Rho (per 1% rate)", "Rho (per 1% change in rate)", "brown"),
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
        f"{pair_label} DCI - Greeks Ladder (T, strike, rates fixed, spot varies)\n"
        f"Fixed throughout: T={T}y, vol={sigma:.2%}, {DEPOSIT_CCY} rate={r:.2%}, {ALT_CCY} rate={ALT_RATE:.2%}, "
        f"CDS={ISSUER_CDS_SPREAD:.2%}, strike={STRIKE:.0%} of S0"
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    pair_label = f"{ALT_CCY}{DEPOSIT_CCY}"
    print("Greeks Ladder - Dual Currency Investment")
    print("=" * 60)
    print("This is NOT a new backtest and does not use the historical path.")
    print("It reuses the same entry conditions as the main backtest script,")
    print("then holds T, strike and rates all fixed and varies ONLY spot, so")
    print("each Greek's value at a given spot level tells you directly how")
    print("much MTM moves for a 1-unit change in that variable at that spot.")
    print("Theta is excluded since T never varies in this analysis.\n")

    print(f"Fetching {pair_label} entry level for {ENTRY_DATE}...")
    path = fetch_fx_path(DEPOSIT_CCY, ALT_CCY, ENTRY_DATE)
    S0 = float(path.iloc[0])
    entry_vol = realized_annualized_vol(path)

    print(f"\nFixed throughout (only spot varies below):")
    print(f"  Pair:            {pair_label} (S = {DEPOSIT_CCY} per 1 {ALT_CCY})")
    print(f"  Entry Date:      {ENTRY_DATE}")
    print(f"  S0 (spot):       {S0:.4f}")
    print(f"  Strike:          {STRIKE:.0%} of S0")
    print(f"  Tenor (T):       {TENOR} year(s)")
    print(f"  Vol (sigma):     {entry_vol:.2%}  (realized historical vol of {pair_label})")
    print(f"  {DEPOSIT_CCY} rate:       {DEPOSIT_RATE:.2%}")
    print(f"  {ALT_CCY} rate:       {ALT_RATE:.2%}")
    print(f"  Issuer CDS:      {ISSUER_CDS_SPREAD:.2%}")

    sensitivity = greek_sensitivity_table(S0, TENOR, entry_vol, DEPOSIT_RATE, ALT_RATE)
    print(f"\nGreeks Ladder (QuantLib closed-form Garman-Kohlhagen for the put, closed-form for the")
    print(f"deposit leg - both exact, not finite-difference):")
    print(sensitivity.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    plot_greek_sensitivity(sensitivity, S0, TENOR, entry_vol, DEPOSIT_RATE, pair_label)
