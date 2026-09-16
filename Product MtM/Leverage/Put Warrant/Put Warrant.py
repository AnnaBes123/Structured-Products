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
    realized_annualized_vol,
    max_drawdown,
    _quantlib_process,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Product terms ---
STRIKE = 1.00                  # put strike, as a fraction of S0 (100% - at the money by default)
RISK_FREE_RATE = 0.04

ENTRY_DATE = "2025-01-02"
TENOR = 1

# TICKER drives dividend handling automatically - see fetch_dividend_yield.
TICKER = "^GSPC"
FRED_SERIES = "SP500"

VOL_TERM_STRUCTURE_TICKERS = {"^VIX": 30, "^VIX3M": 93, "^VIX6M": 182}
VIX_FRED_SERIES = "VIXCLS"

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


# REPLICATION: Long European Vanilla Put, struck at STRIKE - mirror of the
# Call Warrant. No ZCB/par - price/Greeks in raw underlying-price units,
# not % of par. See README.

def put_warrant_price(S, K, T, r, sigma, q=0.0):
    """Plain European put via QuantLib's AnalyticEuropeanEngine. This IS
    the entire product - a put warrant is nothing more than this."""
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


def put_warrant_intrinsic_value(S0, path):
    """Intrinsic value if exercised today, max(Strike-S, 0) - not what
    you'd receive if sold today. See README "Reading the charts"."""
    strike_level = STRIKE * S0
    return pd.Series(np.maximum(strike_level - path.values, 0.0), index=path.index)


def put_warrant_mtm_price_series(S0, path, vol_term_structure, r=RISK_FREE_RATE, q=0.0):
    maturity_date = path.index[-1]
    K = STRIKE * S0
    prices = []
    for date, level in path.items():
        T_remaining = (maturity_date - date).days / 365.25
        vols_today = {tenor: series[date] for tenor, series in vol_term_structure.items()}
        sigma = interpolate_implied_vol(vols_today, max(T_remaining, 0.0))
        prices.append(put_warrant_price(level, K, T_remaining, r, sigma, q)["price"])
    return pd.Series(prices, index=path.index)


def plot_path(path, S0, underlying_name, mtm_vol_term_structure=None, greeks=None, q=0.0):
    index_return_pct = (path / S0 - 1) * 100
    intrinsic_value = put_warrant_intrinsic_value(S0, path)

    _, ax = plt.subplots(figsize=(18, 8))
    ax.plot(index_return_pct.index, index_return_pct.values, color="firebrick", linewidth=1.5,
            label=underlying_name)
    ax.tick_params(axis="y", labelcolor="firebrick")
    ax.set_xlabel("Date")
    ax.set_ylabel("Underlying Return from Entry (%)", color="firebrick")

    strike_pct = (STRIKE - 1) * 100
    ax.axhline(strike_pct, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.text(0.01, strike_pct, f"Strike: {strike_pct:.1f}%", transform=ax.get_yaxis_transform(),
            color="dodgerblue", fontsize=9, va="bottom", ha="left")
    ax.margins(x=0, y=0.05)
    lines, labels = ax.get_legend_handles_labels()

    if greeks is not None:
        greeks_text = (
            "Greeks at Inception:\n"
            f"Δ (Delta): {greeks['delta']:.3f}\n"
            f"ν (Vega):  {greeks['vega'] * 0.01:.4f} per 1% change in vol\n"
            f"ρ (Rho):   {greeks['rho'] * 0.01:.4f} per 1% change in rates\n"
            f"θ (Theta): {greeks['theta']:.4f} per year"
        )
        ax.text(1.06, 0.5, greeks_text, transform=ax.transAxes,
                fontsize=12, fontweight="light", color="black", va="center", ha="left")

    # Right axis: the warrant's own value, in the underlying's price units
    # (points/$) - NOT a percentage, since there's no par to express it
    # against. Deliberately NOT sharing y-limits with the left axis (unlike
    # every note product in this repo) - a warrant's premium and the
    # underlying's own price live on genuinely different scales, and
    # forcing them onto one shared range would just flatten the warrant
    # line into invisibility.
    ax2 = ax.twinx()
    ax2.plot(intrinsic_value.index, intrinsic_value.values, color="indianred", linewidth=1.5,
              linestyle="dashed", label="Put Warrant Intrinsic Value (if exercised today)")

    if mtm_vol_term_structure is not None:
        mtm_price = put_warrant_mtm_price_series(S0, path, mtm_vol_term_structure, q=q)
        mtm_line, = ax2.plot(
            mtm_price.index, mtm_price.values, color="darkred", linewidth=1.5,
            linestyle="solid", label="Put Warrant — Model Value")
        right_lines, right_labels = ax2.get_legend_handles_labels()
        lines += right_lines
        labels += right_labels
    else:
        right_lines, right_labels = ax2.get_legend_handles_labels()
        lines += right_lines
        labels += right_labels

    ax2.set_ylabel(f"Warrant Value ({TICKER} price units)", color="darkred", rotation=270, labelpad=15)
    ax2.tick_params(axis="y", labelcolor="darkred")
    ax2.set_ylim(bottom=0)

    end_date = index_return_pct.index[-1]
    ax.set_title(f"{underlying_name} Price Return Path from {path.index[0].date()} to {end_date.date()} \n"
                 f"vs Model Value ({TICKER} price units) — Put Warrant")
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
    K = STRIKE * S0
    terminal_intrinsic = max(K - S_T, 0.0)

    dividend_yield = fetch_dividend_yield(TICKER)
    print(f"Dividend yield for {underlying_name} ({TICKER}): {dividend_yield:.2%} (flat, continuous - 0% if an index)")

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

    greeks = put_warrant_price(S0, K, TENOR, RISK_FREE_RATE, entry_vol, dividend_yield)
    entry_premium = greeks["price"]

    summary = pd.Series({
        "Entry Date": path.index[0].date(),
        "Maturity Date": path.index[-1].date(),
        "S0": f"{S0:,.2f}",
        "S_T": f"{S_T:,.2f}",
        "Strike Level": f"{K:,.2f}",
        f"{underlying_name} Return": f"{S_T / S0 - 1:.2%}",
        "Entry Premium": f"{entry_premium:,.2f}",
        "Terminal Intrinsic Value": f"{terminal_intrinsic:,.2f}",
        "Warrant Return (on premium paid)": f"{terminal_intrinsic / entry_premium - 1:.2%}" if entry_premium > 0 else "n/a",
        "Realized Vol (ann.)": f"{vol:.2%}",
        "Max Drawdown": f"{max_drawdown(path):.2%}",
    })

    print("\n" + summary.to_string())

    print(f"\nPut Warrant — model value at inception")
    print(f"(QuantLib AnalyticEuropeanEngine, S=S0, K={STRIKE:.0%} of S0={S0:,.2f}, T={TENOR}y,")
    print(f"vol={entry_vol:.2%} {vol_source_desc},")
    print(f"SOFR {RISK_FREE_RATE:.2%}, dividend yield {dividend_yield:.2%}):")
    print(f"  Premium: {entry_premium:,.2f} ({TICKER} price units)")

    print(f"\nPut Warrant Greeks at inception (closed-form):")
    print(f"  Delta: {greeks['delta']:.3f}")
    print(f"  Vega:  {greeks['vega'] * 0.01:.4f}  (per 1% change in vol)")
    print(f"  Rho:   {greeks['rho'] * 0.01:.4f}  (per 1% change in rates)")
    print(f"  Theta: {greeks['theta']:.4f} per year / {greeks['theta'] / 365:.5f} per day")

    print(f"\nEffective gearing at inception (|Delta| x S0 / Premium):")
    gearing = abs(greeks["delta"]) * S0 / entry_premium if entry_premium > 0 else float("nan")
    print(f"  {gearing:.1f}x  (a 1% move in {underlying_name} moves the warrant's value by "
          f"approximately {gearing:.1f}% of the premium paid)")

    plot_path(path, S0, underlying_name, mtm_vol_term_structure=vol_term_structure, greeks=greeks, q=dividend_yield)
