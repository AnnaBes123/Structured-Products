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

STRIKE = 1.00
OUTPERFORMANCE_PARTICIPATION = 2.00
RISK_FREE_RATE = 0.04

ENTRY_DATE = "2025-01-02"
TENOR = 1

TICKER = "NVDA"
FRED_SERIES = "SP500"

VOL_TERM_STRUCTURE_TICKERS = {"^VIX": 30, "^VIX3M": 93, "^VIX6M": 182}
VIX_FRED_SERIES = "VIXCLS"


def fetch_index_path(entry_date, years=TENOR):
    start = pd.Timestamp(entry_date)
    end = start + pd.DateOffset(years=years)
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


def outperformance_certificate_running_return(S0, path):
    index_return = path / S0 - 1
    strike_level = STRIKE * S0
    above_strike = path >= strike_level

    running = index_return.copy()
    running[above_strike] = OUTPERFORMANCE_PARTICIPATION * index_return[above_strike]
    return running


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
    Priced directly here (not via black_scholes_call with K~0) since it has
    an exact closed form and K=0 is numerically awkward for a general
    option-pricing engine (d1/d2 blow up as K->0). See the README."""
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


def outperformance_certificate_mtm_price_series(S0, path, vol_term_structure, r=RISK_FREE_RATE, q=0.0):
    maturity_date = path.index[-1]
    prices = []
    for date, level in path.items():
        T_remaining = (maturity_date - date).days / 365.25
        vols_today = {tenor: series[date] for tenor, series in vol_term_structure.items()}
        sigma = interpolate_implied_vol(vols_today, max(T_remaining, 0.0))
        prices.append(outperformance_certificate_mtm(level, S0, T_remaining, sigma, r, q=q)["price"])
    return pd.Series(prices, index=path.index)


def plot_path(path, S0, underlying_name, mtm_vol_term_structure=None, greeks=None, q=0.0):
    index_return_pct = (path / S0 - 1) * 100
    certificate_return_pct = outperformance_certificate_running_return(S0, path) * 100

    _, ax = plt.subplots(figsize=(18, 8))
    ax.plot(index_return_pct.index, index_return_pct.values, color="firebrick", linewidth=1.5,
            label=underlying_name)
    ax.tick_params(axis="y", labelcolor="firebrick")
    ax.plot(certificate_return_pct.index, certificate_return_pct.values, color="indianred", linewidth=1.5, linestyle="dashed",
            label=f"Outperformance Certificate Redemption Payoff (Relative to Par) ({OUTPERFORMANCE_PARTICIPATION:.0%} above strike)")

    ax.grid(True, which="major", color="lightgrey", linewidth=0.6)
    ax.axhline(0, color="lightgrey", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return from Entry (%)", color="firebrick")
    lines, labels = ax.get_legend_handles_labels()
    ax.margins(x=0, y=0.05)

    if greeks is not None:
        S_T = float(path.iloc[-1])
        certificate_return = certificate_return_pct.iloc[-1] / 100
        fair_value_pct_of_par = greeks["price"] / S0
        vol = realized_annualized_vol(path)

        annotation_text = (
                    "Greeks at Inception:\n"
                    f"Δ (Delta): {greeks['delta']:.2f}\n"
                    f"ν (Vega):  {greeks['vega'] / S0 * 0.01:.4f} per 1% change in vol\n"
                    f"ρ (Rho):   {greeks['rho'] / S0 * 0.01:.4f} per 1% change in rates\n"
                    f"θ (Theta): {greeks['theta'] / S0:.4f} per year\n \n \n"
                    f"Participation Ratio: {OUTPERFORMANCE_PARTICIPATION:.0%}\n"
                    f"{underlying_name} Return: {S_T / S0 - 1:.2%}\n"
                    f"Outperformance Certificate Return: {certificate_return:.2%}\n"
                    f"Model Value vs. Par (%) at Inception: {fair_value_pct_of_par-1:.2%}\n"
                    f"Realized Vol (ann.): {vol:.2%}\n"
                    f"Max Drawdown: {max_drawdown(path):.2%}"
                )
        ax.text(1.06, 0.5, annotation_text, transform=ax.transAxes,
                fontsize=12, fontweight="light", color="black", va="center", ha="left")


    if mtm_vol_term_structure is not None:
        mtm_price = outperformance_certificate_mtm_price_series(S0, path, mtm_vol_term_structure, q=q)
        # Indexed to par (S0), not the certificate's own day-1 value - see README "Reading the charts".
        mtm_gain_over_par_pct = (mtm_price / S0 - 1) * 100

        ax2 = ax.twinx()
        mtm_line, = ax2.plot(
            mtm_gain_over_par_pct.index, mtm_gain_over_par_pct.values, color="darkred", linewidth=1.5,
            linestyle="solid", label="Outperformance Certificate — Model Value (% of Par, Black-Scholes)")
        ax2.set_ylabel("Model Value vs. Par (%)", color="darkred", rotation=270, labelpad=10)
        ax2.tick_params(axis="y", labelcolor="darkred")

        combined_min = min(index_return_pct.min(), certificate_return_pct.min(), mtm_gain_over_par_pct.min())
        combined_max = max(index_return_pct.max(), certificate_return_pct.max(), mtm_gain_over_par_pct.max())
        pad = (combined_max - combined_min) * 0.05
        ax.set_ylim(combined_min - pad, combined_max + pad)
        ax2.set_ylim(combined_min - pad, combined_max + pad)

        lines.append(mtm_line)
        labels.append(mtm_line.get_label())

    end_date = index_return_pct.index[-1]
    ax.set_title(f"{underlying_name} Price Return Path from {path.index[0].date()} to {end_date.date()} \n"
                 f"vs Model Value vs. Par — Outperformance Certificate "
                 f"({OUTPERFORMANCE_PARTICIPATION:.0%} Participation)")
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
    certificate_return = outperformance_certificate_running_return(S0, path).iloc[-1]

    dividend_yield = fetch_dividend_yield(TICKER)
    print(f"Dividend yield for {underlying_name} ({TICKER}): {dividend_yield:.2%} (flat, continuous - 0% if an index)")

    summary = pd.Series({
        "Entry Date": path.index[0].date(),
        "Maturity Date": path.index[-1].date(),
        "S0": f"{S0:,.2f}",
        "S_T": f"{S_T:,.2f}",
        f"{underlying_name} Return": f"{S_T / S0 - 1:.2%}",
        "Outperformance Certificate": f"{certificate_return:.2%}",
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

    greeks = outperformance_certificate_mtm(S0, S0, TENOR, entry_vol, q=dividend_yield)
    fair_value_pct_of_par = greeks["price"] / S0

    print(f"\nOutperformance Certificate — model value at inception")
    print(f"(QuantLib Black-Scholes-Merton, S=S0, T={TENOR}y, vol={entry_vol:.2%}")
    print(f"{vol_source_desc}, dividend")
    print(f"yield {dividend_yield:.2%}):")
    print(f"  {fair_value_pct_of_par:.2%} of par ({fair_value_pct_of_par - 1:+.2%} vs. par)")

    print(f"\nOutperformance Certificate Greeks at inception:")
    print(f"  Delta: {greeks['delta']:.2f}")
    print(f"  Vega:  {greeks['vega'] / S0 * 0.01:.4f}  (per 1% change in vol)")
    print(f"  Rho:   {greeks['rho'] / S0 * 0.01:.4f}  (per 1% change in rates)")
    print(f"  Theta: {greeks['theta'] / S0:.4f} per year / {greeks['theta'] / S0 / 365:.5f} per day")

    print(f"\nModel Value vs. Par at Inception:")
    print(f"  Price: {greeks['price']:,.2f}  vs. Par (S0): {S0:,.2f}  ({fair_value_pct_of_par:.2%} of par)")

    plot_path(path, S0, underlying_name, mtm_vol_term_structure=vol_term_structure, greeks=greeks, q=dividend_yield)
