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
    max_drawdown,
    _quantlib_process,
    zcb_price_and_greeks,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Product terms --- Quoting convention: S = DEPOSIT units per 1 ALT unit
# (e.g. DEPOSIT=USD, ALT=EUR -> S is the standard EURUSD quote). See README
# for the full worked example and why "below strike" can look backwards for
# an inverse-quoted pair like USDJPY.
STRIKE = 0.95                # short put strike on ALT, as a fraction of S0

DEPOSIT_CCY = "USD"              # the deposit / numeraire currency
ALT_CCY = "EUR"               # the currency the put is sold on

DEPOSIT_RATE = 0.04              # DEPOSIT currency short-term rate ("r") - flat assumption, see README
ALT_RATE = 0.02                # ALT currency short-term rate ("q" in Garman-Kohlhagen)

ISSUER_CDS_SPREAD = 0.005308  # Goldman Sachs 5y CDS - deposit-taking bank's credit risk

ENTRY_DATE = "2025-01-02"
TENOR = 0.25                  # 3 months - DCIs are typically short-dated

SPOT_SCENARIO_RANGE = np.arange(0.80, 1.21, 0.02)  # 80% to 120% of S0, in 2pt steps


def fetch_fx_path(deposit_ccy, alt_ccy, entry_date, years=TENOR):
    """S = DEPOSIT per 1 ALT unit. Yahoo's "{CCY1}{CCY2}=X" convention
    returns CCY2-per-1-CCY1, so ALT{DEPOSIT}=X gives that directly."""
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
    if series.index[-1] < end - pd.Timedelta(days=10):
        raise RuntimeError(
            f"Requested window {start.date()} to {end.date()} extends past the last available "
            f"trading day ({series.index[-1].date()}) - this script models a COMPLETED historical "
            f"window, not a live in-progress note. Pick an ENTRY_DATE/TENOR combination that ends "
            f"on or before today."
        )
    return series


# REPLICATION: Long DEPOSIT balance (ZCB) - Short Put on ALT, mechanically
# identical to the Reverse Convertible with ALT as "the stock" and
# Garman-Kohlhagen pricing the put. See README.md and MATHEMATICS.md.

def black_scholes_put(S, K, T, r, sigma, q=0.0):
    """With q = ALT_RATE this process IS Garman-Kohlhagen - see
    MATHEMATICS.md section 1."""
    if T <= 0:
        intrinsic = max(K - S, 0.0)
        delta = -1.0 if S < K else 0.0
        return {"price": intrinsic, "delta": delta, "vega": 0.0, "rho": 0.0, "theta": 0.0}

    today = ql.Date(1, 1, 2000)  # arbitrary fixed anchor - only T (via day count) matters
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
    """Full price. Put quantity is 1/Strike - see ../Fixed Coupon
    Note/MATHEMATICS.md section 3 for the conversion-ratio derivation."""
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


def dci_running_return(S0, path):
    """Redemption Payoff (Relative to Par) - see README "Reading the
    charts" for what this tracker does and doesn't represent."""
    strike_level = STRIKE * S0
    below_strike = path < strike_level
    running = pd.Series(0.0, index=path.index)
    running[below_strike] = path[below_strike] / strike_level - 1.0
    return running


def dci_mtm_price_series(S0, path, sigma, r=DEPOSIT_RATE, alt_rate=ALT_RATE):
    maturity_date = path.index[-1]
    prices = []
    for date, level in path.items():
        T_remaining = (maturity_date - date).days / 365.25
        prices.append(dci_mtm(level, S0, T_remaining, sigma, r, alt_rate)["price"])
    return pd.Series(prices, index=path.index)


def plot_path(path, S0, pair_label, sigma=None, greeks=None):
    """Left axis: raw FX rate LEVEL, not a % return. Right axis: the
    note's own % of par figures. See README "Reading the chart"."""
    strike_level = STRIKE * S0
    dci_return_pct = dci_running_return(S0, path) * 100
    S_T = float(path.iloc[-1])
    exercised = S_T < strike_level

    _, ax = plt.subplots(figsize=(18, 8))
    ax.plot(path.index, path.values, color="firebrick", linewidth=1.5,
            label=f"{pair_label} spot (level)")
    ax.tick_params(axis="y", labelcolor="firebrick")

    # European, not American/barrier: only S_T (the terminal marker below)
    # vs. the strike decides the outcome - nothing that happens mid-path
    # matters. The strike is drawn faint and full-width purely as a visual
    # reference level, and solid/bold only over the last ~8% of the window
    # (where the comparison actually happens) so it doesn't read as a
    # continuously-monitored barrier.
    ax.axhline(strike_level, color="dodgerblue", linewidth=0.5, linestyle="dotted", alpha=0.5)
    x0, x1 = path.index[0], path.index[-1]
    emphasis_start = x0 + (x1 - x0) * 0.92
    ax.hlines(strike_level, emphasis_start, x1, color="dodgerblue", linewidth=2.2, linestyle="solid")
    ax.text(0.01, strike_level, f"Put Strike: {strike_level:.4f}  (compared to spot ONLY at maturity - "
            f"European, not a continuous barrier)", transform=ax.get_yaxis_transform(),
            color="dodgerblue", fontsize=9, va="bottom", ha="left")
    ax.scatter([x1], [S_T], color=("firebrick" if exercised else "seagreen"), s=70, zorder=5,
               label=f"S_T = {S_T:.4f} ({'below strike - exercised' if exercised else 'at/above strike - worthless'})")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{pair_label} Spot Rate ({DEPOSIT_CCY} per 1 {ALT_CCY})", color="firebrick")
    ax.margins(x=0, y=0.05)

    ax2 = ax.twinx()
    l1, = ax2.plot(dci_return_pct.index, dci_return_pct.values, color="indianred", linewidth=1.5,
                    linestyle="dashed", label="DCI Redemption Payoff (Relative to Par) (% of Par)")
    ax_lines, ax_labels = ax.get_legend_handles_labels()
    lines = ax_lines + [l1]
    labels = ax_labels + [l1.get_label()]

    if greeks is not None:
        greeks_text = (
            "Greeks at Inception:\n"
            f"Δ (Delta): {greeks['delta']:.2f}\n"
            f"ν (Vega):  {greeks['vega'] / S0 * 0.01:.4f} per 1% change in vol\n"
            f"ρ (Rho):   {greeks['rho'] / S0 * 0.01:.4f} per 1% change in {DEPOSIT_CCY} rate\n"
            f"θ (Theta): {greeks['theta'] / S0:.4f} per year"
        )
        ax.text(1.08, 0.5, greeks_text, transform=ax.transAxes,
                fontsize=12, fontweight="light", color="black", va="center", ha="left")

    if sigma is not None:
        mtm_price = dci_mtm_price_series(S0, path, sigma)
        mtm_gain_over_par_pct = (mtm_price / S0 - 1) * 100

        mtm_line, = ax2.plot(
            mtm_gain_over_par_pct.index, mtm_gain_over_par_pct.values, color="darkred", linewidth=1.5,
            linestyle="solid", label="DCI — Model Value (% of Par, Garman-Kohlhagen)")
        lines.append(mtm_line)
        labels.append(mtm_line.get_label())

        combined_min = min(dci_return_pct.min(), mtm_gain_over_par_pct.min())
        combined_max = max(dci_return_pct.max(), mtm_gain_over_par_pct.max())
        pad = (combined_max - combined_min) * 0.05 if combined_max > combined_min else 1.0
        ax2.set_ylim(combined_min - pad, combined_max + pad)

    ax2.set_ylabel("Note Return (% of Par)", color="darkred", rotation=270, labelpad=15)
    ax2.tick_params(axis="y", labelcolor="darkred")
    ax2.axhline(0, color="lightgrey", linewidth=0.6)

    end_date = path.index[-1]
    ax.set_title(f"{pair_label} Spot Path from {path.index[0].date()} to {end_date.date()} \n"
                 f"vs Model Value vs. Par — Dual Currency Investment")
    ax.legend(lines, labels, loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    pair_label = f"{ALT_CCY}{DEPOSIT_CCY}"
    print(f"Fetching {pair_label} spot path from {ENTRY_DATE} (entry) to +{TENOR}y (maturity)...")
    print(f"(S = {DEPOSIT_CCY} per 1 {ALT_CCY} - the standard '{pair_label}' quote)")
    path = fetch_fx_path(DEPOSIT_CCY, ALT_CCY, ENTRY_DATE)

    S0 = float(path.iloc[0])
    S_T = float(path.iloc[-1])
    vol = realized_annualized_vol(path)
    dci_return = dci_running_return(S0, path).iloc[-1]

    summary = pd.Series({
        "Entry Date": path.index[0].date(),
        "Maturity Date": path.index[-1].date(),
        "S0": f"{S0:.4f}",
        "S_T": f"{S_T:.4f}",
        "Strike Level": f"{STRIKE * S0:.4f}",
        f"{pair_label} Return": f"{S_T / S0 - 1:.2%}",
        "DCI — Redemption Payoff (vs. Par)": f"{dci_return:.2%}",
        "Realized Vol (ann.)": f"{vol:.2%}",
        "Max Drawdown": f"{max_drawdown(path):.2%}",
    })

    print("\n" + summary.to_string())

    # No generic implied-vol source exists for an arbitrary FX pair the way
    # VIX serves SPX in the Reverse Convertible/Discount Certificate scripts,
    # so realized historical volatility of the pair itself is used as the
    # pricing vol input - a real, computed number, not invented, but a
    # genuine simplification vs. a real FX implied-vol surface (see README).
    entry_vol = vol

    greeks = dci_mtm(S0, S0, TENOR, entry_vol)
    fair_value_pct_of_par = greeks["price"] / S0

    print(f"\nDCI — model value at inception")
    print(f"({DEPOSIT_CCY} deposit discounted at {DEPOSIT_CCY} rate {DEPOSIT_RATE:.2%} + issuer CDS "
          f"{ISSUER_CDS_SPREAD:.2%},")
    print(f"short ALT put priced via QuantLib/Garman-Kohlhagen at {DEPOSIT_CCY} rate {DEPOSIT_RATE:.2%} / "
          f"{ALT_CCY} rate {ALT_RATE:.2%}, T={TENOR}y,")
    print(f"vol={entry_vol:.2%} realized historical vol of {pair_label}):")
    print(f"  {fair_value_pct_of_par:.2%} of par ({fair_value_pct_of_par - 1:+.2%} vs. par)")

    print(f"\nDCI Greeks at inception:")
    print(f"  Delta: {greeks['delta']:.2f}")
    print(f"  Vega:  {greeks['vega'] / S0 * 0.01:.4f}  (per 1% change in vol)")
    print(f"  Rho:   {greeks['rho'] / S0 * 0.01:.4f}  (per 1% change in {DEPOSIT_CCY} rate)")
    print(f"  Theta: {greeks['theta'] / S0:.4f} per year / {greeks['theta'] / S0 / 365:.5f} per day")

    print(f"\nModel Value vs. Par at Inception:")
    print(f"  Price: {greeks['price']:,.4f}  vs. Par (S0): {S0:,.4f}  ({fair_value_pct_of_par:.2%} of par)")

    plot_path(path, S0, pair_label, sigma=entry_vol, greeks=greeks)
