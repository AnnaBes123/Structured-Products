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
    zcb_price_and_greeks,
    fetch_risk_free_rate,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Product terms ---
STRIKE = 1.00                  # long call strike, as a fraction of S0 (100% - at the money)
BARRIER = 1.10                 # up-and-out barrier (H) for the embedded call, H > STRIKE
REBATE = 4.0                   # cash paid immediately if the barrier is touched, in underlying
                                # currency (not a fraction of S0) - not economically important for
                                # MtM purposes here, so set by hand like every other manually-set
                                # term in this repo (STRIKE, BARRIER, etc.); left at 0 by default
GS_CDS_SPREAD = 0.002675       # Goldman Sachs 1y CDS, 26.75 bps (Investing.com) - issuer credit spread, ZCB leg only - tenor-matched to TENOR=1, not the 5y CDS an earlier version of this repo used

ENTRY_DATE = "2025-01-02"
TENOR = 1
RISK_FREE_RATE = fetch_risk_free_rate(ENTRY_DATE)  # 1Y Treasury CMT (FRED DGS1) as of ENTRY_DATE - real, historical; used for BOTH the ZCB leg and the option leg

TICKER = "MCD"
SPX_FRED_SERIES = "SP500"    # FRED fallback if yfinance fails - valid ONLY when TICKER
                             # is literally "^GSPC"; never used as a stand-in for a
                             # single-name stock's own price

SPX_VOL_TERM_STRUCTURE_TICKERS = {"^VIX": 30, "^VIX3M": 93, "^VIX6M": 182}  # SPX-only proxy;
                                                                            # only used when TICKER is an index (see below)
VIX_FRED_SERIES = "VIXCLS"

SPOT_SCENARIO_RANGE = np.arange(0.60, 1.41, 0.05)  # 60% to 140% of S0, in 5pt steps


def fetch_index_path(entry_date, years=TENOR):
    """Daily CLOSE series - the barrier is observed on close. See README
    "How the daily MTM tracks the barrier"."""
    start = pd.Timestamp(entry_date)
    end = start + pd.Timedelta(days=round(years * 365.25))
    path = fetch_daily_closes(TICKER, start, end, fred_series=SPX_FRED_SERIES if TICKER == "^GSPC" else None)
    if path.index[-1] < end - pd.Timedelta(days=10):
        raise RuntimeError(
            f"Requested window {start.date()} to {end.date()} extends past the last available "
            f"trading day ({path.index[-1].date()}) - this script models a COMPLETED historical "
            f"window, not a live in-progress note. Pick an ENTRY_DATE/TENOR combination that ends "
            f"on or before today."
        )
    return path


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
    """Plain (no barrier) European call via QuantLib's AnalyticEuropeanEngine
    (dividend yield q is a native input) - used only for the closed-form
    verification check (barrier pushed unreachable -> the up-and-out call
    should collapse to this, at rebate=0)."""
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


# REPLICATION: ZCB(Principal) + Long Up-and-Out Call (STRIKE=100% of S0,
# barrier > strike, optional rebate). Unlike the SHORT-option barrier
# products elsewhere (BRC, Bonus Certificate), the option leg here is LONG
# and ADDED to a fully protected bond, financed by capping how far
# participation can run (the barrier) rather than an outright cap level.
# Priced on a CRR binomial lattice - see README.md and ../../Yield/Barrier
# Reverse Convertible/MATHEMATICS.md for the full lattice mechanics.

def up_and_out_call_crr(S, K, H, T, r, sigma, steps, rebate=REBATE, q=0.0):
    """CRR up-and-out call. Assumes NOT already breached - caller prices
    this leg at 0 once the barrier has been touched (rebate already paid,
    not part of forward MTM)."""
    if S >= H:
        return rebate  # at/through the barrier right now: knocked out this instant, worth exactly the rebate
    if T <= 0:
        return max(S - K, 0.0)

    today = ql.Date(1, 1, 2000)
    ql.Settings.instance().evaluationDate = today
    process = _quantlib_process(S, r, q, sigma, today)

    days = max(int(round(T * 365)), 1)
    exercise = ql.EuropeanExercise(today + ql.Period(days, ql.Days))
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    option = ql.BarrierOption(ql.Barrier.UpOut, H, rebate, payoff, exercise)

    n = max(int(round(steps)), 2)  # QuantLib's binomial barrier engine errors below 2 steps
    option.setPricingEngine(ql.BinomialCRRBarrierEngine(process, n, n))
    return option.NPV()


def bullish_sharkfin_price(S, S0, T_remaining, sigma, breached, r=RISK_FREE_RATE,
                            credit_spread=GS_CDS_SPREAD, steps=None, q=0.0):
    """Full note price. Once breached, call contributes 0 (rebate already
    received) - note is worth exactly the ZCB, capital at 100%."""
    zcb = zcb_price_and_greeks(S0, T_remaining, r + credit_spread)

    if breached:
        call_price = 0.0
    else:
        K = STRIKE * S0
        H = BARRIER * S0
        n = steps if steps is not None else max(round(T_remaining * 252), 1)
        call_price = up_and_out_call_crr(S, K, H, T_remaining, r, sigma, n, REBATE, q)

    return zcb["price"] + call_price


def verify_against_closed_form(S0, T, sigma, r=RISK_FREE_RATE, q=0.0, steps=None):
    """Barrier pushed unreachable: CRR price must converge onto the
    plain vanilla call (no-barrier limit). See README."""
    K = STRIKE * S0
    n = steps if steps is not None else max(round(T * 252), 1)

    up_and_out_unreachable = up_and_out_call_crr(S0, K, H=50 * S0, T=T, r=r, sigma=sigma, steps=n, rebate=0.0, q=q)
    vanilla_call = black_scholes_call(S0, K, T, r, sigma, q)["price"]
    rel_diff = abs(up_and_out_unreachable - vanilla_call) / vanilla_call

    return {"up_and_out_unreachable": up_and_out_unreachable, "vanilla_call": vanilla_call, "rel_diff": rel_diff}


def bullish_sharkfin_running_return(S0, path):
    """Payoff If Settled Today (Relative to Par) - see README "Reading the
    charts" and "How the daily MTM tracks the barrier"."""
    strike_level = STRIKE * S0
    barrier_level = BARRIER * S0
    breached_so_far = path.cummax() >= barrier_level

    call_payoff = np.maximum(path.values - strike_level, 0.0)
    call_payoff = np.where(breached_so_far, 0.0, call_payoff)
    return pd.Series(call_payoff, index=path.index) / S0


def bullish_sharkfin_mtm_price_series(S0, path, vol_term_structure, r=RISK_FREE_RATE,
                                       credit_spread=GS_CDS_SPREAD, q=0.0):
    maturity_date = path.index[-1]
    barrier_level = BARRIER * S0
    breached_so_far = path.cummax() >= barrier_level

    n_dates = len(path)
    prices = []
    for i, (date, level) in enumerate(path.items()):
        T_remaining = (maturity_date - date).days / 365.25
        steps_remaining = max(n_dates - 1 - i, 1)  # one CRR barrier check per remaining trading day in the actual path
        vols_today = {tenor: series[date] for tenor, series in vol_term_structure.items()}
        sigma = interpolate_implied_vol(vols_today, max(T_remaining, 0.0))
        prices.append(
            bullish_sharkfin_price(level, S0, T_remaining, sigma, breached_so_far[date], r, credit_spread,
                                    steps=steps_remaining, q=q)
        )
    return pd.Series(prices, index=path.index)


def finite_difference_greeks_at(S, S0, T, sigma, breached, r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD,
                                 steps=None, q=0.0):
    """Central finite differences on the full note price - wider bumps
    than the closed-form products. See README "A numerical note on the
    Greeks: lattice sawtooth noise" for the bump-size scan that motivated
    these (2% of spot / 2 vol points)."""
    bump_S, bump_sigma, bump_r, bump_T = S0 * 0.02, 0.02, 0.0001, 21 / 365
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

    price_less_T = bullish_sharkfin_price(S, S0, max(T - bump_T, 0.0), sigma, breached, r, credit_spread, steps=n, q=q)
    theta = (price_less_T - price) / bump_T

    return {"price": price, "delta": delta, "vega": vega, "rho": rho, "theta": theta}


def finite_difference_greeks(S0, T, sigma, r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD, q=0.0):
    return finite_difference_greeks_at(S0, S0, T, sigma, False, r, credit_spread, q=q)


def plot_path(path, S0, underlying_name, mtm_vol_term_structure=None, greeks=None, q=0.0):
    index_return_pct = (path / S0 - 1) * 100
    sharkfin_return_pct = bullish_sharkfin_running_return(S0, path) * 100

    _, ax = plt.subplots(figsize=(18, 8))
    ax.plot(index_return_pct.index, index_return_pct.values, color="firebrick", linewidth=1.5,
            label=underlying_name)
    ax.tick_params(axis="y", labelcolor="firebrick")
    ax.plot(sharkfin_return_pct.index, sharkfin_return_pct.values, color="indianred", linewidth=1.5,
            linestyle="dashed", label="Bullish Sharkfin Payoff If Settled Today (Relative to Par)")

    ax.grid(True, which="major", color="lightgrey", linewidth=0.6)
    ax.axhline(0, color="lightgrey", linewidth=0.8)
    strike_pct = (STRIKE - 1) * 100
    ax.axhline(strike_pct, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.text(0.01, strike_pct, f"Strike: {strike_pct:.1f}%", transform=ax.get_yaxis_transform(),
            color="dodgerblue", fontsize=9, va="bottom", ha="left")

    barrier_pct = (BARRIER - 1) * 100
    ax.axhline(barrier_pct, color="darkgreen", linewidth=0.8, linestyle="dotted")
    ax.text(0.01, barrier_pct, f"Knock-Out Barrier: {barrier_pct:.1f}%", transform=ax.get_yaxis_transform(),
            color="darkgreen", fontsize=9, va="bottom", ha="left")

    breached_level = BARRIER * S0
    breached_dates = path.index[path.cummax() >= breached_level]
    if len(breached_dates) > 0:
        breach_date = breached_dates[0]
        ax.axvline(breach_date, color="darkgreen", linewidth=1.2, linestyle="dashed")
        ax.text(breach_date, 0.98, "  Barrier Knocked Out", transform=ax.get_xaxis_transform(),
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
            f"ρ (Rho):   {greeks['rho'] / S0 * 0.01:.4f} per 1% change in the 1Y Treasury rate\n"
            f"θ (Theta): {greeks['theta'] / S0:.4f} per year"
        )
        ax.text(1.06, 0.5, greeks_text, transform=ax.transAxes,
                fontsize=12, fontweight="light", color="black", va="center", ha="left")

    if mtm_vol_term_structure is not None:
        mtm_price = bullish_sharkfin_mtm_price_series(S0, path, mtm_vol_term_structure, q=q)
        mtm_gain_over_par_pct = (mtm_price / S0 - 1) * 100

        ax2 = ax.twinx()
        mtm_line, = ax2.plot(
            mtm_gain_over_par_pct.index, mtm_gain_over_par_pct.values, color="darkred", linewidth=1.5,
            linestyle="solid", label="Bullish Sharkfin — Model Value (% of Par)")
        ax2.set_ylabel("Model Value vs. Par (%)", color="darkred", rotation=270, labelpad=10)
        ax2.tick_params(axis="y", labelcolor="darkred")

        combined_min = min(index_return_pct.min(), sharkfin_return_pct.min(), mtm_gain_over_par_pct.min(), strike_pct)
        combined_max = max(index_return_pct.max(), sharkfin_return_pct.max(), mtm_gain_over_par_pct.max(), barrier_pct)
        pad = (combined_max - combined_min) * 0.05
        ax.set_ylim(combined_min - pad, combined_max + pad)
        ax2.set_ylim(combined_min - pad, combined_max + pad)

        lines.append(mtm_line)
        labels.append(mtm_line.get_label())

    end_date = index_return_pct.index[-1]
    ax.set_title(f"{underlying_name} Price Return Path from {path.index[0].date()} to {end_date.date()} \n"
                 f"vs Model Value vs. Par — Bullish Sharkfin")
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
    sharkfin_return = bullish_sharkfin_running_return(S0, path).iloc[-1]
    barrier_touched = bool((path.cummax() >= BARRIER * S0).any())

    dividend_yield = fetch_dividend_yield(TICKER)
    print(f"Dividend yield for {underlying_name} ({TICKER}): {dividend_yield:.2%} (flat, continuous - 0% if an index)")

    summary = pd.Series({
        "Entry Date": path.index[0].date(),
        "Maturity Date": path.index[-1].date(),
        "S0": f"{S0:,.2f}",
        "S_T": f"{S_T:,.2f}",
        "Strike Level": f"{STRIKE * S0:,.2f}",
        "Barrier Level": f"{BARRIER * S0:,.2f}",
        "Barrier Touched": "Yes" if barrier_touched else "No",
        f"{underlying_name} Return": f"{S_T / S0 - 1:.2%}",
        "Bullish Sharkfin — Redemption Payoff (vs. Par)": f"{sharkfin_return:.2%}",
        "Realized Vol (ann.)": f"{vol:.2%}",
        "Max Drawdown": f"{max_drawdown(path):.2%}",
    })

    print("\n" + summary.to_string())

    if TICKER.startswith("^"):
        print(f"\nFetching SPX implied vol term structure (VIX/VIX3M/VIX6M) for the same window...")
        raw_term_structure = fetch_spx_vol_term_structure(ENTRY_DATE)
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

    print(f"\nVerifying the QuantLib CRR wiring (barrier pushed unreachable should collapse to the")
    print(f"plain vanilla call struck at the money):")
    check = verify_against_closed_form(S0, TENOR, entry_vol, q=dividend_yield)
    print(f"  Up-and-out call (H unreachable): {check['up_and_out_unreachable']:,.2f}")
    print(f"  Vanilla call (closed form):      {check['vanilla_call']:,.2f}")
    print(f"  Relative difference:             {check['rel_diff']:.3%}  "
          f"({'PASS' if check['rel_diff'] < 0.01 else 'FAIL - investigate before trusting results'})")

    greeks = finite_difference_greeks(S0, TENOR, entry_vol, q=dividend_yield)
    fair_value_pct_of_par = greeks["price"] / S0

    print(f"\nBullish Sharkfin — model value at inception")
    print(f"(ZCB discounted at the 1Y Treasury rate {RISK_FREE_RATE:.2%} + Goldman Sachs CDS {GS_CDS_SPREAD:.2%}, long")
    print(f"up-and-out call priced at the 1Y Treasury rate alone via QuantLib's CRR binomial lattice (barrier checked")
    print(f"once per trading day, rebate={REBATE}, dividend yield {dividend_yield:.2%}), T={TENOR}y,")
    print(f"vol={entry_vol:.2%} {vol_source_desc},")
    print(f"strike={STRIKE:.0%}, barrier={BARRIER:.0%} of S0={S0:,.2f}):")
    print(f"  {fair_value_pct_of_par:.2%} of par ({fair_value_pct_of_par - 1:+.2%} vs. par)")

    print(f"\nBullish Sharkfin Greeks at inception (finite-difference):")
    print(f"  Delta: {greeks['delta']:.2f}")
    print(f"  Vega:  {greeks['vega'] / S0 * 0.01:.4f}  (per 1% change in vol)")
    print(f"  Rho:   {greeks['rho'] / S0 * 0.01:.4f}  (per 1% change in the 1Y Treasury rate)")
    print(f"  Theta: {greeks['theta'] / S0:.4f} per year / {greeks['theta'] / S0 / 365:.5f} per day")

    print(f"\nModel Value vs. Par at Inception:")
    print(f"  Price: {greeks['price']:,.2f}  vs. Par (S0): {S0:,.2f}  ({fair_value_pct_of_par:.2%} of par)")

    plot_path(path, S0, underlying_name, mtm_vol_term_structure=vol_term_structure, greeks=greeks, q=dividend_yield)
