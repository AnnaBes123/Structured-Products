import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql

plt.rcParams["font.family"] = "Arial"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Product terms ---
# Quoting convention used throughout this file: S = how many units of DEPOSIT
# currency 1 unit of ALT currency costs (e.g. DEPOSIT=USD, ALT=EUR -> S is the
# standard "EURUSD" quote, ~1.05-1.15). This is the same convention as
# Garman-Kohlhagen / QuantLib's BlackScholesMertonProcess expects (domestic
# = DEPOSIT, foreign = ALT), and it mirrors the Reverse Convertible's "S" 1:1 -
# ALT plays the role of "the stock", priced in DEPOSIT currency. See README for
# a full worked numeric example and why "below strike" can sound backwards
# if you're used to a USDJPY-style inverse quote.
STRIKE = 0.95                # short put strike on ALT, as a fraction of S0

DEPOSIT_CCY = "USD"              # the deposit / numeraire currency
ALT_CCY = "EUR"               # the currency the put is sold on

# Flat short-term rate ASSUMPTIONS (not fetched - same modeling simplification
# as RISK_FREE_RATE in the Reverse Convertible/FCN/Discount Certificate
# scripts elsewhere in this repo). Set these to whatever real short rates you
# want to model for the two currencies (e.g. SOFR/EFFR for USD, EUR short-term
# rate for EUR, SONIA for GBP, TONA for JPY) - there is no single generic
# per-currency short-rate feed wired up here, see README.
DEPOSIT_RATE = 0.04              # DEPOSIT currency short-term rate ("r")
ALT_RATE = 0.02                # ALT currency short-term rate ("q" - plays the
                               # exact role of a dividend yield in Garman-Kohlhagen)

# Same real, sourced issuer CDS spread used in the Reverse Convertible
# (Goldman Sachs 5y, 53.08 bps) - the deposit leg carries the deposit-taking
# bank's own credit risk, same reasoning as the RC's ZCB leg. Replace if
# modeling a different counterparty.
ISSUER_CDS_SPREAD = 0.005308

# DCIs are typically short-dated (days to a few months), unlike the 1-year
# convention used for RC/FCN elsewhere in this repo.
ENTRY_DATE = "2025-01-02"
TENOR = 0.25                  # 3 months

SPOT_SCENARIO_RANGE = np.arange(0.80, 1.21, 0.02)  # 80% to 120% of S0, in 2pt steps


def fetch_fx_path(deposit_ccy, alt_ccy, entry_date, years=TENOR):
    """S = DEPOSIT amount per 1 ALT unit (see module docstring). Yahoo Finance
    ticker convention: "{CCY1}{CCY2}=X" returns CCY2-per-1-CCY1, verified
    directly (EURUSD=X -> ~1.15 USD per EUR, USDJPY=X -> ~155 JPY per USD) -
    so ALT{DEPOSIT}=X gives exactly the DEPOSIT-per-ALT quote wanted here."""
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


# ---------------------------------------------------------------------------
# REPLICATION: DCI = Long DEPOSIT-currency balance (ZCB) - Short Put on ALT
#
# Mechanically identical to the Reverse Convertible (see that folder's
# README/code) with ALT playing the role of "the stock", DEPOSIT playing the
# role of cash, and Garman-Kohlhagen (Black-Scholes-Merton with the ALT
# short rate standing in for a dividend yield q - holding a foreign currency
# continuously earns its own risk-free rate, same mathematical role as a
# continuous dividend) pricing the option. QuantLib's BlackScholesMertonProcess
# already IS Garman-Kohlhagen once q is the foreign rate - no new engine
# needed.
#
# Scenario A (ALT strengthens or stays flat vs. strike, S_T >= STRIKE*S0):
#   put expires worthless - principal stays in DEPOSIT currency.
# Scenario B (ALT weakens past strike, S_T < STRIKE*S0):
#   bank exercises: principal is converted into ALT at the strike rate.
#   ALT received = Principal_DEPOSIT / STRIKE (same conversion-ratio logic as
#   the Reverse Convertible's Principal/Strike share delivery), worth
#   Principal_DEPOSIT * (S_T / STRIKE) back in DEPOSIT terms at the real spot.
# ---------------------------------------------------------------------------

def zcb_price_and_greeks(principal, T_remaining, funding_rate):
    discount = (1 + funding_rate) ** T_remaining
    price = principal / discount
    rho = -T_remaining * price / (1 + funding_rate)
    theta = price * np.log(1 + funding_rate)
    return {"price": price, "rho": rho, "theta": theta}


def _quantlib_process(S, r, q, sigma, today):
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    rf_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count, ql.Continuous, ql.Annual))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count, ql.Continuous, ql.Annual))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, day_count))
    return ql.BlackScholesMertonProcess(spot, div_ts, rf_ts, vol_ts)


def black_scholes_put(S, K, T, r, sigma, q=0.0):
    """Plain European put via QuantLib's AnalyticEuropeanEngine. With q set
    to the ALT currency's own short rate, this process IS Garman-Kohlhagen -
    the standard closed-form FX option model, not a hand-derived add-on."""
    if T <= 0:
        intrinsic = max(K - S, 0.0)
        delta = -1.0 if S < K else 0.0
        return {"price": intrinsic, "delta": delta, "vega": 0.0, "rho": 0.0, "theta": 0.0}

    today = ql.Date(1, 1, 2000)  # arbitrary fixed anchor - only T (via day count) matters
    ql.Settings.instance().evaluationDate = today
    process = _quantlib_process(S, r, q, sigma, today)

    days = max(int(round(T * 365.25)), 1)
    exercise = ql.EuropeanExercise(today + ql.Period(days, ql.Days))
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    return {
        "price": option.NPV(), "delta": option.delta(), "vega": option.vega(),
        "rho": option.rho(), "theta": option.theta(),
    }


def dci_mtm(S, S0, T_remaining, sigma, r=DEPOSIT_RATE, alt_rate=ALT_RATE, credit_spread=ISSUER_CDS_SPREAD):
    """Short put quantity is Principal/Strike (conversion ratio), NOT 1x -
    the entire principal converts to ALT below the strike, matching physical
    conversion of Principal/Strike units of ALT. Same derivation as the
    Reverse Convertible's put_quantity - see that folder's README."""
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
    """Participation Tracker: the terminal payoff FORMULA applied to each
    day's spot - full principal back (in DEPOSIT) if spot is at/above strike;
    below it, value = Principal * (S/Strike), exactly what conversion into
    Principal/Strike units of ALT is worth back in DEPOSIT terms. NOT what
    you'd actually receive if unwound today - see the MTM line for that."""
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


def realized_annualized_vol(path):
    log_returns = np.log(path / path.shift(1)).dropna()
    return log_returns.std() * np.sqrt(252)


def max_drawdown(path):
    running_max = path.cummax()
    drawdown = path / running_max - 1
    return drawdown.min()


def plot_path(path, S0, pair_label, sigma=None, greeks=None):
    """Left axis: the raw FX rate LEVEL (not a % return - an exchange rate
    isn't a return the way a stock price is, even though the math of a %
    change looks similar). Right axis: the note's own return figures (%
    of par) - the Participation Tracker and the MtM fair value share it,
    since both are genuinely "return"-like quantities, unlike the raw rate."""
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
                    linestyle="dashed", label="DCI Participation Tracker (% of Par)")
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
            linestyle="solid", label="DCI - Approximate MtM (% of Par, Garman-Kohlhagen)")
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
                 f"vs Approximate Mark-to-Market (MtM) Value of Dual Currency Investment")
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
        "DCI Return": f"{dci_return:.2%}",
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

    print(f"\nDCI fair value at inception")
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

    print(f"\nMTM Fair Value vs. Par at Inception:")
    print(f"  Price: {greeks['price']:,.4f}  vs. Par (S0): {S0:,.4f}  ({fair_value_pct_of_par:.2%} of par)")

    plot_path(path, S0, pair_label, sigma=entry_vol, greeks=greeks)
