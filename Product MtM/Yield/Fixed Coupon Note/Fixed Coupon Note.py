import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql

plt.rcParams["font.family"] = "Arial"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Product terms ---
STRIKE = 0.90                # short put strike, as a fraction of S0 (90% - typical FCN strike)
RISK_FREE_RATE = 0.04        # SOFR proxy - used for BOTH the ZCB leg and the option leg
GS_CDS_SPREAD = 0.005308     # Goldman Sachs 5y CDS, 53.08 bps - issuer credit spread, ZCB leg only

ENTRY_DATE = "2025-01-02"
TENOR = 1

TICKER = "^GSPC"
FRED_SERIES = "SP500"

VOL_TERM_STRUCTURE_TICKERS = {"^VIX": 30, "^VIX3M": 93, "^VIX6M": 182}
VIX_FRED_SERIES = "VIXCLS"

SPOT_SCENARIO_RANGE = np.arange(0.60, 1.41, 0.05)  # 60% to 140% of S0, in 5pt steps


def fetch_daily_closes(ticker, start, end, fred_series=None):
    series = None

    try:
        import yfinance as yf
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if data is not None and not data.empty:
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            series = close.dropna()
    except Exception as exc:
        print(f"  yfinance failed for {ticker} ({exc})" + (" ; trying FRED fallback..." if fred_series else ""))

    if (series is None or series.empty) and fred_series:
        try:
            import pandas_datareader.data as web
            data = web.DataReader(fred_series, "fred", start, end)
            series = data[fred_series].dropna()
        except Exception as exc:
            raise RuntimeError(f"Could not fetch {ticker} data from either yfinance or FRED: {exc}")

    if series is None or series.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    return series


def fetch_dividend_yield(ticker):
    """Trailing dividend yield, used as a flat continuous yield q. Indices
    ("^" tickers) are treated as paying none. See the README in this
    folder for the full rationale and field-selection notes."""
    if ticker.startswith("^"):
        return 0.0
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        yield_ = info.get("trailingAnnualDividendYield")
        if yield_ is None:
            rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate")
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            yield_ = (rate / price) if (rate and price) else 0.0
        return float(yield_)
    except Exception as exc:
        print(f"  Could not fetch dividend yield for {ticker} ({exc}); assuming q=0")
        return 0.0


def fetch_underlying_name(ticker):
    """Human-readable underlying name for chart/print labels, falling back
    to the raw ticker symbol if yfinance metadata is unavailable."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker


def fetch_index_path(entry_date, years=TENOR):
    start = pd.Timestamp(entry_date)
    end = start + pd.Timedelta(days=round(years * 365.25))  # DateOffset(years=...) rejects fractional years
    return fetch_daily_closes(TICKER, start, end, fred_series=FRED_SERIES)


def fetch_vol_term_structure(entry_date, years=TENOR):
    start = pd.Timestamp(entry_date)
    end = start + pd.Timedelta(days=round(years * 365.25))  # DateOffset(years=...) rejects fractional years

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


def interpolate_implied_vol(vols_by_tenor, T_years):
    points = sorted(vols_by_tenor.items())

    if T_years <= points[0][0]:
        return points[0][1]
    if T_years >= points[-1][0]:
        return points[-1][1]

    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if t0 <= T_years <= t1:
            var0, var1 = v0 ** 2 * t0, v1 ** 2 * t1
            var_T = var0 + (var1 - var0) * (T_years - t0) / (t1 - t0)
            return np.sqrt(var_T / T_years)

    return points[-1][1]


# ---------------------------------------------------------------------------
# REPLICATION: Fixed Coupon Note = Long Zero-Coupon Bond - Short Put
#
# The ZCB pays Principal at maturity; its price today is discounted at the
# issuer's own funding rate (risk-free + credit spread), since the investor
# is exposed to the ISSUER's default risk on that leg, not the market's.
# The put is priced via QuantLib (AnalyticEuropeanEngine, closed-form
# Black-Scholes-Merton) at the risk-free rate alone - a bank prices/hedges
# the option on standard derivative terms, not its own credit curve. Being
# short the put is what finances the note's enhanced yield ("coupon") over
# a plain bond.
#
# If the underlying is a dividend-paying stock rather than an index, the
# put leg needs the dividend yield q to be theoretically consistent -
# fetch_dividend_yield pulls a flat, continuous yield estimate (0 for an
# index) and threads it through QuantLib's process as a native input.
# ---------------------------------------------------------------------------

def zcb_price_and_greeks(principal, T_remaining, funding_rate):
    """
    Approximate ZCB pricing: principal / (1 + funding_rate)^T. Returns
    price plus its sensitivity to the funding rate ("rho") and to time
    passing ("theta") - both closed-form, verified against finite
    differences.
    """
    discount = (1 + funding_rate) ** T_remaining
    price = principal / discount
    rho = -T_remaining * price / (1 + funding_rate)          # d(price)/d(funding_rate)
    theta = price * np.log(1 + funding_rate)                 # value gained per year as T_remaining shrinks
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
    """Plain European put via QuantLib's AnalyticEuropeanEngine (dividend
    yield q is a native input)."""
    if T <= 0:
        intrinsic = max(K - S, 0.0)
        delta = -1.0 if S < K else 0.0
        return {"price": intrinsic, "delta": delta, "vega": 0.0, "rho": 0.0, "theta": 0.0}

    today = ql.Date(1, 1, 2000)
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


def fixed_coupon_note_mtm(S, S0, T_remaining, sigma, r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD, q=0.0):
    """
    The short put quantity is Principal/Strike, NOT 1x - this is the
    note's conversion ratio (shares per note if physically settled). A
    note struck at 90% of S0 delivers (Principal/0.9*S0) shares below
    the strike, so its value there is Principal*(S_T/Strike), a payoff
    that scales through the origin - not "principal minus a flat cash
    shortfall" (quantity=1 would give that wrong, shallower payoff).

    `q` is the underlying's dividend yield (0 for an index) - priced at
    SOFR alone like the rest of the put leg.
    """
    zcb = zcb_price_and_greeks(S0, T_remaining, r + credit_spread)
    put = black_scholes_put(S, STRIKE * S0, T_remaining, r, sigma, q)
    put_quantity = 1.0 / STRIKE

    return {
        "price": zcb["price"] - put_quantity * put["price"],
        "delta": -put_quantity * put["delta"],
        "vega": -put_quantity * put["vega"],
        "rho": zcb["rho"] - put_quantity * put["rho"],
        "theta": zcb["theta"] - put_quantity * put["theta"],
    }


def fixed_coupon_note_running_return(S0, path):
    """
    Participation Tracker: the terminal payoff FORMULA applied to each
    day's spot - full principal back if spot is at or above the strike;
    below it, value = Principal * (S/Strike), exactly what physical
    delivery of Principal/Strike shares is worth, matching the
    put_quantity = 1/STRIKE used in fixed_coupon_note_mtm above. This is
    NOT what you'd actually receive if the note were sold or unwound
    today - it ignores all remaining time value in the still-live put.
    See the MTM line for the actual fair-value estimate.
    """
    strike_level = STRIKE * S0
    below_strike = path < strike_level
    running = pd.Series(0.0, index=path.index)
    running[below_strike] = path[below_strike] / strike_level - 1.0
    return running


def fixed_coupon_note_mtm_price_series(S0, path, vol_term_structure, r=RISK_FREE_RATE, q=0.0):
    maturity_date = path.index[-1]
    prices = []
    for date, level in path.items():
        T_remaining = (maturity_date - date).days / 365.25
        vols_today = {tenor: series[date] for tenor, series in vol_term_structure.items()}
        sigma = interpolate_implied_vol(vols_today, max(T_remaining, 0.0))
        prices.append(fixed_coupon_note_mtm(level, S0, T_remaining, sigma, r, q=q)["price"])
    return pd.Series(prices, index=path.index)


def realized_annualized_vol(path):
    log_returns = np.log(path / path.shift(1)).dropna()
    return log_returns.std() * np.sqrt(252)


def max_drawdown(path):
    running_max = path.cummax()
    drawdown = path / running_max - 1
    return drawdown.min()


def plot_path(path, S0, underlying_name, mtm_vol_term_structure=None, greeks=None, q=0.0):
    index_return_pct = (path / S0 - 1) * 100
    note_return_pct = fixed_coupon_note_running_return(S0, path) * 100

    _, ax = plt.subplots(figsize=(18, 8))
    ax.plot(index_return_pct.index, index_return_pct.values, color="firebrick", linewidth=1.5,
            label=underlying_name)
    ax.tick_params(axis="y", labelcolor="firebrick")
    ax.plot(note_return_pct.index, note_return_pct.values, color="indianred", linewidth=1.5,
            linestyle="dashed", label="Fixed Coupon Note Participation Tracker")

    ax.grid(True, which="major", color="lightgrey", linewidth=0.6)
    ax.axhline(0, color="lightgrey", linewidth=0.8)
    strike_pct = (STRIKE - 1) * 100
    ax.axhline(strike_pct, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.text(0.01, strike_pct, f"Put Strike: {strike_pct:.1f}%", transform=ax.get_yaxis_transform(),
            color="dodgerblue", fontsize=9, va="bottom", ha="left")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return from Entry (%)", color="firebrick")
    lines, labels = ax.get_legend_handles_labels()
    ax.margins(x=0, y=0.05)

    if greeks is not None:
        greeks_text = (
            "Greeks at Inception:\n"
            f"Δ (Delta): {greeks['delta']:.2f}\n"
            f"ν (Vega):  {greeks['vega'] / S0 * 0.01:.4f} per 1% change in vol\n"
            f"ρ (Rho):   {greeks['rho'] / S0 * 0.01:.4f} per 1% change in SOFR\n"
            f"θ (Theta): {greeks['theta'] / S0:.4f} per year"
        )
        ax.text(1.06, 0.5, greeks_text, transform=ax.transAxes,
                fontsize=12, fontweight="light", color="black", va="center", ha="left")

    if mtm_vol_term_structure is not None:
        mtm_price = fixed_coupon_note_mtm_price_series(S0, path, mtm_vol_term_structure, q=q)
        mtm_gain_over_par_pct = (mtm_price / S0 - 1) * 100

        ax2 = ax.twinx()
        mtm_line, = ax2.plot(
            mtm_gain_over_par_pct.index, mtm_gain_over_par_pct.values, color="darkred", linewidth=1.5,
            linestyle="solid", label="Fixed Coupon Note - Approximate MtM (% of Par, Black-Scholes)")
        ax2.set_ylabel("MtM Fair Value vs. Par (%)", color="darkred", rotation=270, labelpad=10)
        ax2.tick_params(axis="y", labelcolor="darkred")

        combined_min = min(index_return_pct.min(), note_return_pct.min(), mtm_gain_over_par_pct.min(), strike_pct)
        combined_max = max(index_return_pct.max(), note_return_pct.max(), mtm_gain_over_par_pct.max())
        pad = (combined_max - combined_min) * 0.05
        ax.set_ylim(combined_min - pad, combined_max + pad)
        ax2.set_ylim(combined_min - pad, combined_max + pad)

        lines.append(mtm_line)
        labels.append(mtm_line.get_label())

    end_date = index_return_pct.index[-1]
    ax.set_title(f"{underlying_name} Price Return Path from {path.index[0].date()} to {end_date.date()} \n"
                 f"vs Approximate Mark-to-Market (MtM) Value of Fixed Coupon Note")
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
    note_return = fixed_coupon_note_running_return(S0, path).iloc[-1]

    dividend_yield = fetch_dividend_yield(TICKER)
    print(f"Dividend yield for {underlying_name} ({TICKER}): {dividend_yield:.2%} (flat, continuous - 0% if an index)")

    summary = pd.Series({
        "Entry Date": path.index[0].date(),
        "Maturity Date": path.index[-1].date(),
        "S0": f"{S0:,.2f}",
        "S_T": f"{S_T:,.2f}",
        "Strike Level": f"{STRIKE * S0:,.2f}",
        f"{underlying_name} Return": f"{S_T / S0 - 1:.2%}",
        "Fixed Coupon Note Return": f"{note_return:.2%}",
        "Realized Vol (ann.)": f"{vol:.2%}",
        "Max Drawdown": f"{max_drawdown(path):.2%}",
    })

    print("\n" + summary.to_string())

    print(f"\nFetching SPX implied vol term structure (VIX/VIX3M/VIX6M) for the same window...")
    raw_term_structure = fetch_vol_term_structure(ENTRY_DATE)
    vol_term_structure = {
        tenor: series.reindex(path.index).ffill().bfill()
        for tenor, series in raw_term_structure.items()
    }
    vols_at_entry = {tenor: series.iloc[0] for tenor, series in vol_term_structure.items()}
    entry_vol = interpolate_implied_vol(vols_at_entry, TENOR)

    greeks = fixed_coupon_note_mtm(S0, S0, TENOR, entry_vol, q=dividend_yield)
    fair_value_pct_of_par = greeks["price"] / S0

    print(f"\nFixed Coupon Note fair value at inception")
    print(f"(ZCB discounted at SOFR {RISK_FREE_RATE:.2%} + Goldman Sachs CDS {GS_CDS_SPREAD:.2%},")
    print(f"short put priced via QuantLib at SOFR alone (dividend yield {dividend_yield:.2%}), T={TENOR}y,")
    print(f"vol={entry_vol:.2%} interpolated from the {path.index[0].date()} VIX/VIX3M/VIX6M term structure):")
    print(f"  {fair_value_pct_of_par:.2%} of par ({fair_value_pct_of_par - 1:+.2%} vs. par)")

    print(f"\nFixed Coupon Note Greeks at inception:")
    print(f"  Delta: {greeks['delta']:.2f}")
    print(f"  Vega:  {greeks['vega'] / S0 * 0.01:.4f}  (per 1% change in vol)")
    print(f"  Rho:   {greeks['rho'] / S0 * 0.01:.4f}  (per 1% change in SOFR)")
    print(f"  Theta: {greeks['theta'] / S0:.4f} per year / {greeks['theta'] / S0 / 365:.5f} per day")

    print(f"\nMTM Fair Value vs. Par at Inception:")
    print(f"  Price: {greeks['price']:,.2f}  vs. Par (S0): {S0:,.2f}  ({fair_value_pct_of_par:.2%} of par)")

    plot_path(path, S0, underlying_name, mtm_vol_term_structure=vol_term_structure, greeks=greeks, q=dividend_yield)
