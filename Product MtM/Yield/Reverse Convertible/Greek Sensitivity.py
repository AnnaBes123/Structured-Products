import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql

plt.rcParams["font.family"] = "Arial"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Same product terms as "Reverse Convertible.py" in this folder ---
STRIKE = 0.90
RISK_FREE_RATE = 0.04
GS_CDS_SPREAD = 0.005308

ENTRY_DATE = "2025-01-02"
TENOR = 1

TICKER = "^GSPC"
FRED_SERIES = "SP500"

VOL_TERM_STRUCTURE_TICKERS = {"^VIX": 30, "^VIX3M": 93, "^VIX6M": 182}
VIX_FRED_SERIES = "VIXCLS"

# T (TENOR), strike, funding curve (SOFR + CDS spread) are fixed constants
# throughout this whole analysis - the only thing that varies below is
# spot. Vol is also held at its inception value for every row.
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
    ("^" tickers) are treated as paying none. See Reverse Convertible.py in
    this folder for the full rationale and field-selection notes."""
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
    """Plain European put via QuantLib's AnalyticEuropeanEngine (dividend
    yield q is a native input). See Reverse Convertible.py in this folder."""
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


def reverse_convertible_mtm(S, S0, T_remaining, sigma, r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD, q=0.0):
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


# ---------------------------------------------------------------------------
# GREEKS LADDER
#
# T, strike and the funding curve are held constant throughout - the ONLY
# thing that varies across rows is spot. Vol is also pinned at its
# inception value for every row. Each Greek at a given spot level IS the
# answer to "how much does MTM move for a 1-unit change in that variable,
# right now, at this spot": e.g. the Vega value in the row for spot=80%
# tells you directly how many percentage points of MTM a 1% change in
# vol is worth if the index has already dropped 20%.
#
# Theta is deliberately excluded: with T fixed throughout this analysis,
# a "time passing" Greek doesn't belong in a table where time never
# actually moves.
# ---------------------------------------------------------------------------

def greek_sensitivity_table(S0, T, sigma, r=RISK_FREE_RATE, spot_multiples=SPOT_SCENARIO_RANGE, q=0.0):
    rows = []
    for multiple in spot_multiples:
        S = multiple * S0
        g = reverse_convertible_mtm(S, S0, T, sigma, r, q=q)
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
        ("Price (% of Par)", "Fair Value (% of Par)", "firebrick"),
        ("Delta", "Delta", "darkred"),
        ("Vega (per 1% vol)", "Vega (per 1% change in vol)", "indianred"),
        ("Rho (per 1% rate)", "Rho (per 1% change in SOFR)", "brown"),
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
        f"{underlying_name} Reverse Convertible - Greeks Ladder (T and strike fixed, spot varies)\n"
        f"Fixed throughout: T={T}y, vol={sigma:.2%}, SOFR={r:.2%}, CDS={GS_CDS_SPREAD:.2%}, "
        f"strike={STRIKE:.0%} of S0={S0:,.2f}"
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    print("Greeks Ladder - Reverse Convertible")
    print("=" * 60)
    print("This is NOT a new backtest and does not use the historical path.")
    print("It reuses the same entry conditions as the main backtest script,")
    print("then holds T, strike, funding curve and vol all fixed and varies")
    print("ONLY spot, so each Greek's value at a given spot level tells you")
    print("directly how much MTM moves for a 1-unit change in that variable")
    print("at that spot (e.g. the Vega row at spot=80% is what a 1% vol move")
    print("is worth if the index has already dropped 20%). Theta is excluded")
    print("since T never varies in this analysis.\n")

    underlying_name = fetch_underlying_name(TICKER)
    print(f"Fetching {underlying_name} entry level for {ENTRY_DATE}...")
    path = fetch_index_path(ENTRY_DATE)
    S0 = float(path.iloc[0])

    dividend_yield = fetch_dividend_yield(TICKER)

    print(f"Fetching SPX implied vol term structure for {ENTRY_DATE}...")
    raw_term_structure = fetch_vol_term_structure(ENTRY_DATE)
    vols_at_entry = {tenor: series.iloc[0] for tenor, series in raw_term_structure.items()}
    entry_vol = interpolate_implied_vol(vols_at_entry, TENOR)

    print(f"\nFixed throughout (only spot varies below):")
    print(f"  Underlying:      {underlying_name} ({TICKER})")
    print(f"  Entry Date:      {ENTRY_DATE}")
    print(f"  S0 (spot):       {S0:,.2f}")
    print(f"  Strike:          {STRIKE:.0%} of S0")
    print(f"  Tenor (T):       {TENOR} year(s)")
    print(f"  Vol (sigma):     {entry_vol:.2%}  (from the VIX/VIX3M/VIX6M term structure on {ENTRY_DATE})")
    print(f"  SOFR proxy:      {RISK_FREE_RATE:.2%}")
    print(f"  GS CDS spread:   {GS_CDS_SPREAD:.2%}")
    print(f"  Dividend yield:  {dividend_yield:.2%}  (flat, continuous - 0% if an index)")

    sensitivity = greek_sensitivity_table(S0, TENOR, entry_vol, RISK_FREE_RATE, q=dividend_yield)
    print(f"\nGreeks Ladder (QuantLib closed-form Black-Scholes-Merton for the put, closed-form for the")
    print(f"ZCB - both exact, not finite-difference):")
    print(sensitivity.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    plot_greek_sensitivity(sensitivity, S0, TENOR, entry_vol, RISK_FREE_RATE, underlying_name)
