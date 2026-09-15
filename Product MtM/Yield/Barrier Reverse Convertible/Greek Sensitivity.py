import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql

plt.rcParams["font.family"] = "Arial"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Same product terms as "Barrier Reverse Convertible.py" in this folder ---
STRIKE = 0.90
BARRIER = 0.70
RISK_FREE_RATE = 0.04
GS_CDS_SPREAD = 0.005308

ENTRY_DATE = "2025-01-02"
TENOR = 1

TICKER = "^GSPC"
FRED_SERIES = "SP500"

VOL_TERM_STRUCTURE_TICKERS = {"^VIX": 30, "^VIX3M": 93, "^VIX6M": 182}
VIX_FRED_SERIES = "VIXCLS"

# T (TENOR), strike, barrier and the funding curve are fixed constants
# throughout this whole analysis - the only thing that varies below is
# spot. Vol is also held at its inception value for every row. Spot never
# goes low enough to already be "breached" in this ladder (the whole point
# is to see the un-breached Greeks profile) - see the main backtest script
# for what happens once the barrier is actually touched.
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


def black_scholes_put(S, K, T, r, sigma, q=0.0):
    """Plain (no barrier) European put via QuantLib's AnalyticEuropeanEngine
    (dividend yield q is a native input) - used as the base for in-out
    parity and for the "already breached" case."""
    if T <= 0:
        return {"price": max(K - S, 0.0)}

    today = ql.Date(1, 1, 2000)
    ql.Settings.instance().evaluationDate = today
    process = _quantlib_process(S, r, q, sigma, today)

    days = max(int(round(T * 365.25)), 1)
    exercise = ql.EuropeanExercise(today + ql.Period(days, ql.Days))
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return {"price": option.NPV()}


def _quantlib_process(S, r, q, sigma, today):
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    rf_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count, ql.Continuous, ql.Annual))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count, ql.Continuous, ql.Annual))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, day_count))
    return ql.BlackScholesMertonProcess(spot, div_ts, rf_ts, vol_ts)


def down_and_in_put_crr(S, K, H, T, r, sigma, steps, q=0.0):
    """
    Down-and-in put on a Cox-Ross-Rubinstein binomial lattice (QuantLib
    BinomialCRRBarrierEngine), barrier checked once per step. See
    "Barrier Reverse Convertible.py" in this folder for why this replaced
    the closed-form (continuous-monitoring) reflection-principle formula -
    the monitoring frequency here should match how often the note's price
    is actually ever observed (once per trading day), not true continuous
    monitoring, which overstates the touch probability.
    """
    if T <= 0:
        return max(K - S, 0.0) if S <= H else 0.0
    if S <= H:
        return black_scholes_put(S, K, T, r, sigma, q)["price"]

    today = ql.Date(1, 1, 2000)  # arbitrary fixed anchor - only T (via day count) matters
    ql.Settings.instance().evaluationDate = today
    process = _quantlib_process(S, r, q, sigma, today)

    days = max(int(round(T * 365.25)), 1)
    exercise = ql.EuropeanExercise(today + ql.Period(days, ql.Days))
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    option = ql.BarrierOption(ql.Barrier.DownIn, H, 0.0, payoff, exercise)

    n = max(int(round(steps)), 2)  # QuantLib's binomial barrier engine errors below 2 steps
    option.setPricingEngine(ql.BinomialCRRBarrierEngine(process, n, n))
    return option.NPV()


def barrier_reverse_convertible_price(S, S0, T_remaining, sigma, breached, r=RISK_FREE_RATE,
                                       credit_spread=GS_CDS_SPREAD, steps=None, q=0.0):
    """
    Once breached=True, the down-and-in put has permanently knocked in and
    is priced as an ordinary vanilla put from then on.

    `steps` is the CRR lattice's step count (one barrier check per step);
    defaults to an approximate 252-trading-day year if omitted. `q` is the
    underlying's dividend yield (0 for an index).
    """
    zcb = zcb_price_and_greeks(S0, T_remaining, r + credit_spread)
    K = STRIKE * S0

    if breached:
        put_price = black_scholes_put(S, K, T_remaining, r, sigma, q)["price"]
    else:
        H = BARRIER * S0
        n = steps if steps is not None else max(round(T_remaining * 252), 1)
        put_price = down_and_in_put_crr(S, K, H, T_remaining, r, sigma, n, q)

    put_quantity = 1.0 / STRIKE
    return zcb["price"] - put_quantity * put_price


# ---------------------------------------------------------------------------
# GREEKS LADDER
#
# T, strike, barrier and the funding curve are held constant throughout -
# the ONLY thing that varies across rows is spot. Vol is also pinned at its
# inception value for every row. Each Greek at a given spot level IS the
# answer to "how much does MTM move for a 1-unit change in that variable,
# right now, at this spot" - same design as every other product in this
# repo.
#
# Greeks are bump-and-reprice finite differences on the full note price
# (not by differentiating the barrier formula further) - barrier-option
# Greeks are notoriously messy near the barrier (delta in particular can be
# discontinuous there), so a numerical bump is simpler and more robust,
# same approach as the Bonus-Outperformance certificate.
# ---------------------------------------------------------------------------

def finite_difference_greeks_at(S, S0, T, sigma, breached, r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD, steps=None, q=0.0):
    """
    The CRR step count is fixed ONCE (from the un-bumped T) and reused for
    every bumped evaluation, so a differing step count between bumps never
    injects lattice-discreteness noise into the Greek.

    bump_S and bump_sigma are wide by finite-difference convention - a CRR
    lattice's node grid scales multiplicatively with both S and sigma, so a
    small bump can land inside a "sawtooth" discretization jump and return
    a Greek off by an order of magnitude. See the Bullish Sharkfin
    product's README (Product MtM/Capital Protection/) for the bump-size
    scan that surfaced this. Rho keeps a small bump since r doesn't enter
    the lattice's node spacing.
    """
    bump_S, bump_sigma, bump_r = S0 * 0.02, 0.02, 0.0001
    n = steps if steps is not None else max(round(T * 252), 1)

    price = barrier_reverse_convertible_price(S, S0, T, sigma, breached, r, credit_spread, steps=n, q=q)

    price_up_S = barrier_reverse_convertible_price(S + bump_S, S0, T, sigma, breached, r, credit_spread, steps=n, q=q)
    price_down_S = barrier_reverse_convertible_price(S - bump_S, S0, T, sigma, breached, r, credit_spread, steps=n, q=q)
    delta = (price_up_S - price_down_S) / (2 * bump_S)

    price_up_sigma = barrier_reverse_convertible_price(S, S0, T, sigma + bump_sigma, breached, r, credit_spread, steps=n, q=q)
    price_down_sigma = barrier_reverse_convertible_price(S, S0, T, sigma - bump_sigma, breached, r, credit_spread, steps=n, q=q)
    vega = (price_up_sigma - price_down_sigma) / (2 * bump_sigma)

    price_up_r = barrier_reverse_convertible_price(S, S0, T, sigma, breached, r + bump_r, credit_spread, steps=n, q=q)
    price_down_r = barrier_reverse_convertible_price(S, S0, T, sigma, breached, r - bump_r, credit_spread, steps=n, q=q)
    rho = (price_up_r - price_down_r) / (2 * bump_r)

    return {"price": price, "delta": delta, "vega": vega, "rho": rho}


def greek_sensitivity_table(S0, T, sigma, r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD,
                             spot_multiples=SPOT_SCENARIO_RANGE, q=0.0):
    rows = []
    for multiple in spot_multiples:
        S = multiple * S0
        breached = multiple <= BARRIER  # a spot scenario AT/BELOW the barrier is already knocked in
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
        ("Price (% of Par)", "Fair Value (% of Par)", "firebrick"),
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
        f"{underlying_name} Barrier Reverse Convertible - Greeks Ladder (T, strike, barrier fixed, spot varies)\n"
        f"Fixed throughout: T={T}y, vol={sigma:.2%}, SOFR={r:.2%}, CDS={GS_CDS_SPREAD:.2%}, "
        f"strike={STRIKE:.0%}, barrier={BARRIER:.0%} of S0={S0:,.2f}"
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    print("Greeks Ladder - Barrier Reverse Convertible")
    print("=" * 60)
    print("This is NOT a new backtest and does not use the historical path.")
    print("It reuses the same entry conditions as the main backtest script,")
    print("then holds T, strike, barrier, funding curve and vol all fixed and")
    print("varies ONLY spot, so each Greek's value at a given spot level tells")
    print("you directly how much MTM moves for a 1-unit change in that")
    print("variable at that spot. Greeks are finite differences (bump and")
    print("reprice), not further differentiation of the barrier formula,")
    print("since barrier Greeks can be discontinuous right at the barrier.\n")

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
    print(f"  Barrier:         {BARRIER:.0%} of S0 (knock-in, CRR lattice, checked once per trading day)")
    print(f"  Tenor (T):       {TENOR} year(s)")
    print(f"  Vol (sigma):     {entry_vol:.2%}  (from the VIX/VIX3M/VIX6M term structure on {ENTRY_DATE})")
    print(f"  SOFR proxy:      {RISK_FREE_RATE:.2%}")
    print(f"  GS CDS spread:   {GS_CDS_SPREAD:.2%}")
    print(f"  Dividend yield:  {dividend_yield:.2%}  (flat, continuous - 0% if an index)")

    sensitivity = greek_sensitivity_table(S0, TENOR, entry_vol, q=dividend_yield)
    print(f"\nGreeks Ladder (finite differences, bump-and-reprice):")
    print(sensitivity.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    plot_greek_sensitivity(sensitivity, S0, TENOR, entry_vol, RISK_FREE_RATE, underlying_name)
