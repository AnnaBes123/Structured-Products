import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql

plt.rcParams["font.family"] = "Arial"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Same product terms as "Bonus Certificate.py" in this folder ---
STRIKE = 1.00
BONUS_LEVEL = 1.00
BARRIER = 0.70
RISK_FREE_RATE = 0.04

ENTRY_DATE = "2025-01-02"
TENOR = 1

TICKER = "^GSPC"
FRED_SERIES = "SP500"

VOL_TERM_STRUCTURE_TICKERS = {"^VIX": 30, "^VIX3M": 93, "^VIX6M": 182}
VIX_FRED_SERIES = "VIXCLS"

# T, bonus level and barrier are fixed constants throughout this whole
# analysis - the only thing that varies below is spot. Vol and rate are
# also held at their inception values for every row. Uncapped: no short
# call leg, upside is 1:1 and unlimited - see "Bonus Certificate.py".
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


def _quantlib_process(S, r, q, sigma, today):
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    rf_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count, ql.Continuous, ql.Annual))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count, ql.Continuous, ql.Annual))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, day_count))
    return ql.BlackScholesMertonProcess(spot, div_ts, rf_ts, vol_ts)


def lepo_price_and_greeks(S, T, q=0.0):
    """LEPO (Low Exercise Price Option): the risk-neutral PV of receiving
    one share at maturity, S*e^(-qT) - equals spot exactly only when q=0.
    See the README for the full rationale."""
    discount_q = np.exp(-q * T)
    return {
        "price": S * discount_q, "delta": discount_q, "vega": 0.0, "rho": 0.0,
        "theta": q * S * discount_q,
    }


def down_and_out_put_crr(S, K, H, T, r, sigma, steps, q=0.0):
    """
    Down-and-out put via a CRR binomial lattice (QuantLib
    BinomialCRRBarrierEngine), barrier checked once per step - steps
    should be the number of remaining trading days. See "Bonus
    Certificate.py" in this folder and its README for the full rationale
    (this replaced a continuous-monitoring closed form, which overstates
    the touch probability relative to what daily data can ever confirm).
    Assumes the barrier has NOT already been breached.
    """
    if S <= H:
        return 0.0
    if T <= 0:
        return max(K - S, 0.0)

    today = ql.Date(1, 1, 2000)
    ql.Settings.instance().evaluationDate = today
    process = _quantlib_process(S, r, q, sigma, today)

    days = max(int(round(T * 365.25)), 1)
    exercise = ql.EuropeanExercise(today + ql.Period(days, ql.Days))
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    option = ql.BarrierOption(ql.Barrier.DownOut, H, 0.0, payoff, exercise)

    n = max(int(round(steps)), 2)
    option.setPricingEngine(ql.BinomialCRRBarrierEngine(process, n, n))
    return option.NPV()


def bonus_certificate_price(S, S0, T_remaining, sigma, breached, r=RISK_FREE_RATE, q=0.0, steps=None):
    lepo = lepo_price_and_greeks(S, T_remaining, q)

    if breached:
        put_price = 0.0
    else:
        K = BONUS_LEVEL * S0
        H = BARRIER * S0
        n = steps if steps is not None else max(round(T_remaining * 252), 1)
        put_price = down_and_out_put_crr(S, K, H, T_remaining, r, sigma, n, q)

    return lepo["price"] + put_price


def finite_difference_greeks_at(S, S0, T, sigma, breached, r=RISK_FREE_RATE, q=0.0, steps=None):
    """
    Price and Greeks via central finite differences (bump-and-reprice).
    There is no simple closed form once the barrier put is added, so all
    four Greeks here are numerical: each is [price(x+bump) - price(x-bump)]
    / (2*bump) for the relevant variable x.

    The CRR step count is fixed ONCE (from the un-bumped T) and reused for
    every bumped evaluation, so a differing step count between bumps never
    injects lattice-discreteness noise into the Greek.

    bump_S and bump_sigma are wide by finite-difference convention - a CRR
    lattice's node grid scales multiplicatively with both S and sigma, so a
    small bump can land inside a "sawtooth" discretization jump and return
    a Greek off by a large factor. See the Bullish Sharkfin product's
    README (Product MtM/Capital Protection/) for the bump-size scan that
    surfaced this. Rho keeps a small bump since r doesn't enter the
    lattice's node spacing.
    """
    bump_S, bump_sigma, bump_r, bump_T = S0 * 0.02, 0.02, 0.0001, 21 / 365
    n = steps if steps is not None else max(round(T * 252), 1)

    price = bonus_certificate_price(S, S0, T, sigma, breached, r, q=q, steps=n)

    price_up_S = bonus_certificate_price(S + bump_S, S0, T, sigma, breached, r, q=q, steps=n)
    price_down_S = bonus_certificate_price(S - bump_S, S0, T, sigma, breached, r, q=q, steps=n)
    delta = (price_up_S - price_down_S) / (2 * bump_S)

    price_up_sigma = bonus_certificate_price(S, S0, T, sigma + bump_sigma, breached, r, q=q, steps=n)
    price_down_sigma = bonus_certificate_price(S, S0, T, sigma - bump_sigma, breached, r, q=q, steps=n)
    vega = (price_up_sigma - price_down_sigma) / (2 * bump_sigma)

    price_up_r = bonus_certificate_price(S, S0, T, sigma, breached, r + bump_r, q=q, steps=n)
    price_down_r = bonus_certificate_price(S, S0, T, sigma, breached, r - bump_r, q=q, steps=n)
    rho = (price_up_r - price_down_r) / (2 * bump_r)

    price_less_T = bonus_certificate_price(S, S0, max(T - bump_T, 0.0), sigma, breached, r, q=q, steps=n)
    theta = (price_less_T - price) / bump_T

    return {"price": price, "delta": delta, "vega": vega, "rho": rho, "theta": theta}


# ---------------------------------------------------------------------------
# GREEKS LADDER
#
# T, bonus level and barrier are held constant throughout - the ONLY thing
# that varies across rows is spot. Vol and rate are also pinned at their
# inception values for every row. Each Greek at a given spot level IS the
# answer to "how much does MTM move for a 1-unit change in that variable,
# right now, at this spot".
#
# Computed for BOTH "Not Breached" (the protective put still alive) and
# "Breached" (knocked out for good), since the barrier changes every
# Greek's value meaningfully. Below the barrier itself, only "Breached"
# is a physically reachable state, so "Not Breached" is simply omitted
# there - you cannot be below the barrier without having touched it.
# ---------------------------------------------------------------------------

def greek_sensitivity_table(S0, T, sigma, r=RISK_FREE_RATE, spot_multiples=SPOT_SCENARIO_RANGE, q=0.0):
    rows = []
    for multiple in spot_multiples:
        S = multiple * S0
        statuses = ["Breached"] if S <= BARRIER * S0 else ["Not Breached", "Breached"]
        for status in statuses:
            g = finite_difference_greeks_at(S, S0, T, sigma, status == "Breached", r, q=q)
            rows.append({
                "Spot (% of S0)": multiple * 100,
                "Barrier Status": status,
                "Price (% of Par)": g["price"] / S0 * 100,
                "Delta": g["delta"],
                "Vega (per 1% vol)": g["vega"] / S0 * 0.01,
                "Rho (per 1% rate)": g["rho"] / S0 * 0.01,
            })
    return pd.DataFrame(rows)


def plot_greek_sensitivity(table, S0, T, sigma, r, underlying_name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    status_styles = {"Not Breached": {"linestyle": "solid"}, "Breached": {"linestyle": "dashed"}}

    panels = [
        ("Price (% of Par)", "Fair Value (% of Par)"),
        ("Delta", "Delta"),
        ("Vega (per 1% vol)", "Vega (per 1% change in vol)"),
        ("Rho (per 1% rate)", "Rho (per 1% change in rates)"),
    ]

    for ax, (col, ylabel) in zip(axes.flat, panels):
        for status, style in status_styles.items():
            subset = table[table["Barrier Status"] == status]
            if subset.empty:
                continue
            ax.plot(subset["Spot (% of S0)"], subset[col], color="firebrick", linewidth=1.8,
                    marker="o", markersize=3, label=status, **style)
        ax.axvline(100, color="lightgrey", linewidth=0.8, linestyle="dashed")
        ax.axvline(BONUS_LEVEL * 100, color="dodgerblue", linewidth=0.8, linestyle="dotted")
        ax.axvline(BARRIER * 100, color="darkgreen", linewidth=0.8, linestyle="dotted")
        ax.axhline(0, color="lightgrey", linewidth=0.6)
        ax.grid(True, color="lightgrey", linewidth=0.4)
        ax.set_xlabel("Spot (% of S0)")
        ax.set_ylabel(ylabel)
        ax.set_title(col)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"{underlying_name} Bonus Certificate (Uncapped) - Greeks Ladder (T, bonus level, barrier fixed, spot varies)\n"
        f"Fixed throughout: T={T}y, vol={sigma:.2%}, r={r:.2%}, bonus={BONUS_LEVEL:.0%}, "
        f"barrier={BARRIER:.0%} of S0={S0:,.2f}"
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    print("Greeks Ladder - Bonus Certificate")
    print("=" * 60)
    print("This is NOT a new backtest and does not use the historical path.")
    print("It reuses the same entry conditions as the main backtest script,")
    print("then holds T, bonus level, barrier, cap, vol and rate all fixed")
    print("and varies ONLY spot, so each Greek's value at a given spot level")
    print("tells you directly how much MTM moves for a 1-unit change in that")
    print("variable at that spot. Shown for both barrier states where both")
    print("are reachable (Not Breached / Breached).\n")

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
    print(f"  Bonus Level:     {BONUS_LEVEL:.0%} of S0")
    print(f"  Barrier:         {BARRIER:.0%} of S0 = {BARRIER * S0:,.2f}")
    print(f"  Tenor (T):       {TENOR} year(s)")
    print(f"  Vol (sigma):     {entry_vol:.2%}  (from the VIX/VIX3M/VIX6M term structure on {ENTRY_DATE})")
    print(f"  Risk-free rate:  {RISK_FREE_RATE:.2%}")
    print(f"  Dividend yield:  {dividend_yield:.2%}  (flat, continuous - 0% if an index)")

    sensitivity = greek_sensitivity_table(S0, TENOR, entry_vol, RISK_FREE_RATE, q=dividend_yield)
    print(f"\nGreeks Ladder (finite-difference - there is no simple closed form once")
    print(f"the barrier put is added, so all four Greeks are numerical:")
    print(f"[price(x+bump) - price(x-bump)] / (2*bump) for each variable x):")
    print(sensitivity.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    plot_greek_sensitivity(sensitivity, S0, TENOR, entry_vol, RISK_FREE_RATE, underlying_name)
