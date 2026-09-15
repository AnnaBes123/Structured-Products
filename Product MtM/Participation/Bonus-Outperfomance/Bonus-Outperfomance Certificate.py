import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql

plt.rcParams["font.family"] = "Arial"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

STRIKE = 1.00
OUTPERFORMANCE_PARTICIPATION = 2.0
BARRIER = 0.65
RISK_FREE_RATE = 0.04

ENTRY_DATE = "2025-01-02"
TENOR = 1

TICKER = "AMD"
FRED_SERIES = "SP500"

VOL_TERM_STRUCTURE_TICKERS = {"^VIX": 30, "^VIX3M": 93, "^VIX6M": 182}
VIX_FRED_SERIES = "VIXCLS"


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
    end = start + pd.DateOffset(years=years)
    return fetch_daily_closes(TICKER, start, end, fred_series=FRED_SERIES)


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


def bonus_outperformance_certificate_running_return(S0, path):
    index_return = path / S0 - 1
    strike_level = STRIKE * S0
    barrier_level = BARRIER * S0
    above_strike = path >= strike_level
    breached_so_far = path.cummin() <= barrier_level

    running = pd.Series(0.0, index=path.index)
    running[above_strike] = OUTPERFORMANCE_PARTICIPATION * index_return[above_strike]
    below_strike_and_breached = (~above_strike) & breached_so_far
    running[below_strike_and_breached] = index_return[below_strike_and_breached]
    return running


def _quantlib_process(S, r, q, sigma, today):
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    rf_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count, ql.Continuous, ql.Annual))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count, ql.Continuous, ql.Annual))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, day_count))
    return ql.BlackScholesMertonProcess(spot, div_ts, rf_ts, vol_ts)


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

    days = max(int(round(T * 365.25)), 1)
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
    See the README for the full rationale."""
    discount_q = np.exp(-q * T)
    return {
        "price": S * discount_q, "delta": discount_q, "vega": 0.0, "rho": 0.0,
        "theta": q * S * discount_q,
    }


def down_and_out_put_crr(S, K, H, T, r, sigma, steps, q=0.0):
    """
    Down-and-out put, strike K, barrier H < K, priced on a Cox-Ross-
    Rubinstein binomial lattice (QuantLib BinomialCRRBarrierEngine) with
    the barrier checked once per step - steps should be the number of
    remaining trading days, so the model's monitoring frequency matches
    the daily data it's priced against (see the Barrier Reverse Convertible
    product's README for the full rationale on why this replaced a
    continuous-monitoring closed form: continuous monitoring overstates
    the touch probability relative to what daily data can ever confirm).
    Assumes the barrier has NOT already been breached - the caller is
    responsible for pricing this leg at 0 once the barrier has been
    touched at any point in the certificate's life so far.
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

    n = max(int(round(steps)), 2)  # QuantLib's binomial barrier engine errors below 2 steps
    option.setPricingEngine(ql.BinomialCRRBarrierEngine(process, n, n))
    return option.NPV()


def bonus_outperformance_certificate_price(S, S0, T_remaining, sigma, breached, r=RISK_FREE_RATE, q=0.0, steps=None):
    zero_strike_leg = lepo_price_and_greeks(S, T_remaining, q)
    atm_leg = black_scholes_call(S, STRIKE * S0, T_remaining, r, sigma, q)
    atm_quantity = OUTPERFORMANCE_PARTICIPATION - 1.0
    base_price = zero_strike_leg["price"] + atm_quantity * atm_leg["price"]

    if breached:
        put_price = 0.0
    else:
        H = BARRIER * S0
        n = steps if steps is not None else max(round(T_remaining * 252), 1)
        put_price = down_and_out_put_crr(S, STRIKE * S0, H, T_remaining, r, sigma, n, q)

    return base_price + put_price


def bonus_outperformance_certificate_mtm_price_series(S0, path, vol_term_structure, r=RISK_FREE_RATE, q=0.0):
    maturity_date = path.index[-1]
    H = BARRIER * S0
    breached_so_far = path.cummin() <= H

    n_dates = len(path)
    prices = []
    for i, (date, level) in enumerate(path.items()):
        T_remaining = (maturity_date - date).days / 365.25
        steps_remaining = max(n_dates - 1 - i, 1)  # one CRR barrier check per remaining trading day
        vols_today = {tenor: series[date] for tenor, series in vol_term_structure.items()}
        sigma = interpolate_implied_vol(vols_today, max(T_remaining, 0.0))
        prices.append(
            bonus_outperformance_certificate_price(level, S0, T_remaining, sigma, breached_so_far[date], r,
                                                     q=q, steps=steps_remaining)
        )
    return pd.Series(prices, index=path.index)


def finite_difference_greeks_at(S, S0, T, sigma, breached, r=RISK_FREE_RATE, q=0.0, steps=None):
    """
    Price and Greeks (central finite differences) at an arbitrary spot
    S and barrier-breach status - the general form behind
    finite_difference_greeks, which is just this evaluated at S=S0,
    breached=False (the inception case: spot hasn't moved, barrier
    hasn't been touched).

    The CRR step count is fixed ONCE (from the un-bumped T) and reused for
    every bumped evaluation, including the T-bump for theta, so a
    differing step count between bumps never injects lattice-discreteness
    noise into the Greek.

    bump_S and bump_sigma are much wider than the "0.1% of spot" convention
    used for the closed-form (non-barrier) products in this repo. A CRR
    lattice is rebuilt from scratch on every call, and its node grid scales
    multiplicatively with both S and sigma - as those inputs vary
    continuously, which nodes fall above/below the FIXED strike/barrier
    changes in discrete jumps ("sawtooth" artifacts), not smoothly, and a
    small bump can land entirely inside one of those jumps and return a
    Greek off by a large factor (Vega here ranged from ~1030 to ~1580
    depending on bump size before settling near 1550-1580 past a
    2-vol-point bump; see the Bullish Sharkfin product's README,
    Product MtM/Capital Protection/, for the fuller bump-size scan that
    surfaced this). Rho keeps a small bump since r doesn't enter the
    lattice's node spacing at all, only drift/discounting.
    """
    bump_S, bump_sigma, bump_r, bump_T = S0 * 0.02, 0.02, 0.0001, 21 / 365
    n = steps if steps is not None else max(round(T * 252), 1)

    price = bonus_outperformance_certificate_price(S, S0, T, sigma, breached, r, q=q, steps=n)

    price_up_S = bonus_outperformance_certificate_price(S + bump_S, S0, T, sigma, breached, r, q=q, steps=n)
    price_down_S = bonus_outperformance_certificate_price(S - bump_S, S0, T, sigma, breached, r, q=q, steps=n)
    delta = (price_up_S - price_down_S) / (2 * bump_S)

    price_up_sigma = bonus_outperformance_certificate_price(S, S0, T, sigma + bump_sigma, breached, r, q=q, steps=n)
    price_down_sigma = bonus_outperformance_certificate_price(S, S0, T, sigma - bump_sigma, breached, r, q=q, steps=n)
    vega = (price_up_sigma - price_down_sigma) / (2 * bump_sigma)

    price_up_r = bonus_outperformance_certificate_price(S, S0, T, sigma, breached, r + bump_r, q=q, steps=n)
    price_down_r = bonus_outperformance_certificate_price(S, S0, T, sigma, breached, r - bump_r, q=q, steps=n)
    rho = (price_up_r - price_down_r) / (2 * bump_r)

    price_less_T = bonus_outperformance_certificate_price(S, S0, max(T - bump_T, 0.0), sigma, breached, r, q=q, steps=n)
    theta = (price_less_T - price) / bump_T

    return {"price": price, "delta": delta, "vega": vega, "rho": rho, "theta": theta}


def finite_difference_greeks(S0, T, sigma, r=RISK_FREE_RATE, q=0.0):
    return finite_difference_greeks_at(S0, S0, T, sigma, False, r, q=q)


def realized_annualized_vol(path):
    log_returns = np.log(path / path.shift(1)).dropna()
    return log_returns.std() * np.sqrt(252)


def max_drawdown(path):
    running_max = path.cummax()
    drawdown = path / running_max - 1
    return drawdown.min()


def plot_path(path, S0, underlying_name, mtm_vol_term_structure=None, greeks=None, q=0.0):
    index_return_pct = (path / S0 - 1) * 100
    certificate_return_pct = bonus_outperformance_certificate_running_return(S0, path) * 100

    _, ax = plt.subplots(figsize=(18, 8))
    ax.plot(index_return_pct.index, index_return_pct.values, color="firebrick", linewidth=1.5,
            label=underlying_name)
    ax.tick_params(axis="y", labelcolor="firebrick")
    ax.plot(certificate_return_pct.index, certificate_return_pct.values, color="indianred", linewidth=1.5,
            linestyle="dashed",
            label=f"Bonus-Outperformance Certificate Participation Tracker ({OUTPERFORMANCE_PARTICIPATION:.0%} above strike)")

    ax.grid(True, which="major", color="lightgrey", linewidth=0.6)
    ax.axhline(0, color="lightgrey", linewidth=0.8)
    barrier_pct = (BARRIER - 1) * 100
    ax.axhline(barrier_pct, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.text(0.01, barrier_pct, f"Knock-In Barrier: {barrier_pct:.1f}%", transform=ax.get_yaxis_transform(),
            color="dodgerblue", fontsize=9, va="bottom", ha="left")
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
            f"Barrier Percentage: {BARRIER * 100:,.2f}%\n"
            f"{underlying_name} Return: {S_T / S0 - 1:.2%}\n"
            f"Bonus-Outperformance Certificate Return: {certificate_return:.2%}\n"
            f"Approximate MtM Fair Value vs. Par (%) at Inception: {fair_value_pct_of_par-1:.2%}\n"
            f"Realized Vol (ann.): {vol:.2%}\n"
            f"Max Drawdown: {max_drawdown(path):.2%}"
        )
        ax.text(1.06, 0.5, annotation_text, transform=ax.transAxes,
                fontsize=12, fontweight="light", color="black", va="center", ha="left")

    if mtm_vol_term_structure is not None:
        mtm_price = bonus_outperformance_certificate_mtm_price_series(S0, path, mtm_vol_term_structure, q=q)
        # Indexed to par (S0), matching the index and payoff/tracker lines -
        # NOT to the basket's own day-1 value. This is required so that fair
        # value converges exactly to the terminal payoff at maturity (time
        # value = 0 at expiry, so MTM price = payoff price in real dollar
        # terms) - indexing to the basket's own day-1 value instead would
        # make the two lines land on different final percentages, breaking
        # that identity on the chart. The tradeoff: this MTM line starts
        # above 0% on day 1, reflecting the real option premium the
        # participation rate and put protection cost at this vol - see the
        # console printout.
        mtm_gain_over_par_pct = (mtm_price / S0 - 1) * 100

        ax2 = ax.twinx()
        mtm_line, = ax2.plot(
            mtm_gain_over_par_pct.index, mtm_gain_over_par_pct.values, color="darkred", linewidth=1.5,
            linestyle="solid", label="Bonus-Outperformance Certificate - Approximate MtM (% of Par, Black-Scholes)")
        ax2.set_ylabel("MtM Fair Value vs. Par (%)", color="darkred", rotation=270, labelpad=10)
        ax2.tick_params(axis="y", labelcolor="darkred")

        combined_min = min(index_return_pct.min(), certificate_return_pct.min(), mtm_gain_over_par_pct.min(), barrier_pct)
        combined_max = max(index_return_pct.max(), certificate_return_pct.max(), mtm_gain_over_par_pct.max())
        pad = (combined_max - combined_min) * 0.05
        ax.set_ylim(combined_min - pad, combined_max + pad)
        ax2.set_ylim(combined_min - pad, combined_max + pad)

        lines.append(mtm_line)
        labels.append(mtm_line.get_label())

    end_date = index_return_pct.index[-1]
    ax.set_title(f"{underlying_name} Price Return Path from {path.index[0].date()} to {end_date.date()} \n"
                 f"vs Approximate Mark-to-Market (MtM) Value of Bonus-Outperformance Certificate "
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
    certificate_return = bonus_outperformance_certificate_running_return(S0, path).iloc[-1]

    dividend_yield = fetch_dividend_yield(TICKER)
    print(f"Dividend yield for {underlying_name} ({TICKER}): {dividend_yield:.2%} (flat, continuous - 0% if an index)")

    summary = pd.Series({
        "Entry Date": path.index[0].date(),
        "Maturity Date": path.index[-1].date(),
        "S0": f"{S0:,.2f}",
        "S_T": f"{S_T:,.2f}",
        "Barrier Level": f"{BARRIER * S0:,.2f}",
        f"{underlying_name} Return": f"{S_T / S0 - 1:.2%}",
        "Bonus-Outperformance Certificate Return": f"{certificate_return:.2%}",
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

    greeks = finite_difference_greeks(S0, TENOR, entry_vol, q=dividend_yield)
    fair_value_pct_of_par = greeks["price"] / S0

    print(f"\nBonus-Outperformance Certificate fair value at inception")
    print(f"(QuantLib: LEPO leg closed-form, ATM call via AnalyticEuropeanEngine, down-and-out put")
    print(f"via a CRR binomial lattice (barrier checked once per trading day), S=S0, T={TENOR}y,")
    print(f"vol={entry_vol:.2%} interpolated from the {path.index[0].date()} VIX/VIX3M/VIX6M term")
    print(f"structure, dividend yield {dividend_yield:.2%}):")
    print(f"  {fair_value_pct_of_par:.2%} of par ({fair_value_pct_of_par - 1:+.2%} vs. par)")

    print(f"\nBonus-Outperformance Certificate Greeks at inception (finite-difference):")
    print(f"  Delta: {greeks['delta']:.2f}")
    print(f"  Vega:  {greeks['vega'] / S0 * 0.01:.4f}  (per 1% change in vol)")
    print(f"  Rho:   {greeks['rho'] / S0 * 0.01:.4f}  (per 1% change in rates)")
    print(f"  Theta: {greeks['theta'] / S0:.4f} per year / {greeks['theta'] / S0 / 365:.5f} per day")

    print(f"\nMTM Fair Value vs. Par at Inception:")
    print(f"  Price: {greeks['price']:,.2f}  vs. Par (S0): {S0:,.2f}  ({fair_value_pct_of_par:.2%} of par)")

    plot_path(path, S0, underlying_name, mtm_vol_term_structure=vol_term_structure, greeks=greeks, q=dividend_yield)
