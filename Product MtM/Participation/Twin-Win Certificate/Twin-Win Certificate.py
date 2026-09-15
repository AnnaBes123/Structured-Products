import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql

plt.rcParams["font.family"] = "Arial"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Product terms ---
STRIKE = 1.00                  # LEPO reference / down-and-out put strike, as a fraction of S0
BARRIER = 0.85                 # down-and-out barrier (H) for the embedded puts, H < STRIKE
PUT_QUANTITY = 2.0             # the "twin win" multiple - set by hand like every other manually-
                                # set term in this repo. 2.0 is what gives the classic twin-win
                                # payoff: it exactly cancels the LEPO's own 1x downside exposure
                                # below the strike and replaces it with an equal-and-opposite GAIN
                                # (see the replication note below for the algebra) - as long as the
                                # barrier is never touched.
RISK_FREE_RATE = 0.04

ENTRY_DATE = "2025-01-02"
TENOR = 1

# TICKER drives dividend handling automatically: any "^"-prefixed Yahoo
# index ticker (e.g. "^GSPC") is treated as paying no dividend (q=0);
# any real stock ticker (e.g. "AAPL", "MCD") gets a real trailing dividend
# yield fetched and applied - see fetch_dividend_yield below. No separate
# flag needed, just set TICKER to whichever kind of underlying you want.
TICKER = "RACE"
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
    """
    Trailing dividend yield, used as a flat continuous yield q - this is
    what lets TICKER be either a stock (real dividend yield applied) or an
    index (q=0). Indices ("^" tickers) are treated as paying none - Yahoo
    doesn't expose a meaningful per-ticker yield field for them anyway.

    Prefers `trailingAnnualDividendYield` (already a plain fraction, e.g.
    0.0032 for 0.32%) over `dividendYield`, since yfinance/Yahoo have at
    various times returned the latter as a PERCENTAGE (e.g. 0.33 meaning
    0.33%, not 33%) rather than a fraction - mixing the two up would
    silently overstate the yield ~100x. Falls back to dividendRate/price
    if even that field is missing.
    """
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
    """
    Returns the daily CLOSE series. The barrier is observed "on close" -
    breach is determined from the same daily closing price used for
    valuation, matching how the CRR forward-pricing lattice already checks
    the barrier once per trading day (see the Barrier Reverse Convertible
    product's README, Product MtM/Yield/, for the full rationale).
    """
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


def black_scholes_put(S, K, T, r, sigma, q=0.0):
    """Plain European put via QuantLib's AnalyticEuropeanEngine - used only
    for the closed-form verification check (barrier pushed unreachable ->
    the down-and-out put should collapse to this)."""
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


def lepo_price_and_greeks(S, T, q=0.0):
    """LEPO (Low Exercise Price Option): the risk-neutral PV of receiving
    one share at maturity, S*e^(-qT) - equals spot exactly only when q=0.
    Priced directly here (not via a general option engine with K~0) since
    it has an exact closed form and K=0 is a numerically awkward input for
    one. Independent of r entirely - a well-known identity (the PV of a
    future share delivery carries no rate exposure, unlike a strike-
    bearing option)."""
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
    product's README, Product MtM/Yield/, for the full rationale). Assumes
    the barrier has NOT already been breached - the caller is responsible
    for pricing this leg at 0 once the barrier has been touched at any
    point in the certificate's life so far.
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


# ---------------------------------------------------------------------------
# REPLICATION: Twin-Win Certificate
#            = Long LEPO (zero-strike call, = holding the underlying via a
#              forward purchase)
#            + PUT_QUANTITY (2x) Long Down-and-Out Put (struck at
#              STRIKE=100% of S0, barrier BARRIER < STRIKE, checked once
#              per trading day)
#
# The defining feature: as long as the barrier is never touched, a FALL in
# the index below the strike is converted into a GAIN of the same
# magnitude, not a loss - hence "twin win" (you profit whichever direction
# the index moves, as long as it doesn't fall too far). The algebra: below
# the strike (K = STRIKE*S0) and not breached, the certificate is worth
# LEPO(S) + 2*(K-S) = S + 2K - 2S = 2K - S. In return terms (K=S0):
# (2*S0 - S)/S0 - 1 = 1 - S/S0 = -(S/S0 - 1) = -index_return - the exact
# NEGATIVE of the index's own return. A 2x put quantity is what makes this
# exact cancel-and-flip work: 1x would only cancel the LEPO's downside
# (flat payoff below strike, no upside from a fall); anything other than
# 2x leaves either a net loss or a leveraged gain instead of a clean
# mirror image.
#
# Above the strike, the puts are worthless and the certificate is just the
# LEPO - ordinary 1:1 upside participation, same as the plain
# Outperformance certificate's zero-strike leg.
#
# Touch the barrier once and BOTH puts are knocked out for good - the
# "twin win" inversion feature is gone, and the certificate reverts to
# being just the LEPO for the remainder of its life: ordinary 1:1 index
# tracking, upside OR downside, no more floor and no more inversion.
#
# The puts are priced on a Cox-Ross-Rubinstein binomial lattice (QuantLib
# BinomialCRRBarrierEngine) rather than a continuous-monitoring closed
# form - see the Barrier Reverse Convertible product's README
# (Product MtM/Yield/) for the full rationale; the same reasoning applies
# here unchanged.
# ---------------------------------------------------------------------------

def twin_win_certificate_price(S, S0, T_remaining, sigma, breached, r=RISK_FREE_RATE, q=0.0, steps=None):
    lepo = lepo_price_and_greeks(S, T_remaining, q)

    if breached:
        put_price = 0.0
    else:
        K = STRIKE * S0
        H = BARRIER * S0
        n = steps if steps is not None else max(round(T_remaining * 252), 1)
        put_price = down_and_out_put_crr(S, K, H, T_remaining, r, sigma, n, q)

    return lepo["price"] + PUT_QUANTITY * put_price


def verify_against_closed_form(S0, T, sigma, r=RISK_FREE_RATE, q=0.0, steps=None):
    """
    Barrier pushed unreachable (H -> 0): the down-and-out put can never
    knock out, so its CRR price should converge onto the plain vanilla put
    struck at STRIKE - confirms the barrier engine collapses to the
    ordinary closed form in the no-barrier limit, the same style of check
    used for the Bonus Certificate's down-and-out put.
    """
    K = STRIKE * S0
    n = steps if steps is not None else max(round(T * 252), 1)

    down_and_out_unreachable = down_and_out_put_crr(S0, K, H=1e-6, T=T, r=r, sigma=sigma, steps=n, q=q)
    vanilla_put = black_scholes_put(S0, K, T, r, sigma, q)["price"]
    rel_diff = abs(down_and_out_unreachable - vanilla_put) / vanilla_put

    return {"down_and_out_unreachable": down_and_out_unreachable, "vanilla_put": vanilla_put, "rel_diff": rel_diff}


def twin_win_certificate_running_return(S0, path):
    """
    Participation tracker: the terminal payoff FORMULA applied to today's
    spot, as a return over par. NOT what you'd actually receive if the
    certificate were sold or unwound today - it ignores all remaining time
    value in the still-live puts, unlike the MTM fair-value line, which is
    the closest thing to an actual today's-value estimate.

    If the barrier has never been touched up to today: index return above
    the strike (ordinary upside participation), or the exact NEGATIVE of
    the index return below it (the "twin win" inversion - see the
    REPLICATION note above for the algebra). Once the barrier IS touched,
    the inversion is gone for good - plain index return, up or down, for
    every subsequent date.
    """
    strike_level = STRIKE * S0
    barrier_level = BARRIER * S0
    breached_so_far = path.cummin() <= barrier_level

    put_intrinsic = np.maximum(strike_level - path.values, 0.0)
    put_intrinsic = np.where(breached_so_far, 0.0, put_intrinsic)
    payoff = path.values + PUT_QUANTITY * put_intrinsic
    return pd.Series(payoff, index=path.index) / S0 - 1.0


def twin_win_certificate_mtm_price_series(S0, path, vol_term_structure, r=RISK_FREE_RATE, q=0.0):
    maturity_date = path.index[-1]
    barrier_level = BARRIER * S0
    breached_so_far = path.cummin() <= barrier_level

    n_dates = len(path)
    prices = []
    for i, (date, level) in enumerate(path.items()):
        T_remaining = (maturity_date - date).days / 365.25
        steps_remaining = max(n_dates - 1 - i, 1)  # one CRR barrier check per remaining trading day
        vols_today = {tenor: series[date] for tenor, series in vol_term_structure.items()}
        sigma = interpolate_implied_vol(vols_today, max(T_remaining, 0.0))
        prices.append(
            twin_win_certificate_price(level, S0, T_remaining, sigma, breached_so_far[date], r, q=q,
                                        steps=steps_remaining)
        )
    return pd.Series(prices, index=path.index)


def finite_difference_greeks_at(S, S0, T, sigma, breached, r=RISK_FREE_RATE, q=0.0, steps=None):
    """
    Price and Greeks (central finite differences) at an arbitrary spot S
    and barrier-breach status - the general form behind
    finite_difference_greeks, which is just this evaluated at S=S0,
    breached=False (the inception case: spot hasn't moved, barrier hasn't
    been touched).

    The CRR step count is fixed ONCE (from the un-bumped T) and reused for
    every bumped evaluation, including the T-bump for theta, so a
    differing step count between bumps never injects lattice-discreteness
    noise into the Greek.

    bump_S and bump_sigma are much wider than the "0.1% of spot" convention
    used for the closed-form (non-barrier) products in this repo. A CRR
    lattice is rebuilt from scratch on every call, and its node grid scales
    multiplicatively with both S and sigma, so a small bump can land
    entirely inside a discrete "sawtooth" lattice artifact and return a
    Greek off by a large factor - see the Bullish Sharkfin product's
    README (Product MtM/Capital Protection/) for the specific bump-size
    scan that surfaced this.
    """
    bump_S, bump_sigma, bump_r, bump_T = S0 * 0.02, 0.02, 0.0001, 21 / 365
    n = steps if steps is not None else max(round(T * 252), 1)

    price = twin_win_certificate_price(S, S0, T, sigma, breached, r, q=q, steps=n)

    price_up_S = twin_win_certificate_price(S + bump_S, S0, T, sigma, breached, r, q=q, steps=n)
    price_down_S = twin_win_certificate_price(S - bump_S, S0, T, sigma, breached, r, q=q, steps=n)
    delta = (price_up_S - price_down_S) / (2 * bump_S)

    price_up_sigma = twin_win_certificate_price(S, S0, T, sigma + bump_sigma, breached, r, q=q, steps=n)
    price_down_sigma = twin_win_certificate_price(S, S0, T, sigma - bump_sigma, breached, r, q=q, steps=n)
    vega = (price_up_sigma - price_down_sigma) / (2 * bump_sigma)

    price_up_r = twin_win_certificate_price(S, S0, T, sigma, breached, r + bump_r, q=q, steps=n)
    price_down_r = twin_win_certificate_price(S, S0, T, sigma, breached, r - bump_r, q=q, steps=n)
    rho = (price_up_r - price_down_r) / (2 * bump_r)

    price_less_T = twin_win_certificate_price(S, S0, max(T - bump_T, 0.0), sigma, breached, r, q=q, steps=n)
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
    certificate_return_pct = twin_win_certificate_running_return(S0, path) * 100

    _, ax = plt.subplots(figsize=(18, 8))
    ax.plot(index_return_pct.index, index_return_pct.values, color="firebrick", linewidth=1.5,
            label=underlying_name)
    ax.tick_params(axis="y", labelcolor="firebrick")
    ax.plot(certificate_return_pct.index, certificate_return_pct.values, color="indianred", linewidth=1.5,
            linestyle="dashed", label=f"Twin-Win Certificate Participation Tracker ({PUT_QUANTITY:.0f}x below strike)")

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
    breached_dates = path.index[path.cummin() <= breached_level]
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
            f"Participation (Put Quantity): {PUT_QUANTITY:.0f}x\n"
            f"Barrier Percentage: {BARRIER * 100:,.2f}%\n"
            f"{underlying_name} Return: {S_T / S0 - 1:.2%}\n"
            f"Twin-Win Certificate Return: {certificate_return:.2%}\n"
            f"Approximate MtM Fair Value vs. Par (%) at Inception: {fair_value_pct_of_par - 1:.2%}\n"
            f"Realized Vol (ann.): {vol:.2%}\n"
            f"Max Drawdown: {max_drawdown(path):.2%}"
        )
        ax.text(1.06, 0.5, annotation_text, transform=ax.transAxes,
                fontsize=12, fontweight="light", color="black", va="center", ha="left")

    if mtm_vol_term_structure is not None:
        mtm_price = twin_win_certificate_mtm_price_series(S0, path, mtm_vol_term_structure, q=q)
        mtm_gain_over_par_pct = (mtm_price / S0 - 1) * 100

        ax2 = ax.twinx()
        mtm_line, = ax2.plot(
            mtm_gain_over_par_pct.index, mtm_gain_over_par_pct.values, color="darkred", linewidth=1.5,
            linestyle="solid", label="Twin-Win Certificate - Approximate MtM (% of Par)")
        ax2.set_ylabel("MtM Fair Value vs. Par (%)", color="darkred", rotation=270, labelpad=10)
        ax2.tick_params(axis="y", labelcolor="darkred")

        combined_min = min(index_return_pct.min(), certificate_return_pct.min(), mtm_gain_over_par_pct.min(),
                            barrier_pct)
        combined_max = max(index_return_pct.max(), certificate_return_pct.max(), mtm_gain_over_par_pct.max())
        pad = (combined_max - combined_min) * 0.05
        ax.set_ylim(combined_min - pad, combined_max + pad)
        ax2.set_ylim(combined_min - pad, combined_max + pad)

        lines.append(mtm_line)
        labels.append(mtm_line.get_label())

    end_date = index_return_pct.index[-1]
    ax.set_title(f"{underlying_name} Price Return Path from {path.index[0].date()} to {end_date.date()} \n"
                 f"vs Approximate Mark-to-Market (MtM) Value of Twin-Win Certificate "
                 f"({PUT_QUANTITY:.0f}x Downside Participation)")
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
    certificate_return = twin_win_certificate_running_return(S0, path).iloc[-1]
    barrier_touched = bool((path.cummin() <= BARRIER * S0).any())

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
        "Twin-Win Certificate Return": f"{certificate_return:.2%}",
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

    print(f"\nVerifying the QuantLib CRR wiring (barrier pushed unreachable should collapse to the")
    print(f"plain vanilla put struck at the money):")
    check = verify_against_closed_form(S0, TENOR, entry_vol, q=dividend_yield)
    print(f"  Down-and-out put (H unreachable): {check['down_and_out_unreachable']:,.2f}")
    print(f"  Vanilla put (closed form):        {check['vanilla_put']:,.2f}")
    print(f"  Relative difference:              {check['rel_diff']:.3%}  "
          f"({'PASS' if check['rel_diff'] < 0.01 else 'FAIL - investigate before trusting results'})")

    greeks = finite_difference_greeks(S0, TENOR, entry_vol, q=dividend_yield)
    fair_value_pct_of_par = greeks["price"] / S0

    print(f"\nTwin-Win Certificate fair value at inception")
    print(f"(QuantLib: LEPO leg closed-form, {PUT_QUANTITY:.0f}x down-and-out put via a CRR binomial")
    print(f"lattice (barrier checked once per trading day), S=S0, T={TENOR}y, vol={entry_vol:.2%}")
    print(f"interpolated from the {path.index[0].date()} VIX/VIX3M/VIX6M term structure, dividend")
    print(f"yield {dividend_yield:.2%}):")
    print(f"  {fair_value_pct_of_par:.2%} of par ({fair_value_pct_of_par - 1:+.2%} vs. par)")

    print(f"\nTwin-Win Certificate Greeks at inception (finite-difference):")
    print(f"  Delta: {greeks['delta']:.2f}")
    print(f"  Vega:  {greeks['vega'] / S0 * 0.01:.4f}  (per 1% change in vol)")
    print(f"  Rho:   {greeks['rho'] / S0 * 0.01:.4f}  (per 1% change in rates)")
    print(f"  Theta: {greeks['theta'] / S0:.4f} per year / {greeks['theta'] / S0 / 365:.5f} per day")

    print(f"\nMTM Fair Value vs. Par at Inception:")
    print(f"  Price: {greeks['price']:,.2f}  vs. Par (S0): {S0:,.2f}  ({fair_value_pct_of_par:.2%} of par)")

    plot_path(path, S0, underlying_name, mtm_vol_term_structure=vol_term_structure, greeks=greeks, q=dividend_yield)
