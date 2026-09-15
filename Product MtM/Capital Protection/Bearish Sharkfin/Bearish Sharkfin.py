import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql

plt.rcParams["font.family"] = "Arial"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Product terms ---
STRIKE = 1.00                  # long put strike, as a fraction of S0 (100% - at the money)
BARRIER = 0.650                 # down-and-out barrier (H) for the embedded put, H < STRIKE
REBATE = 0.0                   # cash paid immediately if the barrier is touched, in underlying
                                # currency (not a fraction of S0) - not economically important for
                                # MtM purposes here, so set by hand like every other manually-set
                                # term in this repo (STRIKE, BARRIER, etc.); left at 0 by default
RISK_FREE_RATE = 0.04          # SOFR proxy - used for BOTH the ZCB leg and the option leg
GS_CDS_SPREAD = 0.005308       # Goldman Sachs 5y CDS, 53.08 bps - issuer credit spread, ZCB leg only

ENTRY_DATE = "2025-01-02"
TENOR = 1

# TICKER drives dividend handling automatically: any "^"-prefixed Yahoo
# index ticker (e.g. "^GSPC") is treated as paying no dividend (q=0);
# any real stock ticker (e.g. "AAPL", "MCD") gets a real trailing dividend
# yield fetched and applied - see fetch_dividend_yield below. No separate
# flag needed, just set TICKER to whichever kind of underlying you want.
TICKER = "SNDR"
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
    """Plain (no barrier) European put via QuantLib's AnalyticEuropeanEngine
    (dividend yield q is a native input) - used only for the closed-form
    verification check (barrier pushed unreachable -> the down-and-out put
    should collapse to this, at rebate=0)."""
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


# ---------------------------------------------------------------------------
# REPLICATION: Bearish Sharkfin
#            = Long Zero-Coupon Bond (Principal)
#            + Long Down-and-Out Put (struck at STRIKE=100% of S0, barrier
#              BARRIER < STRIKE, checked once per trading day, optional
#              cash rebate paid immediately if the barrier is touched)
#
# The mirror image of the Bullish Sharkfin (Product MtM/Capital Protection/
# Bullish Sharkfin/) - same architecture, flipped onto the downside: a put
# instead of a call, a LOWER barrier instead of an upper one. The ZCB leg
# is what makes this capital-protected: it pays Principal at maturity
# regardless of anything the option leg does, so "if the underlying
# finishes above strike, capital is returned at 100%" falls straight out
# of the put being worthless there - no separate logic needed.
#
# As long as the index never trades down to the barrier, the investor gets
# FULL 1:1 participation in the DOWNSIDE below the strike (the note gains
# value as the index falls) - there is no cap on the put itself, only the
# barrier's all-or-nothing cutoff. Touch the barrier once and the put is
# knocked out for good (the "fin" shape: participation rises as the index
# falls, then drops to flat the instant the barrier is touched) - the note
# then just pays back principal at maturity, plus whatever rebate was
# already received at the moment of the knock-out (not part of the note's
# forward MTM from that point on, since it has already been paid).
#
# The put is priced on a Cox-Ross-Rubinstein binomial lattice (QuantLib
# BinomialCRRBarrierEngine), barrier checked once per lattice step, step
# count = remaining trading days - not the textbook continuous-monitoring
# closed form, which overstates the touch probability relative to what
# daily data can ever confirm. See the Barrier Reverse Convertible
# product's README (Product MtM/Yield/) for the full rationale and lattice
# mechanics; the same reasoning applies here unchanged.
# ---------------------------------------------------------------------------

def down_and_out_put_crr(S, K, H, T, r, sigma, steps, rebate=REBATE, q=0.0):
    """
    Down-and-out put, strike K, barrier H < K, priced on a Cox-Ross-
    Rubinstein binomial lattice with the barrier checked once per step -
    steps should be the number of remaining trading days, so the model's
    monitoring frequency matches the daily data it's priced against.
    Assumes the barrier has NOT already been breached - the caller is
    responsible for pricing this leg at 0 (the rebate has already been
    paid and received, so it isn't part of the note's forward MTM) once
    the barrier has been touched at any point in the note's life so far.
    """
    if S <= H:
        return rebate  # at/through the barrier right now: knocked out this instant, worth exactly the rebate
    if T <= 0:
        return max(K - S, 0.0)

    today = ql.Date(1, 1, 2000)
    ql.Settings.instance().evaluationDate = today
    process = _quantlib_process(S, r, q, sigma, today)

    days = max(int(round(T * 365.25)), 1)
    exercise = ql.EuropeanExercise(today + ql.Period(days, ql.Days))
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    option = ql.BarrierOption(ql.Barrier.DownOut, H, rebate, payoff, exercise)

    n = max(int(round(steps)), 2)  # QuantLib's binomial barrier engine errors below 2 steps
    option.setPricingEngine(ql.BinomialCRRBarrierEngine(process, n, n))
    return option.NPV()


def bearish_sharkfin_price(S, S0, T_remaining, sigma, breached, r=RISK_FREE_RATE,
                            credit_spread=GS_CDS_SPREAD, steps=None, q=0.0):
    """
    Once breached=True, the down-and-out put has permanently knocked out
    and contributes 0 to the note's forward value (any rebate was already
    received historically, not still owed) - the note is worth exactly
    the ZCB from then on, i.e. capital returned at 100%.

    `steps` is the CRR lattice's step count (one barrier check per step) -
    pass the actual number of remaining trading days when pricing off a
    real historical path so the model's monitoring frequency matches the
    data; defaults to an approximate 252-trading-day year if omitted.
    `q` is the underlying's dividend yield (0 for an index).
    """
    zcb = zcb_price_and_greeks(S0, T_remaining, r + credit_spread)

    if breached:
        put_price = 0.0
    else:
        K = STRIKE * S0
        H = BARRIER * S0
        n = steps if steps is not None else max(round(T_remaining * 252), 1)
        put_price = down_and_out_put_crr(S, K, H, T_remaining, r, sigma, n, REBATE, q)

    return zcb["price"] + put_price


def verify_against_closed_form(S0, T, sigma, r=RISK_FREE_RATE, q=0.0, steps=None):
    """
    Barrier pushed unreachable (H -> ~0), rebate forced to 0: the
    down-and-out put can never knock out, so its CRR price should converge
    onto the plain vanilla put struck at STRIKE - confirms the barrier
    engine collapses to the ordinary closed form in the no-barrier limit,
    the same style of check used for the Bullish Sharkfin's up-and-out
    call and the Bonus Certificate's down-and-out put.
    """
    K = STRIKE * S0
    n = steps if steps is not None else max(round(T * 252), 1)

    down_and_out_unreachable = down_and_out_put_crr(S0, K, H=1e-6, T=T, r=r, sigma=sigma, steps=n, rebate=0.0, q=q)
    vanilla_put = black_scholes_put(S0, K, T, r, sigma, q)["price"]
    rel_diff = abs(down_and_out_unreachable - vanilla_put) / vanilla_put

    return {"down_and_out_unreachable": down_and_out_unreachable, "vanilla_put": vanilla_put, "rel_diff": rel_diff}


def bearish_sharkfin_running_return(S0, path):
    """
    Participation tracker: the terminal payoff FORMULA applied to today's
    spot, as a return over par (this is directly "how much downside
    participation has this path locked in so far"). NOT what you'd
    actually receive if the note were sold or unwound today - it ignores
    all remaining time value in the still-live put, unlike the MTM
    fair-value line, which is the closest thing to an actual today's-value
    estimate. If the barrier has never been touched up to today, this is
    the put's intrinsic value, max(Strike - S, 0) - full downside
    participation. Once the barrier IS touched, the put is permanently
    dead - 0% here for every subsequent date, meaning exactly par (100%
    capital returned), regardless of where the index goes afterward.

    Breach is "conducted on close" - checked against the daily CLOSE, the
    same price series used for valuation, matching the CRR forward-pricing
    lattice's own once-per-trading-day observation frequency. A day that
    dips through the barrier intraday and closes back above it does NOT
    knock the put out.
    """
    strike_level = STRIKE * S0
    barrier_level = BARRIER * S0
    breached_so_far = path.cummin() <= barrier_level

    put_payoff = np.maximum(strike_level - path.values, 0.0)
    put_payoff = np.where(breached_so_far, 0.0, put_payoff)
    return pd.Series(put_payoff, index=path.index) / S0


def bearish_sharkfin_mtm_price_series(S0, path, vol_term_structure, r=RISK_FREE_RATE,
                                       credit_spread=GS_CDS_SPREAD, q=0.0):
    maturity_date = path.index[-1]
    barrier_level = BARRIER * S0
    breached_so_far = path.cummin() <= barrier_level

    n_dates = len(path)
    prices = []
    for i, (date, level) in enumerate(path.items()):
        T_remaining = (maturity_date - date).days / 365.25
        steps_remaining = max(n_dates - 1 - i, 1)  # one CRR barrier check per remaining trading day in the actual path
        vols_today = {tenor: series[date] for tenor, series in vol_term_structure.items()}
        sigma = interpolate_implied_vol(vols_today, max(T_remaining, 0.0))
        prices.append(
            bearish_sharkfin_price(level, S0, T_remaining, sigma, breached_so_far[date], r, credit_spread,
                                    steps=steps_remaining, q=q)
        )
    return pd.Series(prices, index=path.index)


def finite_difference_greeks_at(S, S0, T, sigma, breached, r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD,
                                 steps=None, q=0.0):
    """
    Price and Greeks via central finite differences on the FULL note price
    (not by differentiating the barrier formula further) - barrier-option
    Greeks are notoriously messy near the barrier (delta in particular can
    be discontinuous), so a numerical bump is simpler and more robust, same
    approach as every other barrier product in this repo.

    The CRR step count is fixed ONCE (from the un-bumped T) and reused for
    every bumped evaluation, including the T-bump for theta - otherwise a
    tiny T bump could round to a different integer step count and inject
    lattice-discreteness noise into the Greek instead of the actual
    sensitivity.

    bump_S and bump_sigma are much wider than the "0.1% of spot" convention
    used for the closed-form (non-barrier) products in this repo. A CRR
    lattice is rebuilt from scratch on every call, and its node grid scales
    multiplicatively with both S and sigma, so a small bump can land
    entirely inside a discrete "sawtooth" lattice artifact and return a
    Greek off by a large factor - see the Bullish Sharkfin product's
    README (same folder) for the specific bump-size scan that surfaced
    this. Rho keeps a small bump since r doesn't enter the lattice's node
    spacing at all, only drift/discounting.
    """
    bump_S, bump_sigma, bump_r, bump_T = S0 * 0.02, 0.02, 0.0001, 21 / 365
    n = steps if steps is not None else max(round(T * 252), 1)

    price = bearish_sharkfin_price(S, S0, T, sigma, breached, r, credit_spread, steps=n, q=q)

    price_up_S = bearish_sharkfin_price(S + bump_S, S0, T, sigma, breached, r, credit_spread, steps=n, q=q)
    price_down_S = bearish_sharkfin_price(S - bump_S, S0, T, sigma, breached, r, credit_spread, steps=n, q=q)
    delta = (price_up_S - price_down_S) / (2 * bump_S)

    price_up_sigma = bearish_sharkfin_price(S, S0, T, sigma + bump_sigma, breached, r, credit_spread, steps=n, q=q)
    price_down_sigma = bearish_sharkfin_price(S, S0, T, sigma - bump_sigma, breached, r, credit_spread, steps=n, q=q)
    vega = (price_up_sigma - price_down_sigma) / (2 * bump_sigma)

    price_up_r = bearish_sharkfin_price(S, S0, T, sigma, breached, r + bump_r, credit_spread, steps=n, q=q)
    price_down_r = bearish_sharkfin_price(S, S0, T, sigma, breached, r - bump_r, credit_spread, steps=n, q=q)
    rho = (price_up_r - price_down_r) / (2 * bump_r)

    price_less_T = bearish_sharkfin_price(S, S0, max(T - bump_T, 0.0), sigma, breached, r, credit_spread, steps=n, q=q)
    theta = (price_less_T - price) / bump_T

    return {"price": price, "delta": delta, "vega": vega, "rho": rho, "theta": theta}


def finite_difference_greeks(S0, T, sigma, r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD, q=0.0):
    return finite_difference_greeks_at(S0, S0, T, sigma, False, r, credit_spread, q=q)


def realized_annualized_vol(path):
    log_returns = np.log(path / path.shift(1)).dropna()
    return log_returns.std() * np.sqrt(252)


def max_drawdown(path):
    running_max = path.cummax()
    drawdown = path / running_max - 1
    return drawdown.min()


def plot_path(path, S0, underlying_name, mtm_vol_term_structure=None, greeks=None, q=0.0):
    index_return_pct = (path / S0 - 1) * 100
    sharkfin_return_pct = bearish_sharkfin_running_return(S0, path) * 100

    _, ax = plt.subplots(figsize=(18, 8))
    ax.plot(index_return_pct.index, index_return_pct.values, color="firebrick", linewidth=1.5,
            label=underlying_name)
    ax.tick_params(axis="y", labelcolor="firebrick")
    ax.plot(sharkfin_return_pct.index, sharkfin_return_pct.values, color="indianred", linewidth=1.5,
            linestyle="dashed", label="Bearish Sharkfin Participation Tracker")

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
        mtm_price = bearish_sharkfin_mtm_price_series(S0, path, mtm_vol_term_structure, q=q)
        mtm_gain_over_par_pct = (mtm_price / S0 - 1) * 100

        ax2 = ax.twinx()
        mtm_line, = ax2.plot(
            mtm_gain_over_par_pct.index, mtm_gain_over_par_pct.values, color="darkred", linewidth=1.5,
            linestyle="solid", label="Bearish Sharkfin - Approximate MtM (% of Par)")
        ax2.set_ylabel("MtM Fair Value vs. Par (%)", color="darkred", rotation=270, labelpad=10)
        ax2.tick_params(axis="y", labelcolor="darkred")

        combined_min = min(index_return_pct.min(), sharkfin_return_pct.min(), mtm_gain_over_par_pct.min(), barrier_pct)
        combined_max = max(index_return_pct.max(), sharkfin_return_pct.max(), mtm_gain_over_par_pct.max(), strike_pct)
        pad = (combined_max - combined_min) * 0.05
        ax.set_ylim(combined_min - pad, combined_max + pad)
        ax2.set_ylim(combined_min - pad, combined_max + pad)

        lines.append(mtm_line)
        labels.append(mtm_line.get_label())

    end_date = index_return_pct.index[-1]
    ax.set_title(f"{underlying_name} Price Return Path from {path.index[0].date()} to {end_date.date()} \n"
                 f"vs Approximate Mark-to-Market (MtM) Value of Bearish Sharkfin")
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
    sharkfin_return = bearish_sharkfin_running_return(S0, path).iloc[-1]
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
        "Bearish Sharkfin Return": f"{sharkfin_return:.2%}",
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

    print(f"\nBearish Sharkfin fair value at inception")
    print(f"(ZCB discounted at SOFR {RISK_FREE_RATE:.2%} + Goldman Sachs CDS {GS_CDS_SPREAD:.2%}, long")
    print(f"down-and-out put priced at SOFR alone via QuantLib's CRR binomial lattice (barrier checked")
    print(f"once per trading day, rebate={REBATE}, dividend yield {dividend_yield:.2%}), T={TENOR}y,")
    print(f"vol={entry_vol:.2%} interpolated from the {path.index[0].date()} VIX/VIX3M/VIX6M term")
    print(f"structure, strike={STRIKE:.0%}, barrier={BARRIER:.0%} of S0={S0:,.2f}):")
    print(f"  {fair_value_pct_of_par:.2%} of par ({fair_value_pct_of_par - 1:+.2%} vs. par)")

    print(f"\nBearish Sharkfin Greeks at inception (finite-difference):")
    print(f"  Delta: {greeks['delta']:.2f}")
    print(f"  Vega:  {greeks['vega'] / S0 * 0.01:.4f}  (per 1% change in vol)")
    print(f"  Rho:   {greeks['rho'] / S0 * 0.01:.4f}  (per 1% change in SOFR)")
    print(f"  Theta: {greeks['theta'] / S0:.4f} per year / {greeks['theta'] / S0 / 365:.5f} per day")

    print(f"\nMTM Fair Value vs. Par at Inception:")
    print(f"  Price: {greeks['price']:,.2f}  vs. Par (S0): {S0:,.2f}  ({fair_value_pct_of_par:.2%} of par)")

    plot_path(path, S0, underlying_name, mtm_vol_term_structure=vol_term_structure, greeks=greeks, q=dividend_yield)
