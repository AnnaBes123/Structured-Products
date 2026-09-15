import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql

plt.rcParams["font.family"] = "Arial"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Product terms (same base terms as the plain "Reverse Convertible.py", one
# folder up, plus the worst-of basket feature) ---
STRIKE = 0.90                  # worst-of put strike, as a fraction of each name's OWN entry level
RISK_FREE_RATE = 0.04          # SOFR proxy - used for BOTH the ZCB leg and the option leg
GS_CDS_SPREAD = 0.005308       # Goldman Sachs 5y CDS, 53.08 bps - issuer credit spread, ZCB leg only

ENTRY_DATE = "2025-01-02"
TENOR = 1

# Three names from three different sectors on purpose - the whole point of a
# worst-of note is that dispersion (low correlation) across the basket makes
# the note WORSE for the investor than any single-name RC at the same
# strike, since there are more independent chances for "the worst name" to
# breach. Same-sector, highly-correlated names would understate that effect.
TICKERS = ["AAPL", "JPM", "XOM"]

# Vol AND correlation are both estimated from REAL trailing daily closes -
# not fabricated, not implied vol (no free per-name options data source is
# wired up anywhere in this repo - see the README for the single-name RC's
# own SPX-vol-as-proxy simplification, which doesn't generalize cleanly to
# three unrelated names either). CORRELATION_LOOKBACK_YEARS is a TRAILING
# window ending at ENTRY_DATE (data an investor actually had on the pricing
# date) - not the backtest window itself, which would leak forward-looking
# information into a day-1 price.
CORRELATION_LOOKBACK_YEARS = 2

N_MC_PATHS = 50000
MC_SEED = 42

# Common market-wide shock applied to every name at once - see the
# "Greeks ladder" section of the README for why this is the natural x-axis
# for a multi-asset product (as opposed to bumping one name at a time).
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
    ("^" tickers) are treated as paying none. See the single-name RC's
    README (one folder up) for the full rationale and field-selection
    notes - identical mechanism, just called once per basket name here."""
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


def fetch_multi_asset_path(tickers, start, end):
    """
    Daily close series for every basket name, aligned on the trading dates
    ALL of them share (inner join) - a US-listed multi-name basket should
    have near-identical calendars, but this guards against any one name's
    occasional missing print instead of silently misaligning dates.
    """
    series = {ticker: fetch_daily_closes(ticker, start, end) for ticker in tickers}
    df = pd.concat(series, axis=1)
    return df.dropna(how="any")


def estimate_vols_and_correlation(calibration_price_df):
    """
    Realized annualized vol per name AND the full correlation matrix,
    estimated from the SAME real trailing daily-close history (log
    returns) - both are genuinely sourced numbers, not assumptions, same
    standard as every other real-data input in this repo. See
    CORRELATION_LOOKBACK_YEARS above for why this window ends at
    ENTRY_DATE rather than overlapping the backtest itself.
    """
    log_returns = np.log(calibration_price_df / calibration_price_df.shift(1)).dropna()
    vols = log_returns.std() * np.sqrt(252)
    corr = log_returns.corr()
    return vols.to_dict(), corr


def realized_annualized_vol(path):
    log_returns = np.log(path / path.shift(1)).dropna()
    return log_returns.std() * np.sqrt(252)


def max_drawdown(path):
    running_max = path.cummax()
    drawdown = path / running_max - 1
    return drawdown.min()


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
    """Plain European put via QuantLib's AnalyticEuropeanEngine - the N=1
    degenerate case of the worst-of basket, used only in
    verify_against_closed_form."""
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


# ---------------------------------------------------------------------------
# REPLICATION: Multi-RC (Worst-of Reverse Convertible)
#            = Long Zero-Coupon Bond (Principal)
#            - Short Put on the WORST-OF RELATIVE PERFORMANCE across the
#              basket (struck at STRIKE, quantity 1/STRIKE)
#
# "Worst-of relative performance" at any date = min_i(S_i / S0_i) - each
# name normalized by its OWN entry level, not compared in raw dollar terms
# (comparing a $50 stock's price to a $500 stock's would be meaningless).
# Below STRIKE, the note converts into Principal/Strike units of WHICHEVER
# NAME is currently the worst performer - the standard real-world
# convention for worst-of reverse-convertible-style notes, and the direct
# N-asset generalization of the single-name RC's own conversion-ratio
# logic (see that product's README for the 1/Strike derivation).
#
# PRICED NATIVELY VIA QUANTLIB'S OWN BASKET-OPTION MACHINERY, not a
# hand-rolled Monte Carlo:
#   - Each name's process is a BlackScholesMertonProcess, spot normalized
#     to RELATIVE performance (1.0 at entry, or S_i(t)/S0_i at a later
#     MTM date) - a driftless-in-log rescaling that works because a
#     continuous-dividend GBM's relative-performance process obeys the
#     exact same SDE (same vol, same q) regardless of the dollar level
#     it's expressed in.
#   - ql.StochasticProcessArray bundles the per-name processes together
#     with the full correlation matrix - this is where correlation
#     actually enters the model (QuantLib handles the Cholesky-style
#     transform internally; nothing hand-rolled here).
#   - ql.MinBasketPayoff wraps a plain vanilla put payoff so it's applied
#     to the MINIMUM of the (relative-performance) basket, not any single
#     name in isolation - this is what makes it "worst-of."
#   - ql.MCEuropeanBasketEngine prices it - QuantLib has no closed-form
#     engine for baskets of more than 2 names, so Monte Carlo is the live
#     pricer for any basket size. For exactly 2 names, QuantLib DOES have
#     a closed-form engine (ql.StulzEngine, the Stulz 1982 bivariate
#     model) - used here only as a convergence BENCHMARK in
#     verify_against_closed_form, the same "closed form as a sanity
#     check, not the live pricer" pattern as the Barrier Reverse
#     Convertible's CRR-vs-AnalyticBarrierEngine check.
# ---------------------------------------------------------------------------

def worst_of_put_price(relative_spots, K, T, r, sigmas, qs, corr_matrix, n_paths=N_MC_PATHS, seed=MC_SEED):
    """
    MC price of a worst-of put on relative performance, via QuantLib's
    native basket-option engine (StochasticProcessArray + MinBasketPayoff
    + MCEuropeanBasketEngine) - not a hand-rolled correlated simulation.
    `relative_spots[i]` is name i's CURRENT performance relative to ITS
    OWN entry level (1.0 at inception; S_i(t)/S0_i at a later MTM date).
    """
    n = len(relative_spots)
    today = ql.Date(1, 1, 2000)
    ql.Settings.instance().evaluationDate = today

    processes = [_quantlib_process(relative_spots[i], r, qs[i], sigmas[i], today) for i in range(n)]

    corr = ql.Matrix(n, n)
    for i in range(n):
        for j in range(n):
            corr[i][j] = float(corr_matrix[i][j])

    process_array = ql.StochasticProcessArray(processes, corr)
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    basket_payoff = ql.MinBasketPayoff(payoff)

    days = max(int(round(T * 365.25)), 1)
    exercise = ql.EuropeanExercise(today + ql.Period(days, ql.Days))
    option = ql.BasketOption(basket_payoff, exercise)

    engine = ql.MCEuropeanBasketEngine(process_array, "PseudoRandom", timeStepsPerYear=1,
                                        requiredSamples=n_paths, seed=seed)
    option.setPricingEngine(engine)
    return option.NPV()


def stulz_put_price(relative_spot_1, relative_spot_2, K, T, r, sigma_1, sigma_2, q_1, q_2, rho):
    """Closed-form (Stulz 1982) 2-asset worst-of put via QuantLib's own
    StulzEngine - exact, no Monte Carlo noise. Used ONLY as a convergence
    benchmark for worst_of_put_price at N=2, never as the live pricer for
    an arbitrary-size basket."""
    today = ql.Date(1, 1, 2000)
    ql.Settings.instance().evaluationDate = today
    p1 = _quantlib_process(relative_spot_1, r, q_1, sigma_1, today)
    p2 = _quantlib_process(relative_spot_2, r, q_2, sigma_2, today)

    payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    basket_payoff = ql.MinBasketPayoff(payoff)
    days = max(int(round(T * 365.25)), 1)
    exercise = ql.EuropeanExercise(today + ql.Period(days, ql.Days))
    option = ql.BasketOption(basket_payoff, exercise)
    option.setPricingEngine(ql.StulzEngine(p1, p2, rho))
    return option.NPV()


def multi_rc_price(relative_spots, T_remaining, sigmas, qs, corr_matrix, r=RISK_FREE_RATE,
                     credit_spread=GS_CDS_SPREAD, n_paths=N_MC_PATHS, seed=MC_SEED):
    """
    Principal = 1.0 ("PAR" units) throughout - there's no single dollar S0
    to anchor to with more than one underlying, so everything is carried
    in fractions of par instead, same as every "% of par" figure printed
    elsewhere in this repo, just made the primary unit here rather than a
    display-time conversion.
    """
    zcb = zcb_price_and_greeks(1.0, T_remaining, r + credit_spread)
    put_quantity = 1.0 / STRIKE

    if T_remaining <= 0:
        worst_of = min(relative_spots)
        put_payoff = max(STRIKE - worst_of, 0.0)
        return {"price": zcb["price"] - put_quantity * put_payoff, "worst_of": worst_of}

    put_price = worst_of_put_price(relative_spots, STRIKE, T_remaining, r, sigmas, qs, corr_matrix, n_paths, seed)
    return {"price": zcb["price"] - put_quantity * put_price, "worst_of": min(relative_spots)}


def verify_against_closed_form(sigmas, qs, corr_matrix, T, r=RISK_FREE_RATE, n_paths=N_MC_PATHS):
    """
    Two independent checks that the QuantLib basket-option wiring is
    correct, before trusting it for the actual basket in TICKERS:

    1. N=1 degenerate case: a "basket" of exactly one name must reduce to
       an ordinary vanilla put (QuantLib's AnalyticEuropeanEngine) - the
       MinBasketPayoff of a single asset is trivially just that asset.
    2. N=2 convergence: MCEuropeanBasketEngine (the general, arbitrary-N
       live pricer) should converge onto ql.StulzEngine's exact
       closed-form price as the path count grows, for the standard
       2-asset worst-of case - confirms StochasticProcessArray's
       correlation handling and MinBasketPayoff are wired correctly.
    """
    single_relative = [1.0]
    mc_single = worst_of_put_price(single_relative, STRIKE, T, r, sigmas[:1], qs[:1],
                                     np.array([[1.0]]), n_paths=n_paths, seed=MC_SEED)
    vanilla_single = black_scholes_put(1.0, STRIKE, T, r, sigmas[0], qs[0])["price"]
    n1_rel_diff = abs(mc_single - vanilla_single) / vanilla_single

    two_relative = [1.0, 1.0]
    rho_12 = float(corr_matrix[0][1])
    mc_two = worst_of_put_price(two_relative, STRIKE, T, r, sigmas[:2], qs[:2],
                                  np.array(corr_matrix)[:2, :2], n_paths=200_000, seed=MC_SEED)
    stulz_two = stulz_put_price(1.0, 1.0, STRIKE, T, r, sigmas[0], sigmas[1], qs[0], qs[1], rho_12)
    n2_rel_diff = abs(mc_two - stulz_two) / stulz_two

    return {
        "mc_single": mc_single, "vanilla_single": vanilla_single, "n1_rel_diff": n1_rel_diff,
        "mc_two": mc_two, "stulz_two": stulz_two, "n2_rel_diff": n2_rel_diff,
    }


def worst_of_running_return(S0_list, price_df):
    """
    Participation Tracker: the terminal payoff FORMULA applied to each
    day's relative-performance basket - full principal back if the WORST
    performer (relative to its own entry level) is at or above STRIKE;
    below it, value = Principal * (worst_of / STRIKE), exactly the
    single-name RC's conversion-ratio formula applied to whichever name
    is currently worst. This is NOT what you'd actually receive if the
    note were sold or unwound today - it ignores all remaining time value
    in the still-live worst-of put. See the MTM line for the actual
    fair-value estimate.
    """
    relative = price_df / pd.Series(S0_list, index=price_df.columns)
    worst_of = relative.min(axis=1)
    running = pd.Series(0.0, index=price_df.index)
    below_strike = worst_of < STRIKE
    running[below_strike] = worst_of[below_strike] / STRIKE - 1.0
    return running, worst_of


def multi_rc_mtm_price_series(S0_list, price_df, sigmas, qs, corr_matrix, r=RISK_FREE_RATE,
                                credit_spread=GS_CDS_SPREAD, n_paths=N_MC_PATHS):
    maturity_date = price_df.index[-1]
    relative_path = price_df / pd.Series(S0_list, index=price_df.columns)

    prices = []
    for date, row in relative_path.iterrows():
        T_remaining = (maturity_date - date).days / 365.25
        result = multi_rc_price(row.values.tolist(), T_remaining, sigmas, qs, corr_matrix, r, credit_spread, n_paths)
        prices.append(result["price"])
    return pd.Series(prices, index=price_df.index)


def finite_difference_greeks(relative_spots, T, sigmas, qs, corr_matrix, r=RISK_FREE_RATE,
                              credit_spread=GS_CDS_SPREAD, n_paths=N_MC_PATHS, seed=MC_SEED):
    """
    Bump-and-reprice Greeks, one Delta and one Vega PER NAME (a multi-asset
    product's risk is genuinely a vector, not a scalar - an issuer hedging
    this note needs to know its exposure to each underlying separately),
    plus a single Rho and Theta (shared funding rate / time, same as every
    other product here). Common random numbers (same MC_SEED reused for
    every bumped evaluation) keep the finite differences from being
    swamped by independent Monte Carlo noise - confirmed stable down to a
    ~0.1% bump in a direct bump-size scan (see the README), so the modest
    1%-of-relative-spot / 0.5-vol-point bumps here (much tighter than the
    CRR-lattice products' 2%/2-vol-point convention) are already
    well within the stable region - this is a smooth MC price, not a
    lattice with discrete sawtooth artifacts.
    """
    n = len(relative_spots)
    bump_S, bump_sigma, bump_r, bump_T = 0.01, 0.005, 0.0001, 7 / 365

    base = multi_rc_price(relative_spots, T, sigmas, qs, corr_matrix, r, credit_spread, n_paths, seed)

    deltas = []
    for i in range(n):
        up = list(relative_spots); up[i] += bump_S
        down = list(relative_spots); down[i] -= bump_S
        price_up = multi_rc_price(up, T, sigmas, qs, corr_matrix, r, credit_spread, n_paths, seed)["price"]
        price_down = multi_rc_price(down, T, sigmas, qs, corr_matrix, r, credit_spread, n_paths, seed)["price"]
        deltas.append((price_up - price_down) / (2 * bump_S))

    vegas = []
    for i in range(n):
        sig_up = list(sigmas); sig_up[i] += bump_sigma
        sig_down = list(sigmas); sig_down[i] -= bump_sigma
        price_up = multi_rc_price(relative_spots, T, sig_up, qs, corr_matrix, r, credit_spread, n_paths, seed)["price"]
        price_down = multi_rc_price(relative_spots, T, sig_down, qs, corr_matrix, r, credit_spread, n_paths, seed)["price"]
        vegas.append((price_up - price_down) / (2 * bump_sigma))

    price_up_r = multi_rc_price(relative_spots, T, sigmas, qs, corr_matrix, r + bump_r, credit_spread, n_paths, seed)["price"]
    price_down_r = multi_rc_price(relative_spots, T, sigmas, qs, corr_matrix, r - bump_r, credit_spread, n_paths, seed)["price"]
    rho = (price_up_r - price_down_r) / (2 * bump_r)

    price_less_T = multi_rc_price(relative_spots, max(T - bump_T, 0.0), sigmas, qs, corr_matrix, r, credit_spread, n_paths, seed)["price"]
    theta = (price_less_T - base["price"]) / bump_T

    return {"price": base["price"], "deltas": deltas, "vegas": vegas, "rho": rho, "theta": theta}


def plot_path(price_df, S0_list, tickers, underlying_names, mtm_enabled=True, sigmas=None, qs=None,
              corr_matrix=None, greeks=None):
    index_return_pct = (price_df / pd.Series(S0_list, index=price_df.columns) - 1) * 100
    tracker, worst_of = worst_of_running_return(S0_list, price_df)
    tracker_pct = tracker * 100

    _, ax = plt.subplots(figsize=(18, 8))
    asset_colors = ["steelblue", "darkorange", "seagreen", "mediumorchid", "peru", "slategray"]
    for i, ticker in enumerate(tickers):
        ax.plot(index_return_pct.index, index_return_pct[ticker].values, color=asset_colors[i % len(asset_colors)],
                linewidth=1.1, alpha=0.85, label=underlying_names[i])

    ax.plot(tracker_pct.index, tracker_pct.values, color="indianred", linewidth=1.8,
            linestyle="dashed", label="Multi-RC Participation Tracker (worst-of)")

    ax.grid(True, which="major", color="lightgrey", linewidth=0.6)
    ax.axhline(0, color="lightgrey", linewidth=0.8)
    strike_pct = (STRIKE - 1) * 100
    ax.axhline(strike_pct, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.text(0.01, strike_pct, f"Strike: {strike_pct:.1f}%", transform=ax.get_yaxis_transform(),
            color="dodgerblue", fontsize=9, va="bottom", ha="left")

    ax.set_xlabel("Date")
    ax.set_ylabel("Return from Entry (%)")
    lines, labels = ax.get_legend_handles_labels()
    ax.margins(x=0, y=0.05)

    if greeks is not None:
        delta_lines = "\n".join(f"  {t}: {d:.3f}" for t, d in zip(tickers, greeks["deltas"]))
        vega_lines = "\n".join(f"  {t}: {v:.4f}" for t, v in zip(tickers, greeks["vegas"]))
        greeks_text = (
            "Greeks at Inception:\n"
            f"Δ (Delta) per name:\n{delta_lines}\n"
            f"ν (Vega) per name (per 1% vol):\n{vega_lines}\n"
            f"ρ (Rho): {greeks['rho'] * 0.01:.4f} per 1% change in SOFR\n"
            f"θ (Theta): {greeks['theta']:.4f} per year"
        )
        ax.text(1.16, 0.5, greeks_text, transform=ax.transAxes,
                fontsize=9, fontweight="light", color="black", va="center", ha="left")

    combined_min = min(index_return_pct.min().min(), tracker_pct.min(), strike_pct)
    combined_max = max(index_return_pct.max().max(), tracker_pct.max())

    if mtm_enabled:
        mtm_price = multi_rc_mtm_price_series(S0_list, price_df, sigmas, qs, corr_matrix)
        mtm_gain_over_par_pct = (mtm_price - 1.0) * 100

        ax2 = ax.twinx()
        mtm_line, = ax2.plot(
            mtm_gain_over_par_pct.index, mtm_gain_over_par_pct.values, color="darkred", linewidth=1.8,
            linestyle="solid", label="Multi-RC - Approximate MtM (% of Par, QuantLib basket MC)")
        ax2.set_ylabel("MtM Fair Value vs. Par (%)", color="darkred", rotation=270, labelpad=10)
        ax2.tick_params(axis="y", labelcolor="darkred")

        combined_min = min(combined_min, mtm_gain_over_par_pct.min())
        combined_max = max(combined_max, mtm_gain_over_par_pct.max())
        pad = (combined_max - combined_min) * 0.05
        ax.set_ylim(combined_min - pad, combined_max + pad)
        ax2.set_ylim(combined_min - pad, combined_max + pad)

        lines.append(mtm_line)
        labels.append(mtm_line.get_label())
    else:
        pad = (combined_max - combined_min) * 0.05
        ax.set_ylim(combined_min - pad, combined_max + pad)

    basket_label = " / ".join(underlying_names)
    end_date = index_return_pct.index[-1]
    ax.set_title(f"Worst-of Basket ({basket_label}) Price Return Path from {price_df.index[0].date()} to {end_date.date()} \n"
                 f"vs Approximate Mark-to-Market (MtM) Value of Multi-RC")
    ax.legend(lines, labels, loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    underlying_names = [fetch_underlying_name(t) for t in TICKERS]
    print(f"Basket: {', '.join(f'{n} ({t})' for n, t in zip(underlying_names, TICKERS))}")

    entry_ts = pd.Timestamp(ENTRY_DATE)
    maturity_ts = entry_ts + pd.Timedelta(days=round(TENOR * 365.25))
    calibration_start = entry_ts - pd.Timedelta(days=round(CORRELATION_LOOKBACK_YEARS * 365.25))

    print(f"\nFetching {CORRELATION_LOOKBACK_YEARS}y TRAILING history (ending {ENTRY_DATE}, before the note")
    print(f"starts) to estimate vol and correlation from real data - no look-ahead into the backtest window...")
    calibration_df = fetch_multi_asset_path(TICKERS, calibration_start, entry_ts)
    vols_dict, corr_df = estimate_vols_and_correlation(calibration_df)
    sigmas = [vols_dict[t] for t in TICKERS]
    corr_matrix = corr_df.loc[TICKERS, TICKERS].values

    print(f"\nRealized annualized vol (from the {CORRELATION_LOOKBACK_YEARS}y trailing window):")
    for t, n, s in zip(TICKERS, underlying_names, sigmas):
        print(f"  {n} ({t}): {s:.2%}")

    print(f"\nRealized correlation matrix (from the same trailing window):")
    print(corr_df.loc[TICKERS, TICKERS].to_string(float_format=lambda x: f"{x:.3f}"))

    dividend_yields = [fetch_dividend_yield(t) for t in TICKERS]
    print(f"\nDividend yields (flat, continuous):")
    for t, n, q in zip(TICKERS, underlying_names, dividend_yields):
        print(f"  {n} ({t}): {q:.2%}")

    print(f"\nFetching {ENTRY_DATE} (entry) to +{TENOR}y (maturity) backtest path...")
    price_df = fetch_multi_asset_path(TICKERS, entry_ts, maturity_ts)
    S0_list = [float(price_df[t].iloc[0]) for t in TICKERS]
    S_T_list = [float(price_df[t].iloc[-1]) for t in TICKERS]

    print(f"\nVerifying the QuantLib basket-option wiring (N=1 reduces to a plain vanilla put; N=2")
    print(f"converges onto QuantLib's closed-form StulzEngine as path count grows):")
    check = verify_against_closed_form(sigmas, dividend_yields, corr_matrix, TENOR)
    print(f"  N=1: MC basket price={check['mc_single']:.5f}  vanilla put (closed form)={check['vanilla_single']:.5f}"
          f"  rel diff={check['n1_rel_diff']:.3%}  "
          f"({'PASS' if check['n1_rel_diff'] < 0.01 else 'FAIL - investigate before trusting results'})")
    print(f"  N=2: MC basket (200k paths)={check['mc_two']:.5f}  Stulz closed form={check['stulz_two']:.5f}"
          f"  rel diff={check['n2_rel_diff']:.3%}  "
          f"({'PASS' if check['n2_rel_diff'] < 0.02 else 'FAIL - investigate before trusting results'})")

    tracker, worst_of = worst_of_running_return(S0_list, price_df)
    note_return = tracker.iloc[-1]
    worst_performer_idx = int(np.argmin([S_T_list[i] / S0_list[i] for i in range(len(TICKERS))]))

    summary_rows = {
        "Entry Date": price_df.index[0].date(),
        "Maturity Date": price_df.index[-1].date(),
        "Strike": f"{STRIKE:.0%} of each name's own entry level",
    }
    for t, n, s0, st in zip(TICKERS, underlying_names, S0_list, S_T_list):
        summary_rows[f"{n} ({t}) Return"] = f"{st / s0 - 1:+.2%}"
    summary_rows["Worst Performer"] = f"{underlying_names[worst_performer_idx]} ({TICKERS[worst_performer_idx]})"
    summary_rows["Worst-of Relative Performance (T)"] = f"{worst_of.iloc[-1]:.2%}"
    summary_rows["Multi-RC Return"] = f"{note_return:.2%}"

    print("\n" + pd.Series(summary_rows).to_string())

    greeks = finite_difference_greeks([1.0] * len(TICKERS), TENOR, sigmas, dividend_yields, corr_matrix)
    fair_value_pct_of_par = greeks["price"]

    print(f"\nMulti-RC fair value at inception")
    print(f"(QuantLib MCEuropeanBasketEngine, {N_MC_PATHS:,} paths, ZCB discounted at SOFR")
    print(f"{RISK_FREE_RATE:.2%} + Goldman Sachs CDS {GS_CDS_SPREAD:.2%}, worst-of put at SOFR alone,")
    print(f"T={TENOR}y, strike={STRIKE:.0%}):")
    print(f"  {fair_value_pct_of_par:.2%} of par ({fair_value_pct_of_par - 1:+.2%} vs. par)")

    for t, n, d, v in zip(TICKERS, underlying_names, greeks["deltas"], greeks["vegas"]):
        print(f"  Delta ({n}): {d:.3f}   Vega ({n}, per 1% vol): {v * 0.01:.4f}")
    print(f"  Rho: {greeks['rho'] * 0.01:.4f}  (per 1% change in SOFR)")
    print(f"  Theta: {greeks['theta']:.4f} per year / {greeks['theta'] / 365:.5f} per day")

    print(f"\nFor comparison, a SINGLE-name Reverse Convertible at the same {STRIKE:.0%} strike on just the worst")
    print(f"performer alone would be worth (closed form, ignoring the other two names entirely):")
    worst_idx = int(np.argmin(sigmas))  # informative only - not used in pricing
    single_name_zcb = zcb_price_and_greeks(1.0, TENOR, RISK_FREE_RATE + GS_CDS_SPREAD)
    for t, n, s, q in zip(TICKERS, underlying_names, sigmas, dividend_yields):
        single_put = black_scholes_put(1.0, STRIKE, TENOR, RISK_FREE_RATE, s, q)["price"]
        single_price = single_name_zcb["price"] - (1.0 / STRIKE) * single_put
        print(f"  {n} alone: {single_price:.2%} of par")
    print(f"  Multi-RC (worst-of all three): {fair_value_pct_of_par:.2%} of par  <- should be LOWER than")
    print(f"  every single-name figure above: the worst-of feature means there are three independent")
    print(f"  chances for one name to breach the strike, not just one - this is the real cost of")
    print(f"  dispersion, and it should show up here as a real, computed number, not an assertion.")

    plot_path(price_df, S0_list, TICKERS, underlying_names, mtm_enabled=True,
              sigmas=sigmas, qs=dividend_yields, corr_matrix=corr_matrix, greeks=greeks)
