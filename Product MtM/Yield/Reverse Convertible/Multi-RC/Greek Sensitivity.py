import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql

plt.rcParams["font.family"] = "Arial"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Same product terms as "Multi-RC.py" in this folder ---
STRIKE = 0.90
RISK_FREE_RATE = 0.04
GS_CDS_SPREAD = 0.005308

ENTRY_DATE = "2025-01-02"
TENOR = 1

TICKERS = ["AAPL", "JPM", "XOM"]
CORRELATION_LOOKBACK_YEARS = 2

N_MC_PATHS = 50000
MC_SEED = 42

# T, strike and the funding curve are fixed constants throughout this whole
# analysis - the only thing that varies below is a COMMON multiplicative
# shock applied to every name in the basket AT ONCE (see the README for why
# this, rather than bumping one name at a time, is the natural x-axis for a
# multi-asset product's Greeks ladder). Vol/correlation/dividend yields are
# held at their (real, trailing-window-estimated) entry values throughout.
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
    """Trailing dividend yield, used as a flat continuous yield q. See the
    README in this folder and the single-name RC's README (one folder up)
    for the full rationale and field-selection notes."""
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
    series = {ticker: fetch_daily_closes(ticker, start, end) for ticker in tickers}
    df = pd.concat(series, axis=1)
    return df.dropna(how="any")


def estimate_vols_and_correlation(calibration_price_df):
    """Realized annualized vol per name and the full correlation matrix,
    both from real trailing daily-close history - see Multi-RC.py in
    this folder for the full rationale."""
    log_returns = np.log(calibration_price_df / calibration_price_df.shift(1)).dropna()
    vols = log_returns.std() * np.sqrt(252)
    corr = log_returns.corr()
    return vols.to_dict(), corr


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


def worst_of_put_price(relative_spots, K, T, r, sigmas, qs, corr_matrix, n_paths=N_MC_PATHS, seed=MC_SEED):
    """MC price of a worst-of put via QuantLib's own basket-option engine
    (StochasticProcessArray + MinBasketPayoff + MCEuropeanBasketEngine) -
    see Multi-RC.py in this folder for the full rationale."""
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


def multi_rc_price(relative_spots, T_remaining, sigmas, qs, corr_matrix, r=RISK_FREE_RATE,
                     credit_spread=GS_CDS_SPREAD, n_paths=N_MC_PATHS, seed=MC_SEED):
    zcb = zcb_price_and_greeks(1.0, T_remaining, r + credit_spread)
    put_quantity = 1.0 / STRIKE

    if T_remaining <= 0:
        worst_of = min(relative_spots)
        put_payoff = max(STRIKE - worst_of, 0.0)
        return {"price": zcb["price"] - put_quantity * put_payoff}

    put_price = worst_of_put_price(relative_spots, STRIKE, T_remaining, r, sigmas, qs, corr_matrix, n_paths, seed)
    return {"price": zcb["price"] - put_quantity * put_price}


# ---------------------------------------------------------------------------
# GREEKS LADDER
#
# T, strike, funding curve, vol, correlation and dividend yields are all
# held constant throughout - the only thing that varies across rows is a
# COMMON multiplicative shock applied to every name's relative-performance
# spot AT ONCE (e.g. row "80%" = every name in the basket is simultaneously
# 20% below its own entry level). This is the natural x-axis for a
# multi-asset product: a per-name x-axis would need N separate ladders (one
# holding the other N-1 names fixed), which doesn't compose into a single
# chart the way a shared "market-wide move" scenario does, and it's also
# the more economically meaningful scenario to stress-test a worst-of note
# against (a broad selloff, not one name idiosyncratically diverging).
#
# Delta and Vega are reported PER NAME (one line per basket member on those
# two panels) - a multi-asset product's risk is a genuine vector, not a
# scalar. Rho stays a single shared-rate line, same as every other product.
# Theta is excluded, same fixed-tenor reasoning as every other Greeks
# ladder in this repo.
# ---------------------------------------------------------------------------

def greek_sensitivity_table(T, sigmas, qs, corr_matrix, tickers, r=RISK_FREE_RATE,
                             spot_multiples=SPOT_SCENARIO_RANGE, n_paths=N_MC_PATHS, seed=MC_SEED):
    bump_S, bump_sigma = 0.01, 0.005
    n = len(tickers)
    rows = []
    for multiple in spot_multiples:
        relative_spots = [multiple] * n
        base = multi_rc_price(relative_spots, T, sigmas, qs, corr_matrix, r, n_paths=n_paths, seed=seed)["price"]

        deltas = []
        for i in range(n):
            up = list(relative_spots); up[i] += bump_S
            down = list(relative_spots); down[i] -= bump_S
            price_up = multi_rc_price(up, T, sigmas, qs, corr_matrix, r, n_paths=n_paths, seed=seed)["price"]
            price_down = multi_rc_price(down, T, sigmas, qs, corr_matrix, r, n_paths=n_paths, seed=seed)["price"]
            deltas.append((price_up - price_down) / (2 * bump_S))

        vegas = []
        for i in range(n):
            sig_up = list(sigmas); sig_up[i] += bump_sigma
            sig_down = list(sigmas); sig_down[i] -= bump_sigma
            price_up = multi_rc_price(relative_spots, T, sig_up, qs, corr_matrix, r, n_paths=n_paths, seed=seed)["price"]
            price_down = multi_rc_price(relative_spots, T, sig_down, qs, corr_matrix, r, n_paths=n_paths, seed=seed)["price"]
            vegas.append((price_up - price_down) / (2 * bump_sigma))

        bump_r = 0.0001
        price_up_r = multi_rc_price(relative_spots, T, sigmas, qs, corr_matrix, r + bump_r, n_paths=n_paths, seed=seed)["price"]
        price_down_r = multi_rc_price(relative_spots, T, sigmas, qs, corr_matrix, r - bump_r, n_paths=n_paths, seed=seed)["price"]
        rho = (price_up_r - price_down_r) / (2 * bump_r)

        row = {"Spot (% of S0, all names)": multiple * 100, "Price (% of Par)": base * 100, "Rho (per 1% rate)": rho * 0.01}
        for i, ticker in enumerate(tickers):
            row[f"Delta ({ticker})"] = deltas[i]
            row[f"Vega ({ticker}, per 1% vol)"] = vegas[i] * 0.01
        rows.append(row)
    return pd.DataFrame(rows)


def plot_greek_sensitivity(table, tickers, T, r, underlying_names):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    spot = table["Spot (% of S0, all names)"]
    asset_colors = ["steelblue", "darkorange", "seagreen", "mediumorchid", "peru", "slategray"]

    ax = axes[0, 0]
    ax.plot(spot, table["Price (% of Par)"], color="firebrick", linewidth=1.8, marker="o", markersize=3)
    ax.axvline(100, color="lightgrey", linewidth=0.8, linestyle="dashed")
    ax.axvline(STRIKE * 100, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.set_title("Price (% of Par)")
    ax.set_xlabel("Spot (% of S0, all names shocked together)")
    ax.set_ylabel("Fair Value (% of Par)")
    ax.grid(True, color="lightgrey", linewidth=0.4)

    ax = axes[0, 1]
    for i, (ticker, name) in enumerate(zip(tickers, underlying_names)):
        ax.plot(spot, table[f"Delta ({ticker})"], color=asset_colors[i % len(asset_colors)], linewidth=1.8,
                marker="o", markersize=3, label=name)
    ax.axvline(100, color="lightgrey", linewidth=0.8, linestyle="dashed")
    ax.axvline(STRIKE * 100, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.axhline(0, color="lightgrey", linewidth=0.6)
    ax.set_title("Delta (per name)")
    ax.set_xlabel("Spot (% of S0, all names shocked together)")
    ax.set_ylabel("Delta")
    ax.legend(fontsize=8)
    ax.grid(True, color="lightgrey", linewidth=0.4)

    ax = axes[1, 0]
    for i, (ticker, name) in enumerate(zip(tickers, underlying_names)):
        ax.plot(spot, table[f"Vega ({ticker}, per 1% vol)"], color=asset_colors[i % len(asset_colors)], linewidth=1.8,
                marker="o", markersize=3, label=name)
    ax.axvline(100, color="lightgrey", linewidth=0.8, linestyle="dashed")
    ax.axvline(STRIKE * 100, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.axhline(0, color="lightgrey", linewidth=0.6)
    ax.set_title("Vega (per name, per 1% change in that name's vol)")
    ax.set_xlabel("Spot (% of S0, all names shocked together)")
    ax.set_ylabel("Vega")
    ax.legend(fontsize=8)
    ax.grid(True, color="lightgrey", linewidth=0.4)

    ax = axes[1, 1]
    ax.plot(spot, table["Rho (per 1% rate)"], color="brown", linewidth=1.8, marker="o", markersize=3)
    ax.axvline(100, color="lightgrey", linewidth=0.8, linestyle="dashed")
    ax.axvline(STRIKE * 100, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.axhline(0, color="lightgrey", linewidth=0.6)
    ax.set_title("Rho (per 1% change in SOFR)")
    ax.set_xlabel("Spot (% of S0, all names shocked together)")
    ax.set_ylabel("Rho")
    ax.grid(True, color="lightgrey", linewidth=0.4)

    basket_label = " / ".join(underlying_names)
    fig.suptitle(
        f"Worst-of Basket ({basket_label}) Multi-RC - Greeks Ladder\n"
        f"(T={T}y, strike={STRIKE:.0%}, r={r:.2%} fixed throughout; x-axis is a COMMON shock applied "
        f"to every name in the basket at once)"
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    print("Greeks Ladder - Multi-RC (Worst-of Reverse Convertible)")
    print("=" * 60)
    print("This is NOT a new backtest and does not use the historical path.")
    print("It reuses the same entry conditions (and the same trailing-window")
    print("vol/correlation estimate) as the main backtest script, then holds")
    print("T, strike, funding curve, vol and correlation fixed and varies ONLY")
    print("a COMMON shock applied to every name in the basket at once. Delta")
    print("and Vega are reported per name (a multi-asset product's risk is a")
    print("vector, not a scalar). Theta is excluded since T never varies.\n")

    underlying_names = [fetch_underlying_name(t) for t in TICKERS]
    print(f"Basket: {', '.join(f'{n} ({t})' for n, t in zip(underlying_names, TICKERS))}")

    entry_ts = pd.Timestamp(ENTRY_DATE)
    calibration_start = entry_ts - pd.Timedelta(days=round(CORRELATION_LOOKBACK_YEARS * 365.25))

    print(f"\nFetching {CORRELATION_LOOKBACK_YEARS}y trailing history (ending {ENTRY_DATE}) for vol/correlation...")
    calibration_df = fetch_multi_asset_path(TICKERS, calibration_start, entry_ts)
    vols_dict, corr_df = estimate_vols_and_correlation(calibration_df)
    sigmas = [vols_dict[t] for t in TICKERS]
    corr_matrix = corr_df.loc[TICKERS, TICKERS].values

    dividend_yields = [fetch_dividend_yield(t) for t in TICKERS]

    print(f"\nFixed throughout (only the common spot shock varies below):")
    for t, n, s, q in zip(TICKERS, underlying_names, sigmas, dividend_yields):
        print(f"  {n} ({t}): vol={s:.2%}, dividend yield={q:.2%}")
    print(f"  Strike:          {STRIKE:.0%} of each name's own entry level")
    print(f"  Tenor (T):       {TENOR} year(s)")
    print(f"  SOFR proxy:      {RISK_FREE_RATE:.2%}")

    sensitivity = greek_sensitivity_table(TENOR, sigmas, dividend_yields, corr_matrix, TICKERS)
    print(f"\nGreeks Ladder (QuantLib MCEuropeanBasketEngine finite differences, common random numbers):")
    print(sensitivity.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    plot_greek_sensitivity(sensitivity, TICKERS, TENOR, RISK_FREE_RATE, underlying_names)
