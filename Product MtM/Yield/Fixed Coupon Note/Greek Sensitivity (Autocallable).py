import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql
from scipy.stats import norm, multivariate_normal

plt.rcParams["font.family"] = "Arial"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Same product terms as "Fixed Coupon Note (Autocallable).py" in this folder ---
STRIKE = 0.90
RISK_FREE_RATE = 0.04
GS_CDS_SPREAD = 0.005308
TRIGGER = 1.00
OBS_PER_YEAR = 4

ENTRY_DATE = "2025-01-01"
TENOR = 1

TICKER = "MCD"
FRED_SERIES = "SP500"

VOL_TERM_STRUCTURE_TICKERS = {"^VIX": 30, "^VIX3M": 93, "^VIX6M": 182}
VIX_FRED_SERIES = "VIXCLS"

N_MC_PATHS = 50000
MC_SEED = 42

# T (TENOR), strike, trigger and the funding curve are fixed constants
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


def digital_call_price(S, K, T, r, sigma, Q=1.0, q=0.0):
    """
    Cash-or-nothing digital call: pays Q if S_T >= K, priced under Black-Scholes
    as c = Q * e^(-rT) * N(d2). This is the MARGINAL probability of being above
    the trigger at a single date in isolation - see autocall_digital_strip_price
    below for the joint, first-passage-correct version used across multiple
    observation dates (the dates are NOT independent: they come from the same
    underlying Brownian path). `q` is the underlying's dividend yield - the
    risk-neutral drift of log S is r-q under a continuous yield.
    """
    if T <= 0:
        return Q if S >= K else 0.0
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return Q * np.exp(-r * T) * norm.cdf(d2)


def autocall_digital_strip_price(S0, obs_times, r, sigma, trigger, Q, q=0.0):
    """
    Closed-form value of the autocall feature as a strip of short digital
    calls, one per observation date, CORRECTLY accounting for the fact that
    the dates are jointly driven by one Brownian path (same construction as
    the Autocall Density / Autocall Monte Carlo exercise in Graphs (Maths),
    and as "Fixed Coupon Note (Autocallable).py" in this folder):

    Let X_i = W_ti / sqrt(t_i), the standardized Brownian value at each date.
    Marginally X_i ~ N(0,1), and corr(X_i, X_j) = sqrt(min(ti,tj)/max(ti,tj))
    since Cov(W_ti, W_tj) = min(ti, tj). "Called at t_i" means S_ti >= K,
    i.e. X_i > -d2_i (d2_i being the usual Black-Scholes d2 for that date).

    P(called for the FIRST time exactly at t_i)
        = P(X_1<=k_1,...,X_(i-1)<=k_(i-1))   [not called on any earlier date]
        - P(X_1<=k_1,...,X_i<=k_i)           [not called by t_i either]
      where k_j = -d2_j.

    Each date's digital payoff (Q - the fixed cash-or-nothing payout if
    in-the-money, i.e. the strike K=trigger*S0 itself, since being called
    means the note redeems at par at exactly that level) is then discounted
    and weighted by that EXACT first-passage probability, so the dates are
    no longer double-counted the way summing independent N(d2)'s would.
    """
    obs_times = np.asarray(obs_times, dtype=float)
    n = len(obs_times)
    if n == 0:
        return {"total_price": 0.0, "total_prob_called": 0.0, "per_date": []}

    K = trigger * S0
    d2 = (np.log(S0 / K) + (r - q - 0.5 * sigma ** 2) * obs_times) / (sigma * np.sqrt(obs_times))
    k = -d2

    # corr(X_i, X_j) for the standardized Brownian value at each observation date
    R = np.sqrt(np.minimum.outer(obs_times, obs_times) / np.maximum.outer(obs_times, obs_times))

    not_called_before = 1.0  # P(not called before the 1st observation) is trivially 1
    not_called_upto = 1.0
    per_date = []
    total_price = 0.0
    for i in range(n):
        cov_upto = R[:i + 1, :i + 1]
        not_called_upto = multivariate_normal(mean=np.zeros(i + 1), cov=cov_upto).cdf(k[:i + 1])

        prob_exact = not_called_before - not_called_upto
        price_i = Q * np.exp(-r * obs_times[i]) * prob_exact
        total_price += price_i
        per_date.append({"T": obs_times[i], "prob_exact": prob_exact, "price": price_i})

        not_called_before = not_called_upto

    return {"total_price": total_price, "total_prob_called": 1.0 - not_called_upto, "per_date": per_date}


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


def get_observation_dates(entry_date, tenor=TENOR, obs_per_year=OBS_PER_YEAR):
    start = pd.Timestamp(entry_date)
    n_obs = int(round(tenor * obs_per_year))
    return [start + pd.Timedelta(days=round(tenor * 365.25 * i / n_obs)) for i in range(1, n_obs + 1)]


def simulate_forward_price(S_t, S0, valuation_date, maturity_date, future_call_obs_dates,
                            sigma, r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD,
                            strike=STRIKE, trigger=TRIGGER, n_paths=N_MC_PATHS, seed=MC_SEED, q=0.0):
    T_remaining = (maturity_date - valuation_date).days / 365.25
    if T_remaining <= 0:
        put_quantity = 1.0 / strike
        put_payoff = -put_quantity * max(strike * S0 - S_t, 0.0)
        return {"price": S0 + put_payoff, "prob_called": 0.0}

    grid_dates = list(future_call_obs_dates) + [maturity_date]
    grid_years = np.array([(d - valuation_date).days / 365.25 for d in grid_dates])
    dt = np.diff(np.concatenate([[0.0], grid_years]))
    dt = np.clip(dt, 1e-8, None)

    rng = np.random.default_rng(seed)
    n_steps = len(dt)
    z = rng.standard_normal((n_paths, n_steps))

    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    log_levels = np.log(S_t) + np.cumsum(drift + diffusion, axis=1)
    levels = np.exp(log_levels)

    n_call_obs = len(future_call_obs_dates)
    if n_call_obs > 0:
        triggered = levels[:, :n_call_obs] >= trigger * S0
        any_triggered = triggered.any(axis=1)
        first_call_idx = np.argmax(triggered, axis=1)
    else:
        any_triggered = np.zeros(n_paths, dtype=bool)
        first_call_idx = np.zeros(n_paths, dtype=int)

    S_T = levels[:, -1]
    call_time = grid_years[first_call_idx]
    settle_time = np.where(any_triggered, call_time, grid_years[-1])

    principal_pv = S0 * (1 + r + credit_spread) ** (-settle_time)

    put_quantity = 1.0 / strike
    put_payoff = np.where(any_triggered, 0.0, -put_quantity * np.maximum(strike * S0 - S_T, 0.0))
    put_pv = put_payoff * np.exp(-r * grid_years[-1])

    price = float(np.mean(principal_pv + put_pv))
    return {"price": price, "prob_called": float(np.mean(any_triggered))}


# ---------------------------------------------------------------------------
# GREEKS LADDER
#
# T, strike, trigger and the funding curve are held constant throughout -
# the ONLY thing that varies across rows is spot. Vol is also pinned at
# its inception value for every row. Each Greek at a given spot level IS
# the answer to "how much does MTM move for a 1-unit change in that
# variable, right now, at this spot" - same design as every other product
# in this repo.
#
# Unlike the closed-form products, these Greeks are bump-and-reprice
# finite differences on the Monte Carlo engine, using common random
# numbers (same MC_SEED - identical draws) across the base and bumped
# runs so the finite differences aren't swamped by independent MC noise.
#
# Theta is deliberately excluded: with T fixed throughout this analysis,
# a "time passing" Greek doesn't belong in a table where time never
# actually moves.
# ---------------------------------------------------------------------------

def finite_difference_greeks(S_t, S0, valuation_date, maturity_date, future_call_obs_dates, sigma, r, credit_spread, q=0.0):
    eps_S = 0.005 * S0
    eps_sigma = 0.001
    eps_r = 0.0001

    def price_at(S=S_t, sig=sigma, rate=r):
        return simulate_forward_price(S, S0, valuation_date, maturity_date, future_call_obs_dates,
                                       sig, rate, credit_spread, q=q)["price"]

    price_mid = price_at()
    delta = (price_at(S=S_t + eps_S) - price_at(S=S_t - eps_S)) / (2 * eps_S)
    vega = (price_at(sig=sigma + eps_sigma) - price_at(sig=sigma - eps_sigma)) / (2 * eps_sigma)
    rho = (price_at(rate=r + eps_r) - price_at(rate=r - eps_r)) / (2 * eps_r)

    return {"price": price_mid, "delta": delta, "vega": vega, "rho": rho}


def greek_sensitivity_table(S0, entry_date, maturity_date, future_call_obs_dates, sigma,
                             r=RISK_FREE_RATE, credit_spread=GS_CDS_SPREAD, spot_multiples=SPOT_SCENARIO_RANGE, q=0.0):
    rows = []
    for multiple in spot_multiples:
        S = multiple * S0
        g = finite_difference_greeks(S, S0, entry_date, maturity_date, future_call_obs_dates, sigma, r, credit_spread, q=q)
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
        ax.axvline(TRIGGER * 100, color="dodgerblue", linewidth=0.8, linestyle="dotted")
        ax.axhline(0, color="lightgrey", linewidth=0.6)
        ax.grid(True, color="lightgrey", linewidth=0.4)
        ax.set_xlabel("Spot (% of S0)")
        ax.set_ylabel(ylabel)
        ax.set_title(col)

    fig.suptitle(
        f"{underlying_name} Autocallable Fixed Coupon Note - Greeks Ladder (T, strike, trigger fixed, spot varies)\n"
        f"Fixed throughout: T={T}y, vol={sigma:.2%}, SOFR={r:.2%}, CDS={GS_CDS_SPREAD:.2%}, "
        f"strike={STRIKE:.0%}, trigger={TRIGGER:.0%} of S0={S0:,.2f} (Monte Carlo, {N_MC_PATHS:,} paths)"
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    print("Greeks Ladder - Autocallable Fixed Coupon Note")
    print("=" * 60)
    print("This is NOT a new backtest and does not use the historical path.")
    print("It reuses the same entry conditions as the main backtest script,")
    print("then holds T, strike, trigger, funding curve and vol all fixed and")
    print("varies ONLY spot, so each Greek's value at a given spot level tells")
    print("you directly how much MTM moves for a 1-unit change in that")
    print("variable at that spot. Greeks are Monte Carlo finite differences")
    print("(common random numbers), not closed-form, since the autocall")
    print("feature has no simple algebraic formula. Theta is excluded since")
    print("T never varies in this analysis.\n")

    underlying_name = fetch_underlying_name(TICKER)
    print(f"Fetching {underlying_name} entry level for {ENTRY_DATE}...")
    path = fetch_index_path(ENTRY_DATE)
    S0 = float(path.iloc[0])
    entry_date = path.index[0]
    maturity_date = entry_date + pd.Timedelta(days=round(TENOR * 365.25))

    dividend_yield = fetch_dividend_yield(TICKER)

    observation_dates = get_observation_dates(ENTRY_DATE, TENOR, OBS_PER_YEAR)
    future_call_obs_dates = observation_dates[:-1]  # exclude maturity - see main script for rationale

    print(f"Fetching SPX implied vol term structure for {ENTRY_DATE}...")
    raw_term_structure = fetch_vol_term_structure(ENTRY_DATE)
    vols_at_entry = {tenor: series.iloc[0] for tenor, series in raw_term_structure.items()}
    entry_vol = interpolate_implied_vol(vols_at_entry, TENOR)

    print(f"\nFixed throughout (only spot varies below):")
    print(f"  Underlying:      {underlying_name} ({TICKER})")
    print(f"  Entry Date:      {ENTRY_DATE}")
    print(f"  S0 (spot):       {S0:,.2f}")
    print(f"  Strike:          {STRIKE:.0%} of S0")
    print(f"  Autocall trigger:{TRIGGER:.0%} of S0, checked quarterly ({len(future_call_obs_dates)} obs dates)")
    print(f"  Tenor (T):       {TENOR} year(s)")
    print(f"  Vol (sigma):     {entry_vol:.2%}  (from the VIX/VIX3M/VIX6M term structure on {ENTRY_DATE})")
    print(f"  SOFR proxy:      {RISK_FREE_RATE:.2%}")
    print(f"  GS CDS spread:   {GS_CDS_SPREAD:.2%}")
    print(f"  Dividend yield:  {dividend_yield:.2%}  (flat, continuous - 0% if an index)")

    obs_years_at_entry = [(d - entry_date).days / 365.25 for d in future_call_obs_dates]
    mc_entry = simulate_forward_price(S0, S0, entry_date, maturity_date, future_call_obs_dates, entry_vol, q=dividend_yield)
    strip = autocall_digital_strip_price(S0, obs_years_at_entry, RISK_FREE_RATE, entry_vol, TRIGGER,
                                          Q=TRIGGER * S0, q=dividend_yield)

    print(f"\nClosed-form check on the autocall feature (short digital calls, trigger={TRIGGER:.0%}):")
    print(f"  Each date's digital call c=Q*e^(-rT)*N(d2) is weighted by the EXACT first-passage")
    print(f"  probability of being called then (not earlier) - same joint-normal correlation")
    print(f"  structure as the main backtest script and Autocall Density.")
    for obs_date, leg in zip(future_call_obs_dates, strip["per_date"]):
        print(f"  Obs {obs_date.date()} (T={leg['T']:.2f}y): "
              f"P(called exactly here)={leg['prob_exact']:.2%}, digital call PV={leg['price']:,.2f}")
    print(f"  Closed-form autocall strip PV: {strip['total_price']:,.2f}")
    print(f"  Closed-form P(called before maturity): {strip['total_prob_called']:.2%}")
    print(f"  Monte Carlo P(called before maturity):  {mc_entry['prob_called']:.2%}")

    sensitivity = greek_sensitivity_table(S0, entry_date, maturity_date, future_call_obs_dates, entry_vol, q=dividend_yield)
    print(f"\nGreeks Ladder (Monte Carlo finite differences, common random numbers):")
    print(sensitivity.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    plot_greek_sensitivity(sensitivity, S0, TENOR, entry_vol, RISK_FREE_RATE, underlying_name)
