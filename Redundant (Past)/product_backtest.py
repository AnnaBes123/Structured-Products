"""
Realized 1-year payoff of an outperformance certificate vs. the S&P 500.

This is a DETERMINISTIC BACKTEST over one real historical index path -
no simulation, no Monte Carlo. We pull the actual daily S&P 500 closes
for the year following ENTRY_DATE, rebase that path so day 0 = 0%
(i.e. we track the running % return from entry, not the raw index
level), and evaluate the certificate's payoff formula against that
path - alongside a Black-Scholes mark-to-market valuation.

Data source: yfinance (^GSPC), falling back to FRED ("SP500") via
pandas_datareader if yfinance is unavailable or returns no data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------------------------------------------------------------
# PRODUCT TERMS - edit these to reprice the certificate
# ---------------------------------------------------------------------------

# Outperformance certificate: 1:1 with the index below strike, leveraged
# participation above strike.
STRIKE = 1.00                        # strike as a fraction of S0 (100% = entry level)
OUTPERFORMANCE_PARTICIPATION = 2.00  # participation on gains above strike (200% = double the gain)
RISK_FREE_RATE = 0.04                # flat risk-free rate assumption, used for Black-Scholes MTM

ENTRY_DATE = "2020-01-02"            # the one path we're backtesting
TENOR = 1                            # years

TICKER = "^GSPC"
FRED_SERIES = "SP500"

# CBOE SPX implied-vol term structure, used for the Black-Scholes MTM
# below: VIX itself is only a 30-day vol, which isn't the right input
# once the certificate has, say, 11 months left - these three tenor
# points let us interpolate the vol that actually matches each day's
# remaining time-to-maturity (see interpolate_implied_vol).
VOL_TERM_STRUCTURE_TICKERS = {"^VIX": 30, "^VIX3M": 93, "^VIX6M": 182}  # ticker -> tenor in days
VIX_FRED_SERIES = "VIXCLS"   # FRED fallback, available for ^VIX only


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def fetch_daily_closes(ticker, start, end, fred_series=None):
    """
    Pull daily closes for `ticker` between start and end. Tries
    yfinance first; if that fails or returns no data and a FRED series
    id was given, falls back to FRED via pandas_datareader. Returns a
    pandas Series of closes indexed by date.
    """
    series = None

    try:
        import yfinance as yf
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if data is not None and not data.empty:
            close = data["Close"]
            if isinstance(close, pd.DataFrame):  # yfinance can return a 1-col frame
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
            raise RuntimeError(
                f"Could not fetch {ticker} data from either yfinance or FRED: {exc}"
            )

    if series is None or series.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    return series


def fetch_index_path(entry_date, years=TENOR):
    """
    Pull daily closing prices for the index from entry_date to
    entry_date + `years` years.
    """
    start = pd.Timestamp(entry_date)
    end = start + pd.DateOffset(years=years)
    return fetch_daily_closes(TICKER, start, end, fred_series=FRED_SERIES)


def fetch_vol_term_structure(entry_date, years=TENOR):
    """
    Pull the CBOE SPX implied-vol term structure for the backtest
    window: VIX (30-day), VIX3M (93-day) and VIX6M (182-day), each
    quoted in volatility points (e.g. 15.2 means "the market is
    pricing in ~15.2% annualized vol") and converted here to decimal.

    Returns {tenor_in_years: pd.Series of decimal vol}. A tenor that
    fails to fetch is dropped rather than failing the whole run (VIX
    itself, with the FRED fallback, is the one point we really need).
    """
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


# ---------------------------------------------------------------------------
# PRODUCT PAYOFF
#
# Takes S0 (entry level) and the full daily path, and returns a pandas
# Series of the certificate's RUNNING return over time - i.e. "what the
# certificate would be worth if it settled on this day". The realized
# return at maturity is just the last value of that series, so the
# same function drives both the summary table and the overlay chart.
# ---------------------------------------------------------------------------

def outperformance_certificate_running_return(S0, path):
    """
    Outperformance certificate, strike = 100% of S0.
    Plain-English: below the strike, the certificate just mirrors the
    index move 1:1, like holding the index directly. Above the strike,
    every extra point of index gain is multiplied by the participation
    rate, so the certificate outperforms the index on the way up.
    """
    index_return = path / S0 - 1
    strike_level = STRIKE * S0
    above_strike = path >= strike_level

    running = index_return.copy()
    running[above_strike] = OUTPERFORMANCE_PARTICIPATION * index_return[above_strike]
    return running


# ---------------------------------------------------------------------------
# BLACK-SCHOLES MARK-TO-MARKET
#
# The running returns above are the payoff IF the certificate settled
# today - they ignore the option's remaining time value. This section
# instead prices the certificate as a small option portfolio using
# Black-Scholes, so we can see its fair value (and Greeks) before
# maturity, not just its payoff at maturity.
#
# An outperformance certificate is economically:
#   1.0x  zero-strike call        (= just owning the underlying)
# + (participation - 1)x  ATM call, struck at the entry level
# which reproduces the same payoff at maturity, but also carries
# time value away from maturity.
# ---------------------------------------------------------------------------

def black_scholes_call(S, K, T, r, sigma):
    """
    European call price and Greeks (Black-Scholes, no dividends).
    A zero strike (K = 0) is priced directly as the underlying itself:
    a call struck at zero is always exercised, so it's economically
    identical to just owning the stock (price = S, delta = 1, no
    sensitivity to vol or rates).
    """
    if K <= 0:
        return {"price": S, "delta": 1.0, "vega": 0.0, "rho": 0.0, "theta": 0.0}

    if T <= 0:
        # Exactly at/past maturity: no time value left, so use the
        # intrinsic payoff directly rather than plugging a tiny epsilon
        # into the Black-Scholes formula (which would leave a small
        # residual of time value and never quite match the payoff line).
        intrinsic = max(S - K, 0.0)
        delta = 1.0 if S > K else 0.0
        return {"price": intrinsic, "delta": delta, "vega": 0.0, "rho": 0.0, "theta": 0.0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    vega = S * np.sqrt(T) * norm.pdf(d1)   # change in price per 1.00 (100 vol pts) change in sigma
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)  # change in price per 1.00 (100%) change in r
    # change in price per year of time passing (S, sigma held fixed) - this is what
    # erodes the option's time value as the certificate approaches maturity
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    return {"price": price, "delta": delta, "vega": vega, "rho": rho, "theta": theta}


def interpolate_implied_vol(vols_by_tenor, T_years):
    """
    Interpolate SPX implied vol for a given time-to-maturity T_years
    from a term structure {tenor_in_years: vol_decimal} (e.g. the VIX/
    VIX3M/VIX6M points from fetch_vol_term_structure). Uses variance-
    time interpolation - total variance (vol^2 * T) is interpolated
    linearly between the two bracketing tenors, then converted back to
    a vol - the standard way to interpolate a vol term structure.
    Outside the range of available tenors (e.g. beyond VIX6M's 182
    days), the nearest endpoint's vol is held flat, since we have no
    longer-dated real market data here.
    """
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

    return points[-1][1]  # unreachable given the checks above, but a safe fallback


def outperformance_certificate_mtm(S, S0, T_remaining, sigma, r=RISK_FREE_RATE):
    """
    Mark-to-market value and Greeks of the outperformance certificate,
    replicated as 1.0x zero-strike call + (participation - 1)x ATM call.
    """
    zero_strike_leg = black_scholes_call(S, 0.0, T_remaining, r, sigma)
    atm_leg = black_scholes_call(S, STRIKE * S0, T_remaining, r, sigma)
    atm_quantity = OUTPERFORMANCE_PARTICIPATION - 1.0

    return {
        "price": zero_strike_leg["price"] + atm_quantity * atm_leg["price"],
        "delta": zero_strike_leg["delta"] + atm_quantity * atm_leg["delta"],
        "vega": zero_strike_leg["vega"] + atm_quantity * atm_leg["vega"],
        "rho": zero_strike_leg["rho"] + atm_quantity * atm_leg["rho"],
        "theta": zero_strike_leg["theta"] + atm_quantity * atm_leg["theta"],
    }


def outperformance_certificate_mtm_price_series(S0, path, vol_term_structure, r=RISK_FREE_RATE):
    """
    Running Black-Scholes fair value of the certificate (in the same
    price terms as the index, e.g. 3500 not 0.05), one valuation per
    day in the path as time-to-maturity shrinks.

    vol_term_structure is {tenor_in_years: pd.Series of decimal vol,
    aligned to path.index} - e.g. VIX/VIX3M/VIX6M from
    fetch_vol_term_structure. On EACH date, the vol actually used is
    interpolated (via interpolate_implied_vol) to match that day's
    remaining time-to-maturity, rather than reusing one flat tenor's
    vol for the certificate's whole life - a 30-day VIX print isn't
    the right input once 11 months are still left. Using the
    full-window REALIZED vol instead, for comparison, would ALSO leak
    information from later in the path (e.g. a future crash) into the
    valuation on day 1, before that crash happened - implied vol as of
    each date avoids that too.

    NOTE: at inception (S = S0, full tenor) this fair value is NOT
    generally equal to S0/par - it's whatever the option replication
    actually costs given OUTPERFORMANCE_PARTICIPATION, RISK_FREE_RATE
    and the entry-date vol. With no dividend yield assumed here, the
    zero-strike leg is worth exactly S0 with no slack left over, so
    ANY extra participation above 100% is unfunded, real option
    premium - a genuinely higher cost than par, not a bug. In
    practice, issuers fund this out of the underlying's dividend
    yield (a zero-strike call is worth S0*e^(-qT) < S0 when q > 0,
    freeing up budget for the extra ATM calls) - which this
    simplified model omits. See the "vs. par" figure printed in main
    for the size of the resulting gap.
    """
    maturity_date = path.index[-1]
    prices = []
    for date, level in path.items():
        T_remaining = (maturity_date - date).days / 365.25
        vols_today = {tenor: series[date] for tenor, series in vol_term_structure.items()}
        sigma = interpolate_implied_vol(vols_today, max(T_remaining, 0.0))
        prices.append(outperformance_certificate_mtm(level, S0, T_remaining, sigma, r)["price"])
    return pd.Series(prices, index=path.index)


# ---------------------------------------------------------------------------
# RISK METRICS
# ---------------------------------------------------------------------------

def realized_annualized_vol(path):
    """Annualized volatility from daily log returns (std * sqrt(252))."""
    log_returns = np.log(path / path.shift(1)).dropna()
    return log_returns.std() * np.sqrt(252)


def max_drawdown(path):
    """Worst peak-to-trough decline over the window, as a negative decimal."""
    running_max = path.cummax()
    drawdown = path / running_max - 1
    return drawdown.min()


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def plot_path(path, S0, mtm_vol_term_structure=None):
    """
    Plot the S&P 500's running return over the year, overlaid with the
    outperformance certificate's running return (payoff-if-settled-today).

    If mtm_vol_term_structure is given (the {tenor_years: vol Series}
    dict from fetch_vol_term_structure), also overlay the certificate's
    Black-Scholes mark-to-market value
    (dashed, on its own right-hand axis) - the certificate is
    structurally "index + a free-standing call option", so its value
    must always sit at or above both the index line and its own
    settle-today payoff line (time value is never negative for a long
    option), meeting the payoff line only exactly at maturity, where
    time value has fully decayed to zero.
    """
    index_return_pct = (path / S0 - 1) * 100
    certificate_return_pct = outperformance_certificate_running_return(S0, path) * 100

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(index_return_pct.index, index_return_pct.values, color="tab:blue", linewidth=1.5,
            label="S&P 500")
    ax.plot(certificate_return_pct.index, certificate_return_pct.values, color="tab:orange", linewidth=1.5,
            label="Outperformance Certificate Participation Tracker")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return from Entry (%)")
    lines, labels = ax.get_legend_handles_labels()

    if mtm_vol_term_structure is not None:
        # A second y-axis, in red, kept on the SAME scale as the left one -
        # the numbers line up 1:1 with the left axis, this just makes it
        # visually unambiguous that the dashed red line is a different kind
        # of quantity (a fair value vs. par) from the solid "return since
        # entry" lines on the left axis, not a competing return figure.
        mtm_price = outperformance_certificate_mtm_price_series(S0, path, mtm_vol_term_structure)
        mtm_gain_over_par_pct = (mtm_price / S0 - 1) * 100  # fair value, as a gain/loss vs. par

        ax2 = ax.twinx()
        mtm_line, = ax2.plot(
            mtm_gain_over_par_pct.index, mtm_gain_over_par_pct.values, color="red", linewidth=1.5,
            linestyle="dashdot", label="Outperformance Certificate - Fair Value (% of Par, Black-Scholes)")
        ax2.set_ylabel("MTM Fair Value vs. Par (%)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        # Label the line's endpoint directly, so its final level is
        # readable without cross-referencing the legend.
        last_date = mtm_gain_over_par_pct.index[-1]
        last_value = mtm_gain_over_par_pct.iloc[-1]


        ax2.set_ylim(ax.get_ylim())  # keep both axes on the same numeric scale
        ax.margins(x=0.12)  # leave room on the right for the end-of-line label

        lines.append(mtm_line)
        labels.append(mtm_line.get_label())

    end_date = index_return_pct.index[-1]
    ax.set_title(f"S&P 500 Path from {path.index[0].date()} (Entry) to {end_date.date()} (Maturity)")
    ax.legend(lines, labels, loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig("product_backtest_results.png", dpi=150)
    print("\nChart saved to product_backtest_results.png")
    plt.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Fetching S&P 500 path from {ENTRY_DATE} (entry) to +{TENOR}y (maturity)...")
    path = fetch_index_path(ENTRY_DATE)

    S0 = float(path.iloc[0])
    S_T = float(path.iloc[-1])
    vol = realized_annualized_vol(path)  # backward-looking summary stat only - NOT used for MTM below
    certificate_return = outperformance_certificate_running_return(S0, path).iloc[-1]

    summary = pd.Series({
        "Entry Date": path.index[0].date(),
        "Maturity Date": path.index[-1].date(),
        "S0": f"{S0:,.2f}",
        "S_T": f"{S_T:,.2f}",
        "S&P 500 Return": f"{S_T / S0 - 1:.2%}",
        "Outperformance Certificate": f"{certificate_return:.2%}",
        "Realized Vol (ann.)": f"{vol:.2%}",
        "Max Drawdown": f"{max_drawdown(path):.2%}",
    })

    print("\n" + summary.to_string())

    print(f"\nFetching SPX implied vol term structure (VIX/VIX3M/VIX6M) for the same window...")
    raw_term_structure = fetch_vol_term_structure(ENTRY_DATE)
    # Align each tenor's series to the index's trading days.
    # (Each can be missing the odd day; ffill/bfill covers small gaps
    # between the different calendars.)
    vol_term_structure = {
        tenor: series.reindex(path.index).ffill().bfill()
        for tenor, series in raw_term_structure.items()
    }
    vols_at_entry = {tenor: series.iloc[0] for tenor, series in vol_term_structure.items()}
    entry_vol = interpolate_implied_vol(vols_at_entry, TENOR)

    # Day-1 Greeks: how the certificate's Black-Scholes value would
    # move for a small change in the underlying / vol / rates / time,
    # priced at inception (S = S0, full tenor remaining, vol =
    # interpolated from the entry-date SPX vol term structure for a
    # T={TENOR}y maturity - the market's actual vol view for THAT
    # tenor on that day, not a 30-day VIX print applied to a much
    # longer holding period, and not a full-year realized vol that
    # hadn't happened yet).
    greeks = outperformance_certificate_mtm(S0, S0, TENOR, entry_vol)
    fair_value_pct_of_par = greeks["price"] / S0

    print(f"\nOutperformance Certificate fair value at inception")
    print(f"(Black-Scholes, S=S0, T={TENOR}y, vol={entry_vol:.2%} interpolated from the")
    print(f"{path.index[0].date()} VIX/VIX3M/VIX6M term structure for a {TENOR}y maturity):")
    print(f"  {fair_value_pct_of_par:.2%} of par ({fair_value_pct_of_par - 1:+.2%} vs. par)")
    print(f"  With 0% dividend yield assumed, the {OUTPERFORMANCE_PARTICIPATION:.0%} participation is")
    print(f"  unfunded, real option premium - a real issuer would fund it from the underlying's")
    print(f"  dividend yield instead, or dial participation down, to price the note at par.")
    print(f"  (Note: this term-structure vol, {entry_vol:.2%}, differs from the {vol:.2%} REALIZED")
    print(f"  vol over the full year above - that full-year number is inflated by the crash that")
    print(f"  hadn't happened yet on day 1, so using it here would have been look-ahead bias. It's")
    print(f"  also not the same as the raw 30-day VIX print, since our tenor here is {TENOR}y, and")
    print(f"  VIX6M's 182-day point is held flat beyond 182 days for lack of longer-dated data.)")
    print(f"  The MTM line in the chart is a return on par (S0), like every other line, so it sits")
    print(f"  at or above the certificate's own payoff line throughout, meeting it only at maturity.")

    print(f"\nOutperformance Certificate Greeks at inception:")
    print(f"  Delta: {greeks['delta']:.2f}  (certificate return per 1.00 return in the index)")
    print(f"  Vega:  {greeks['vega'] / S0 * 0.01:.4f}  (change in return per 1% change in vol)")
    print(f"  Rho:   {greeks['rho'] / S0 * 0.01:.4f}  (change in return per 1% change in rates)")
    print(f"  Theta: {greeks['theta'] / S0:.4f} per year / {greeks['theta'] / S0 / 365:.5f} per day"
          f"  (change in return from time passing alone, S & vol unchanged)")

    plot_path(path, S0, mtm_vol_term_structure=vol_term_structure)
