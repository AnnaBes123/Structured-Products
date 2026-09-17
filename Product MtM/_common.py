"""Shared helpers for the Product MtM scripts.

Every product folder used to carry its own copy of these functions,
byte-for-byte identical apart from a stray docstring cross-reference here
and there. Consolidated here so there's exactly one definition of each to
read, test, and fix.
"""

import numpy as np
import pandas as pd
import QuantLib as ql


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
    """Flat continuous dividend yield q (0 for an index). See each
    product's README "Underlying selection" section for field-selection
    notes."""
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


def fetch_risk_free_rate(entry_date, fred_series="DGS1", lookback_days=10):
    """Most recent FRED DGS1 (1-Year Treasury Constant Maturity Rate) print
    on or before entry_date - a free, real, historical risk-free proxy
    whose tenor matches every product here except the Dual Currency
    Investment (TENOR != 1). Returns a decimal (e.g. 0.0434), not a
    percentage. See Product MtM/README.md "The Zero Coupon Bond
    Assumption" for why this replaced a flat hand-set number."""
    import pandas_datareader.data as web
    end = pd.Timestamp(entry_date)
    start = end - pd.Timedelta(days=lookback_days)
    data = web.DataReader(fred_series, "fred", start, end).dropna()
    if data.empty:
        raise RuntimeError(
            f"No {fred_series} data available from FRED in the {lookback_days} days "
            f"before {end.date()} - try a larger lookback_days or check the date."
        )
    return float(data[fred_series].iloc[-1]) / 100.0


def fetch_trailing_realized_vol(ticker, entry_date, lookback_years=2):
    """Single-name vol input: trailing realized vol on the ticker's own
    history (see each product's README "Volatility" section for why, vs.
    VIX)."""
    start = pd.Timestamp(entry_date) - pd.Timedelta(days=round(lookback_years * 365.25))
    end = pd.Timestamp(entry_date)
    closes = fetch_daily_closes(ticker, start, end)
    log_returns = np.log(closes / closes.shift(1)).dropna()
    return float(log_returns.std() * np.sqrt(252))


def realized_annualized_vol(path):
    log_returns = np.log(path / path.shift(1)).dropna()
    return log_returns.std() * np.sqrt(252)


def max_drawdown(path):
    running_max = path.cummax()
    drawdown = path / running_max - 1
    return drawdown.min()


def _quantlib_process(S, r, q, sigma, today):
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    rf_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count, ql.Continuous, ql.Annual))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count, ql.Continuous, ql.Annual))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, day_count))
    return ql.BlackScholesMertonProcess(spot, div_ts, rf_ts, vol_ts)


def zcb_price_and_greeks(principal, T_remaining, funding_rate):
    """Continuous-compounding ZCB price plus rho/theta - see each
    product's MATHEMATICS.md section 1."""
    price = principal * np.exp(-funding_rate * T_remaining)
    rho = -T_remaining * price
    theta = price * funding_rate
    return {"price": price, "rho": rho, "theta": theta}
