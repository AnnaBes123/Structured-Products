"""
Rolling historical backtest of one or more Fixed Coupon Notes (FCN) - a worst-of, physically-
settled, autocallable note with a EUROPEAN (maturity-only) downside test, no knock-in barrier.
Not a `Product MtM/` script - no option replication, no QuantLib, no mark-to-market. Launches the
note on every trading day of an explicit launch period and classifies each launch mechanically
against one real historical price path (or worst-of basket of paths).

Full product mechanics, terminology, and scope/limitations: see README.md. Design history and
rationale (why launches are handled this way, bugs that got fixed, rejected alternatives): see
DEVLOG.md. Both are one directory up from PRODUCTS below.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OUTCOME_COLORS = {
    "AUTOCALL": "dodgerblue",
    "MATURITY_CASH": "seagreen",
    "PHYSICAL_DELIVERY": "firebrick",
    "OUTSTANDING": "lightgrey",
}

# ---------------------------------------------------------------------------
# PRODUCTS - each one is a fully independent FCN, backtested and reported on
# separately (see DEVLOG.md "Why per-product terms"). Use the printed
# "Observed underlying behavior" diagnostics to pick STRIKE/AUTOCALL_TRIGGER
# for a new product before trusting its results.
# ---------------------------------------------------------------------------

PRODUCTS = [
    dict(
        TICKERS=["AMD"],          # 1 ticker = single-underlying; 2+ = worst-of basket
        STRIKE=0.90,                # fraction of INITIAL_VALUE, checked ONLY at maturity
        AUTOCALL_TRIGGER=1.00,      # fraction of INITIAL_VALUE, checked at each observation
        TENOR_MONTHS=12,
        AUTOCALL_FREQUENCY_MONTHS=3,   # None disables autocall entirely
        AUTOCALL_LOCKOUT_MONTHS=0,
    ),
    # Add more products here, e.g.:
    # dict(TICKERS=["GME"], STRIKE=0.70, AUTOCALL_TRIGGER=1.00,
    #      TENOR_MONTHS=12, AUTOCALL_FREQUENCY_MONTHS=3, AUTOCALL_LOCKOUT_MONTHS=0),
]


def product_label(terms):
    """Display/filename label - derived from TICKERS, not hand-maintained, so changing TICKERS is
    enough on its own (see DEVLOG.md "Product label derived from TICKERS")."""
    return "/".join(terms["TICKERS"]) + " FCN"

INITIAL_VALUE = 1.00   # always 1.00 by construction (each underlying normalized to its own launch
                        # level); named so STRIKE/AUTOCALL_TRIGGER read as "fraction of", not a price

# Explicit launch-period / data-cutoff boundaries, shared across every product in PRODUCTS.
LAUNCH_START = "2001-09-18"   # first candidate launch date (inclusive)
LAUNCH_END = None              # last candidate launch date (inclusive); None -> DATA_AS_OF
DATA_AS_OF = None              # price-data cutoff (inclusive); None -> most recent COMPLETE
                                # calendar day before now (see resolve_run_dates)

MAX_OBSERVATION_GAP_DAYS = 10   # calendar-day tolerance before a rolled observation date is
                                  # flagged as an unresolved data gap - see map_observation

# Candidate levels for the per-product "Observed underlying behavior" diagnostic - purely
# descriptive, independent of whatever STRIKE/AUTOCALL_TRIGGER that product actually uses.
DIAGNOSTIC_TRIGGER_CANDIDATES = (0.90, 0.95, 1.00, 1.05, 1.10)
DIAGNOSTIC_STRIKE_CANDIDATES = (0.50, 0.60, 0.70, 0.80, 0.90)


# ---------------------------------------------------------------------------
# OBSERVATION SCHEDULE + VALIDATION
# ---------------------------------------------------------------------------

def build_observation_offsets(tenor_months, frequency_months, lockout_months):
    """Month-offsets (from launch) of autocall observation dates - always > 0 and < tenor_months,
    so month-zero/launch-day autocall is impossible by construction (see DEVLOG.md)."""
    if frequency_months is None:
        return []
    start = lockout_months + frequency_months
    return list(range(start, tenor_months, frequency_months))


def validate_terms(name, terms):
    tickers = terms["TICKERS"]
    strike, trigger = terms["STRIKE"], terms["AUTOCALL_TRIGGER"]
    tenor, freq, lockout = terms["TENOR_MONTHS"], terms["AUTOCALL_FREQUENCY_MONTHS"], terms["AUTOCALL_LOCKOUT_MONTHS"]

    if not (isinstance(tickers, (list, tuple)) and len(tickers) >= 1):
        raise ValueError(f"{name}: TICKERS must be a nonempty list, got {tickers!r}")
    if len(set(tickers)) != len(tickers):
        raise ValueError(f"{name}: TICKERS must not contain duplicates, got {tickers!r}")
    if not (0 < strike <= INITIAL_VALUE):
        raise ValueError(f"{name}: STRIKE must satisfy 0 < STRIKE <= INITIAL_VALUE, got {strike!r}")
    if not (trigger > 0):
        raise ValueError(f"{name}: AUTOCALL_TRIGGER must be > 0, got {trigger!r}")
    if not (isinstance(tenor, int) and not isinstance(tenor, bool) and tenor > 0):
        raise ValueError(f"{name}: TENOR_MONTHS must be a positive integer, got {tenor!r}")
    if freq is not None and not (isinstance(freq, int) and not isinstance(freq, bool) and freq > 0):
        raise ValueError(f"{name}: AUTOCALL_FREQUENCY_MONTHS must be None or a positive integer, got {freq!r}")
    if not (isinstance(lockout, int) and not isinstance(lockout, bool) and lockout >= 0):
        raise ValueError(f"{name}: AUTOCALL_LOCKOUT_MONTHS must be a nonnegative integer, got {lockout!r}")

    if freq is not None:
        offsets = build_observation_offsets(tenor, freq, lockout)
        if any(m <= 0 for m in offsets):
            raise ValueError(f"{name}: computed autocall observation offsets must all be > 0 "
                              f"(no launch-day autocall), got {offsets!r}")
        if any(m >= tenor for m in offsets):
            raise ValueError(f"{name}: computed autocall observation offsets must all be strictly "
                              f"before TENOR_MONTHS, got {offsets!r}")


for _terms in PRODUCTS:
    validate_terms(product_label(_terms), _terms)


def resolve_run_dates():
    """Resolves LAUNCH_START/LAUNCH_END/DATA_AS_OF into concrete pd.Timestamps.
    DATA_AS_OF=None -> yesterday, not today (see DEVLOG.md)."""
    if DATA_AS_OF is None:
        data_as_of = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    else:
        data_as_of = pd.Timestamp(DATA_AS_OF)

    launch_start = pd.Timestamp(LAUNCH_START)
    launch_end = data_as_of if LAUNCH_END is None else pd.Timestamp(LAUNCH_END)

    if launch_start > launch_end:
        raise ValueError(f"LAUNCH_START ({launch_start.date()}) must be <= LAUNCH_END ({launch_end.date()})")
    if launch_end > data_as_of:
        raise ValueError(f"LAUNCH_END ({launch_end.date()}) must be <= DATA_AS_OF ({data_as_of.date()})")

    return launch_start, launch_end, data_as_of


# ---------------------------------------------------------------------------
# DATA LOADING - same yfinance/FRED-fallback pattern used throughout the repo
# ---------------------------------------------------------------------------

def fetch_daily_closes(ticker, start, end, fred_series=None):
    """Daily closes for `ticker`. yfinance first, FRED fallback if given and yfinance is empty.
    auto_adjust=False: split-adjusted but NOT dividend-adjusted, deliberately (see README)."""
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
            raise RuntimeError(f"Could not fetch {ticker} data from either yfinance or FRED: {exc}")

    if series is None or series.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    return series


def fetch_underlying_name(ticker):
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker


def fetch_multi_asset_path(tickers, start, end):
    """Aligned daily-close matrix for all tickers. Inner join across tickers - a date missing on
    ANY ticker is dropped from ALL (see DEVLOG.md "Basket dates" for why, and what it can drop)."""
    fred = "SP500" if tickers == ["^GSPC"] else None
    if len(tickers) == 1:
        series = {tickers[0]: fetch_daily_closes(tickers[0], start, end, fred_series=fred)}
    else:
        series = {t: fetch_daily_closes(t, start, end) for t in tickers}
    df = pd.concat(series, axis=1)
    df.columns = tickers
    n_before = len(df)
    df = df.dropna(how="any")
    n_after = len(df)
    if len(tickers) > 1 and n_before > n_after:
        dropped_pct = (n_before - n_after) / n_before
        flag = "  <-- large mismatch, verify basket calendars" if dropped_pct > 0.05 else ""
        print(f"  Basket calendar alignment: {n_before - n_after} of {n_before} dates dropped "
              f"({dropped_pct:.1%}) because at least one ticker had no price that day.{flag}")
    return df


def validate_price_df(price_df, tickers):
    """Sorted/unique date index and finite, strictly positive prices - anything else would
    silently corrupt every ratio computed downstream."""
    if price_df.empty:
        raise ValueError("price_df is empty - no overlapping trading days across the requested tickers/dates")
    idx = price_df.index
    if not idx.is_monotonic_increasing:
        raise ValueError("price_df index is not sorted ascending")
    if not idx.is_unique:
        dupes = idx[idx.duplicated()]
        raise ValueError(f"price_df index has duplicate dates: {list(dupes[:5])}{'...' if len(dupes) > 5 else ''}")
    values = price_df[tickers].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("price_df contains non-finite (NaN/inf) prices")
    if not (values > 0).all():
        raise ValueError("price_df contains zero or negative prices")


def realized_annualized_vol_log(price_series):
    """Annualized realized vol from log returns - same convention as Product MtM/_common.py."""
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    return float(log_returns.std() * np.sqrt(252))


# ---------------------------------------------------------------------------
# OBSERVATION -> TRADING DAY MAPPING (business-day convention)
# ---------------------------------------------------------------------------

def map_observation(target_calendar_date, all_dates, max_gap_days=MAX_OBSERVATION_GAP_DAYS):
    """Maps a scheduled calendar date to the next available trading day ("following" convention).
    Returns (position, actual_date, gap_days, resolved); position/actual_date are None if the
    target is beyond the last available date. resolved=False if gap_days > max_gap_days - a large
    gap is treated as a data hole, not a holiday roll (see DEVLOG.md)."""
    pos = int(all_dates.searchsorted(target_calendar_date))
    if pos >= len(all_dates):
        return None, None, None, False
    actual_date = all_dates[pos]
    gap_days = (actual_date - target_calendar_date).days
    resolved = gap_days <= max_gap_days
    return pos, actual_date, gap_days, resolved


# ---------------------------------------------------------------------------
# PER-LAUNCH CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_launch(prices, all_dates, i, tenor_months, obs_offsets, strike, trigger,
                     data_as_of, max_gap_days=MAX_OBSERVATION_GAP_DAYS):
    """Classifies ONE launch (position i) into AUTOCALL / MATURITY_CASH / PHYSICAL_DELIVERY /
    OUTSTANDING as of data_as_of. Returns a dict with the full per-launch audit detail (see
    build_audit_dataframe)."""
    launch_date = all_dates[i]
    s0 = prices[i]  # each ticker's own level at launch (worst-of is normalized per-ticker)
    maturity_calendar = launch_date + pd.DateOffset(months=tenor_months)

    j = None
    maturity_actual_date = None
    full_tenor_observable = maturity_calendar <= data_as_of
    if full_tenor_observable:
        pos, actual_date, gap, resolved = map_observation(maturity_calendar, all_dates, max_gap_days)
        if pos is None or not resolved:
            # Nominal maturity has passed but the matched trading day is unavailable/too far - an
            # unresolved gap, not a resolved outcome.
            full_tenor_observable = False
        else:
            j = pos
            maturity_actual_date = actual_date

    autocalled = False
    autocall_month = None
    autocall_date = None
    obs_records = []

    for m in obs_offsets:
        obs_calendar = launch_date + pd.DateOffset(months=m)
        rec = {"month": m, "scheduled": obs_calendar}

        if autocalled:
            rec["status"] = "NOT_APPLICABLE_ALREADY_AUTOCALLED"
            obs_records.append(rec)
            continue
        if obs_calendar > data_as_of:
            rec["status"] = "NOT_YET_OBSERVABLE"
            obs_records.append(rec)
            continue

        pos, actual_date, gap, resolved = map_observation(obs_calendar, all_dates, max_gap_days)
        if pos is None:
            rec["status"] = "NOT_YET_OBSERVABLE"
            obs_records.append(rec)
            continue
        if j is not None and pos >= j:
            # Must resolve strictly before maturity's own trading day to count as an early-autocall
            # check - maturity's terminal test already covers that day (see DEVLOG.md).
            rec["status"] = "COINCIDES_WITH_OR_AFTER_MATURITY_SKIPPED"
            obs_records.append(rec)
            continue
        if not resolved:
            rec.update(actual=actual_date, gap_days=gap, status="UNRESOLVED_GAP")
            obs_records.append(rec)
            continue

        assert pos > i, (
            f"observation for launch {launch_date.date()} resolved at/before the launch date itself "
            f"(pos={pos}, launch pos={i}) - this would be a month-zero autocall and must never happen"
        )
        ratio_vec = prices[pos] / s0
        worst_ratio = float(ratio_vec.min())
        rec.update(actual=actual_date, gap_days=gap, ratio=worst_ratio, status="OBSERVED")
        obs_records.append(rec)

        if worst_ratio >= trigger:
            autocalled = True
            autocall_month = m
            autocall_date = actual_date

    # Descriptive-only (not used by the classification above): the underlying's own max/terminal
    # return over the window observed so far - lets the cohort table sanity-check the outcome mix.
    window_end_pos = j if j is not None else (len(all_dates) - 1)
    window_ratio = (prices[i:window_end_pos + 1] / s0).min(axis=1)
    max_return_pct = float(window_ratio.max() - 1) * 100
    terminal_return_pct = float(window_ratio[-1] - 1) * 100

    if autocalled:
        outcome = "AUTOCALL"
        final_valuation_date = autocall_date
        recovery_fraction = None
    elif full_tenor_observable:
        worst_at_maturity = float((prices[j] / s0).min())
        final_valuation_date = maturity_actual_date
        if worst_at_maturity >= strike:
            outcome = "MATURITY_CASH"
            recovery_fraction = None
        else:
            outcome = "PHYSICAL_DELIVERY"
            recovery_fraction = worst_at_maturity / strike
    else:
        outcome = "OUTSTANDING"
        final_valuation_date = None
        recovery_fraction = None

    return {
        "launch_date": launch_date,
        "launch_pos": i,
        "s0": s0,
        "maturity_calendar": maturity_calendar,
        "maturity_actual_date": maturity_actual_date,
        "full_tenor_observable": full_tenor_observable,
        "obs_records": obs_records,
        "outcome": outcome,
        "autocall_month": autocall_month,
        "autocall_date": autocall_date,
        "final_valuation_date": final_valuation_date,
        "recovery_fraction": recovery_fraction,
        "max_return_pct": max_return_pct,
        "terminal_return_pct": terminal_return_pct,
        "terminal_is_at_maturity": j is not None,
    }


def completed_tenor_positions(all_dates, tenor_months, launch_start, launch_end, data_as_of,
                               max_gap_days=MAX_OBSERVATION_GAP_DAYS):
    """(launch position, maturity position) pairs restricted to launches whose NOMINAL maturity is
    on or before data_as_of and resolves cleanly. Feeds diagnostic_stats only - real classification
    logic lives in classify_launch."""
    candidates = np.where((all_dates >= launch_start) & (all_dates <= launch_end))[0]
    out = []
    for i in candidates:
        maturity_calendar = all_dates[i] + pd.DateOffset(months=tenor_months)
        if maturity_calendar > data_as_of:
            continue
        pos, _actual_date, _gap, resolved = map_observation(maturity_calendar, all_dates, max_gap_days)
        if pos is not None and resolved:
            out.append((int(i), int(pos)))
    return out


# ---------------------------------------------------------------------------
# ROLLING BACKTEST
# ---------------------------------------------------------------------------

def run_backtest(price_df, terms, launch_start, launch_end, data_as_of):
    """One pass over every trading-day launch in [launch_start, launch_end]. Returns (results,
    coverage) - coverage reports requested vs. actually-available launch/data coverage."""
    all_dates = price_df.index
    prices = price_df.to_numpy()
    obs_offsets = build_observation_offsets(
        terms["TENOR_MONTHS"], terms["AUTOCALL_FREQUENCY_MONTHS"], terms["AUTOCALL_LOCKOUT_MONTHS"]
    )

    candidate_positions = np.where((all_dates >= launch_start) & (all_dates <= launch_end))[0]
    if len(candidate_positions) == 0:
        raise RuntimeError(
            f"No trading day between {launch_start.date()} and {launch_end.date()} is present in the "
            f"fetched data ({all_dates[0].date()} to {all_dates[-1].date()})."
        )

    results = [
        classify_launch(prices, all_dates, int(i), terms["TENOR_MONTHS"], obs_offsets,
                         terms["STRIKE"], terms["AUTOCALL_TRIGGER"], data_as_of)
        for i in candidate_positions
    ]

    data_start, data_end = all_dates[0], all_dates[-1]
    coverage = {
        "requested_launch_start": launch_start,
        "requested_launch_end": launch_end,
        "data_as_of": data_as_of,
        "data_start": data_start,
        "data_end": data_end,
        "first_actual_launch": all_dates[candidate_positions[0]],
        "last_actual_launch": all_dates[candidate_positions[-1]],
        "n_launches_tested": len(candidate_positions),
        "insufficient_history": bool(data_start > launch_start),
    }
    return results, coverage


# ---------------------------------------------------------------------------
# AUDIT TABLE + INDEPENDENT CROSS-CHECK
# ---------------------------------------------------------------------------

def build_audit_dataframe(results, terms, obs_offsets):
    """One row per launch - initial prices, absolute strike/trigger levels, scheduled+actual
    observation dates/ratios, first autocall date, final valuation date, outcome, recovery
    fraction. This IS the audit trail every reported number can be recomputed from."""
    tickers = terms["TICKERS"]
    rows = []
    for r in results:
        row = {"Launch": r["launch_date"]}
        for k, t in enumerate(tickers):
            row[f"Initial_{t}"] = float(r["s0"][k])
            row[f"Strike_Abs_{t}"] = float(r["s0"][k]) * terms["STRIKE"]
            row[f"Trigger_Abs_{t}"] = float(r["s0"][k]) * terms["AUTOCALL_TRIGGER"]
        for rec in r["obs_records"]:
            m = rec["month"]
            row[f"Obs{m}_Scheduled"] = rec.get("scheduled")
            row[f"Obs{m}_Actual"] = rec.get("actual")
            row[f"Obs{m}_Ratio"] = rec.get("ratio")
            row[f"Obs{m}_Status"] = rec.get("status")
        row["FirstAutocallMonth"] = r["autocall_month"]
        row["FirstAutocallDate"] = r["autocall_date"]
        row["FinalValuationDate"] = r["final_valuation_date"]
        row["Outcome"] = r["outcome"]
        row["RecoveryFraction"] = r["recovery_fraction"]
        row["FullTenorObservable"] = r["full_tenor_observable"]
        row["MaxReturnPct"] = r["max_return_pct"]
        row["TerminalReturnPct"] = r["terminal_return_pct"]
        rows.append(row)
    return pd.DataFrame(rows)


def cross_check_autocall_rate(audit_df, terms, obs_offsets):
    """Independently reconstructs AUTOCALL purely from the stored Obs*_Ratio/Obs*_Status audit
    columns - a separate code path from classify_launch - and asserts an exact match, launch by
    launch (see DEVLOG.md "Cross-check as a second, independent code path")."""
    trigger = terms["AUTOCALL_TRIGGER"]

    def reconstruct(row):
        for m in obs_offsets:
            status = row.get(f"Obs{m}_Status")
            if status == "OBSERVED":
                if row[f"Obs{m}_Ratio"] >= trigger:
                    return m
            elif status == "NOT_YET_OBSERVABLE":
                break
            # COINCIDES_WITH_OR_AFTER_MATURITY_SKIPPED / UNRESOLVED_GAP / NOT_APPLICABLE_ALREADY_
            # AUTOCALLED all mean "keep looking", matching classify_launch's `continue`.
        return None

    reconstructed_month = audit_df.apply(reconstruct, axis=1)
    reconstructed_autocall = reconstructed_month.notna()
    recorded_autocall = audit_df["Outcome"].eq("AUTOCALL")

    mismatch = audit_df[reconstructed_autocall != recorded_autocall]
    assert mismatch.empty, (
        f"Cross-check FAILED: {len(mismatch)} launch(es) where the audit-table reconstruction "
        f"disagrees with the recorded Outcome. First mismatch:\n{mismatch.iloc[0]}"
    )
    month_mismatch = audit_df.loc[recorded_autocall, "FirstAutocallMonth"] != reconstructed_month[recorded_autocall]
    assert not month_mismatch.any(), "Cross-check FAILED: reconstructed autocall month disagrees with recorded month"

    n = len(audit_df)
    n_reconstructed = int(reconstructed_autocall.sum())
    rate = n_reconstructed / n if n else float("nan")
    print(f"Independent cross-check: AUTOCALL reconstructed from stored audit columns = {n_reconstructed}/{n} "
          f"({rate:.1%}) - exact match against the reported Outcome column.")
    return rate


# ---------------------------------------------------------------------------
# DESCRIPTIVE DIAGNOSTICS (independent of the product's own STRIKE/TRIGGER)
# ---------------------------------------------------------------------------

def diagnostic_stats(price_df, tenor_months, frequency_months, lockout_months,
                      launch_start, launch_end, data_as_of):
    """Descriptive-only summary of this product's OWN underlying(s), restricted to the completed-
    tenor cohort: for each candidate trigger, fraction of launches that would have cleared it at
    SOME observation; for each candidate strike, fraction that would have finished below it at
    maturity IGNORING autocall."""
    all_dates = price_df.index
    prices = price_df.to_numpy()
    obs_offsets = build_observation_offsets(tenor_months, frequency_months, lockout_months)
    launches = completed_tenor_positions(all_dates, tenor_months, launch_start, launch_end, data_as_of)

    if not launches:
        return {"n": 0, "trigger_hit_rates": {}, "strike_breach_rates_ignoring_autocall": {}}

    max_at_obs = np.empty(len(launches))
    worst_at_maturity = np.empty(len(launches))
    for k, (i, j) in enumerate(launches):
        window = prices[i:j + 1]
        worst_of = (window / window[0]).min(axis=1)
        obs_positions = []
        for m in obs_offsets:
            pos, _actual, _gap, resolved = map_observation(all_dates[i] + pd.DateOffset(months=m), all_dates)
            if pos is not None and resolved and i <= pos < i + len(worst_of):
                obs_positions.append(pos - i)
        max_at_obs[k] = worst_of[obs_positions].max() if obs_positions else -np.inf
        worst_at_maturity[k] = worst_of[-1]

    return {
        "n": len(launches),
        "trigger_hit_rates": {t: float((max_at_obs >= t).mean()) for t in DIAGNOSTIC_TRIGGER_CANDIDATES},
        "strike_breach_rates_ignoring_autocall": {s: float((worst_at_maturity < s).mean()) for s in DIAGNOSTIC_STRIKE_CANDIDATES},
    }


def print_diagnostics(basket_desc, price_df, tickers, terms, launch_start, launch_end, data_as_of):
    print(f"\nObserved underlying behavior - {basket_desc}:")
    for t in tickers:
        vol = realized_annualized_vol_log(price_df[t])
        print(f"  {t}: annualized realized vol (log returns, full fetched history) = {vol:.1%}")

    diag = diagnostic_stats(price_df, terms["TENOR_MONTHS"], terms["AUTOCALL_FREQUENCY_MONTHS"],
                             terms["AUTOCALL_LOCKOUT_MONTHS"], launch_start, launch_end, data_as_of)
    if diag["n"] == 0:
        print("  No completed-tenor window available yet for this product/launch period - skipping diagnostic grid.")
        return
    print(f"  Based on {diag['n']} historical {terms['TENOR_MONTHS']}m windows in the completed-tenor cohort "
          f"(ignoring this product's own STRIKE/AUTOCALL_TRIGGER):")
    print(f"    P(worst-of clears trigger at SOME observation)   P(worst-of finishes below strike at maturity, no autocall)")
    triggers = list(diag["trigger_hit_rates"].items())
    strikes = list(diag["strike_breach_rates_ignoring_autocall"].items())
    for (trig, p_trig), (strike, p_strike) in zip(triggers, strikes):
        print(f"      trigger {trig:>5.0%}: {p_trig:>6.1%}                              strike {strike:>5.0%}: {p_strike:>6.1%}")


def yearly_breakdown(results):
    """Groups launches by calendar year of launch. A recent year will typically show a large
    OUTSTANDING share - that's expected, not a bug."""
    cohorts = {}
    for r in results:
        year = r["launch_date"].year
        c = cohorts.setdefault(year, {
            "n": 0, "autocall": 0, "maturity_cash": 0, "delivery": 0, "outstanding": 0,
            "sum_max_return_pct": 0.0, "sum_terminal_return_pct": 0.0,
        })
        c["n"] += 1
        c["sum_max_return_pct"] += r["max_return_pct"]
        c["sum_terminal_return_pct"] += r["terminal_return_pct"]
        if r["outcome"] == "AUTOCALL":
            c["autocall"] += 1
        elif r["outcome"] == "MATURITY_CASH":
            c["maturity_cash"] += 1
        elif r["outcome"] == "PHYSICAL_DELIVERY":
            c["delivery"] += 1
        else:
            c["outstanding"] += 1
    return dict(sorted(cohorts.items()))


# ---------------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------------

def plot_outcome_mix_by_year(cohorts, terms, name, basket_desc, worst_of_note, n_total, results,
                              data_as_of, safe_name):
    """100%-stacked bar chart, one bar per launch year, of the outcome mix among that year's
    launches."""
    years = list(cohorts.keys())
    autocall_pct = [cohorts[y]["autocall"] / cohorts[y]["n"] * 100 for y in years]
    maturity_pct = [cohorts[y]["maturity_cash"] / cohorts[y]["n"] * 100 for y in years]
    delivery_pct = [cohorts[y]["delivery"] / cohorts[y]["n"] * 100 for y in years]
    outstanding_pct = [cohorts[y]["outstanding"] / cohorts[y]["n"] * 100 for y in years]

    fig, ax = plt.subplots(figsize=(max(8, len(years) * 0.4), 5.5))
    ax.bar(years, autocall_pct, color=OUTCOME_COLORS["AUTOCALL"], width=0.7, label="AUTOCALL (early, at par)")
    bottom1 = autocall_pct
    ax.bar(years, maturity_pct, bottom=bottom1, color=OUTCOME_COLORS["MATURITY_CASH"], width=0.7,
           label="MATURITY_CASH (full tenor, capital back)")
    bottom2 = [a + m for a, m in zip(autocall_pct, maturity_pct)]
    ax.bar(years, delivery_pct, bottom=bottom2, color=OUTCOME_COLORS["PHYSICAL_DELIVERY"], width=0.7,
           label="PHYSICAL_DELIVERY (capital loss)")
    bottom3 = [b + d for b, d in zip(bottom2, delivery_pct)]
    ax.bar(years, outstanding_pct, bottom=bottom3, color=OUTCOME_COLORS["OUTSTANDING"], width=0.7,
           label="OUTSTANDING (not yet resolved)")
    ax.set_ylabel("Share of that launch year's daily launches (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=8)
    ax.grid(True, axis="y", which="major", color="lightgrey", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(
        f"FCN Rolling Backtest - {name}: {basket_desc}{worst_of_note}, by launch year\n"
        f"Strike {terms['STRIKE']:.0%} / Autocall {terms['AUTOCALL_TRIGGER']:.0%} / Tenor {terms['TENOR_MONTHS']}m "
        f"({n_total} launches, {results[0]['launch_date'].date()} to {results[-1]['launch_date'].date()}, "
        f"data as-of {data_as_of.date()})",
        fontsize=10,
    )
    fig.tight_layout()
    output_png = os.path.join(SCRIPT_DIR, f"{safe_name}.png")
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_png


def plot_growth_with_launch_outcomes(price_df, tickers, results, terms, name, basket_desc,
                                      worst_of_note, data_as_of, safe_name):
    """Underlying's (or worst-of basket's) growth path, normalized to 1.0 at the start of the
    fetched history, with one dot per tested launch date colored by that launch's outcome. Visual
    sanity-check for a headline percentage that looks implausible - see DEVLOG.md "Growth chart
    with launch dots"."""
    growth = (price_df[tickers] / price_df[tickers].iloc[0]).min(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(growth.index, growth.to_numpy(), color="black", linewidth=0.8, alpha=0.6, zorder=1)

    for outcome, color in OUTCOME_COLORS.items():
        dates = [r["launch_date"] for r in results if r["outcome"] == outcome]
        if not dates:
            continue
        ax.scatter(dates, growth.loc[dates].to_numpy(), s=8, color=color, alpha=0.5,
                   linewidths=0, label=f"{outcome} launch", zorder=2)

    ax.set_ylabel("Worst-of growth vs. start of fetched history (x)" if len(tickers) > 1
                  else "Price growth vs. start of fetched history (x)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=8, markerscale=2)
    ax.grid(True, axis="y", which="major", color="lightgrey", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title(
        f"FCN Rolling Backtest - {name}: {basket_desc}{worst_of_note}\n"
        f"Underlying growth with launch outcomes (data as-of {data_as_of.date()})",
        fontsize=10,
    )
    fig.tight_layout()
    output_png = os.path.join(SCRIPT_DIR, f"{safe_name} - growth.png")
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_png


# ---------------------------------------------------------------------------
# SYNTHETIC CORRECTNESS TESTS
# ---------------------------------------------------------------------------

def run_synthetic_tests():
    """Self-contained correctness tests against constructed price paths - covers each outcome
    bucket, boundary equalities, holiday/gap handling, cutoff/OUTSTANDING handling, and input
    validation. Run before every real backtest (see __main__) - if any fail, the classification
    logic is broken and the real numbers below aren't trustworthy."""
    print(f"\n{'=' * 72}\nSYNTHETIC CORRECTNESS TESTS\n{'=' * 72}")

    tenor_months = 12
    obs_offsets = build_observation_offsets(tenor_months, 3, 0)
    assert obs_offsets == [3, 6, 9], obs_offsets
    assert 0 not in obs_offsets
    strike, trigger = 0.70, 1.00
    n_run = 0

    def make_dates(n, start="2020-01-06", drop=None):
        dates = pd.bdate_range(start=start, periods=n)
        if drop:
            dates = dates.delete(sorted(set(int(d) for d in drop if 0 <= d < len(dates))))
        return dates

    def run_case(prices_1d, dates, i, data_as_of):
        prices = np.asarray(prices_1d, dtype=float).reshape(-1, 1)
        return classify_launch(prices, dates, i, tenor_months, obs_offsets, strike, trigger,
                                pd.Timestamp(data_as_of))

    n_days = 380  # comfortably > 12 months of business days
    dates = make_dates(n_days)
    data_as_of_full = dates[-1]
    m3_pos = int(dates.searchsorted(dates[0] + pd.DateOffset(months=3)))
    j_pos = int(dates.searchsorted(dates[0] + pd.DateOffset(months=tenor_months)))

    # (a) AUTOCALL: worst-of rises above trigger by month 3
    prices = np.full(n_days, 100.0)
    prices[m3_pos:] = 110.0
    r = run_case(prices, dates, 0, data_as_of_full)
    assert r["outcome"] == "AUTOCALL" and r["autocall_month"] == 3, r
    n_run += 1

    # (b) MATURITY_CASH: never triggers, finishes >= strike
    prices = np.full(n_days, 100.0)
    prices[1:] = 90.0
    r = run_case(prices, dates, 0, data_as_of_full)
    assert r["outcome"] == "MATURITY_CASH", r
    n_run += 1

    # (c) PHYSICAL_DELIVERY: never triggers, finishes below strike
    prices = np.full(n_days, 100.0)
    prices[1:] = 60.0
    r = run_case(prices, dates, 0, data_as_of_full)
    assert r["outcome"] == "PHYSICAL_DELIVERY", r
    assert abs(r["recovery_fraction"] - (0.60 / 0.70)) < 1e-9, r
    n_run += 1

    # (d) equality AT the trigger -> AUTOCALL (inclusive boundary)
    prices = np.full(n_days, 100.0)
    prices[m3_pos:] = 100.0
    r = run_case(prices, dates, 0, data_as_of_full)
    assert r["outcome"] == "AUTOCALL", r
    n_run += 1

    # (e) equality AT the strike at maturity -> MATURITY_CASH (inclusive boundary)
    prices = np.full(n_days, 100.0)
    prices[1:] = 90.0
    prices[j_pos] = 70.0
    r = run_case(prices, dates, 0, data_as_of_full)
    assert r["outcome"] == "MATURITY_CASH", r
    n_run += 1

    # (f) no launch-day autocall: day-0 ratio is trivially 1.00 (=trigger) but day 0 is never
    # scheduled, so a note that dips right after launch and never recovers must NOT be AUTOCALL.
    prices = np.full(n_days, 100.0)
    prices[1:] = 50.0
    r = run_case(prices, dates, 0, data_as_of_full)
    assert r["outcome"] != "AUTOCALL", r
    assert r["autocall_month"] is None, r
    n_run += 1

    # (g) holiday roll: drop the exact scheduled month-3 day; observation rolls to the next
    # available day with a small, resolved gap.
    dates_holiday = make_dates(n_days, drop=[m3_pos])
    prices = np.full(n_days - 1, 100.0)
    prices[m3_pos:] = 110.0  # index shifts down by one after the drop
    r = run_case(prices, dates_holiday, 0, dates_holiday[-1])
    assert r["outcome"] == "AUTOCALL" and r["autocall_month"] == 3, r
    obs3 = next(o for o in r["obs_records"] if o["month"] == 3)
    assert obs3["status"] == "OBSERVED" and 0 < obs3["gap_days"] <= MAX_OBSERVATION_GAP_DAYS, obs3
    n_run += 1

    # (h) missing data: a large hole spanning month 3 must be flagged UNRESOLVED, not silently
    # evaluated - a spike placed right after the hole must not trigger a spurious autocall.
    big_gap_drop = range(m3_pos - 20, m3_pos + 20)
    dates_gap = make_dates(n_days, drop=big_gap_drop)
    prices = np.full(len(dates_gap), 90.0)
    prices[0] = 100.0
    spike_pos = int(dates_gap.searchsorted(dates[0] + pd.DateOffset(months=3)))
    prices[spike_pos:spike_pos + 2] = 500.0
    r = run_case(prices, dates_gap, 0, dates_gap[-1])
    obs3 = next(o for o in r["obs_records"] if o["month"] == 3)
    assert obs3["status"] == "UNRESOLVED_GAP", obs3
    assert r["outcome"] != "AUTOCALL", r
    n_run += 1

    # (i) cutoff before the first scheduled observation -> OUTSTANDING
    r = run_case(np.full(n_days, 100.0), dates, 0, dates[0] + pd.Timedelta(days=10))
    assert r["outcome"] == "OUTSTANDING", r
    assert all(o["status"] == "NOT_YET_OBSERVABLE" for o in r["obs_records"]), r["obs_records"]
    n_run += 1

    # (j) some observations resolved (none triggered), but maturity still beyond the cutoff
    prices = np.full(n_days, 100.0)
    prices[1:] = 90.0
    r = run_case(prices, dates, 0, dates[0] + pd.DateOffset(months=8))
    assert r["outcome"] == "OUTSTANDING", r
    assert r["final_valuation_date"] is None, r
    observed_months = [o["month"] for o in r["obs_records"] if o["status"] == "OBSERVED"]
    assert observed_months == [3, 6], observed_months
    n_run += 1

    # (k) an observation resolving to the SAME actual trading day as maturity (collapsed calendar)
    # must be excluded from the early-autocall check, not treated as an early autocall.
    m9_calendar = dates[0] + pd.DateOffset(months=9)
    m9_pos_orig = int(dates.searchsorted(m9_calendar))
    drop_collapse = range(m9_pos_orig, j_pos)
    dates_collapsed = make_dates(n_days, drop=drop_collapse)
    prices = np.full(len(dates_collapsed), 100.0)
    prices[1:] = 90.0
    collapsed_pos = int(dates_collapsed.searchsorted(m9_calendar))
    prices[collapsed_pos:] = 100.0  # exactly at trigger AND exactly what maturity will read
    r = run_case(prices, dates_collapsed, 0, dates_collapsed[-1])
    obs9 = next((o for o in r["obs_records"] if o["month"] == 9), None)
    assert obs9 is not None and obs9["status"] == "COINCIDES_WITH_OR_AFTER_MATURITY_SKIPPED", obs9
    assert r["autocall_month"] is None, r
    assert r["outcome"] == "MATURITY_CASH", r
    n_run += 1

    # (l) insufficient-history coverage flag
    dates_short = make_dates(200, start="2023-01-02")
    price_df_short = pd.DataFrame(np.full((200, 1), 100.0), index=dates_short, columns=["TEST"])
    requested_start = dates_short[0] - pd.Timedelta(days=400)
    short_terms = dict(STRIKE=0.7, AUTOCALL_TRIGGER=1.0, TENOR_MONTHS=12,
                        AUTOCALL_FREQUENCY_MONTHS=3, AUTOCALL_LOCKOUT_MONTHS=0)
    _results_s, coverage_s = run_backtest(price_df_short, short_terms, requested_start,
                                           dates_short[-1], dates_short[-1])
    assert coverage_s["insufficient_history"] is True, coverage_s
    assert coverage_s["data_start"] == dates_short[0], coverage_s
    n_run += 1

    # (m) validation: lockout, tickers, price sanity, date-index sanity
    def expect_value_error(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"expected ValueError, none raised ({fn})")

    base = dict(TICKERS=["A"], STRIKE=0.7, AUTOCALL_TRIGGER=1.0, TENOR_MONTHS=12,
                AUTOCALL_FREQUENCY_MONTHS=3, AUTOCALL_LOCKOUT_MONTHS=0)
    expect_value_error(lambda: validate_terms("X", {**base, "AUTOCALL_LOCKOUT_MONTHS": -1}))
    expect_value_error(lambda: validate_terms("X", {**base, "TICKERS": ["A", "A"]}))
    expect_value_error(lambda: validate_terms("X", {**base, "TICKERS": []}))
    expect_value_error(lambda: validate_terms("X", {**base, "AUTOCALL_FREQUENCY_MONTHS": 0}))

    bad_price_df = pd.DataFrame({"TEST": [100.0, -5.0, 101.0]}, index=pd.bdate_range("2024-01-02", periods=3))
    expect_value_error(lambda: validate_price_df(bad_price_df, ["TEST"]))

    nan_price_df = pd.DataFrame({"TEST": [100.0, np.nan, 101.0]}, index=pd.bdate_range("2024-01-02", periods=3))
    expect_value_error(lambda: validate_price_df(nan_price_df, ["TEST"]))

    dup_idx_df = pd.DataFrame({"TEST": [100.0, 101.0]}, index=[pd.Timestamp("2024-01-02")] * 2)
    expect_value_error(lambda: validate_price_df(dup_idx_df, ["TEST"]))
    n_run += 1

    print(f"All {n_run} synthetic test groups passed.")


# ---------------------------------------------------------------------------
# MAIN PER-PRODUCT REPORT
# ---------------------------------------------------------------------------

def backtest_product(terms):
    name = product_label(terms)
    tickers = terms["TICKERS"]
    underlying_names = [fetch_underlying_name(t) for t in tickers]
    basket_desc = " / ".join(f"{n} ({t})" for n, t in zip(underlying_names, tickers))
    worst_of_note = " (worst-of basket)" if len(tickers) > 1 else ""

    tenor_months = terms["TENOR_MONTHS"]
    freq = terms["AUTOCALL_FREQUENCY_MONTHS"]
    lockout = terms["AUTOCALL_LOCKOUT_MONTHS"]
    strike, trigger = terms["STRIKE"], terms["AUTOCALL_TRIGGER"]
    obs_offsets = build_observation_offsets(tenor_months, freq, lockout)

    launch_start, launch_end, data_as_of = resolve_run_dates()

    print(f"\n{'#' * 72}\n{name}\n{'#' * 72}")
    print(f"Requested launch period: {launch_start.date()} to {launch_end.date()}  |  Data as-of: {data_as_of.date()}")
    print(f"Business-day convention: observations roll forward to the next trading day ('following'); "
          f"a roll of more than {MAX_OBSERVATION_GAP_DAYS} calendar days is flagged as an unresolved data gap.")

    fetch_start = launch_start
    fetch_end = data_as_of + pd.Timedelta(days=1)  # yfinance `end` is exclusive of the boundary itself
    print(f"Fetching {basket_desc}{worst_of_note} from {fetch_start.date()} through {data_as_of.date()}...")
    price_df = fetch_multi_asset_path(tickers, fetch_start, fetch_end)
    price_df = price_df[price_df.index <= data_as_of]  # hard cutoff - never use data past DATA_AS_OF
    validate_price_df(price_df, tickers)

    data_start, data_end = price_df.index[0], price_df.index[-1]
    print(f"  {len(price_df)} trading days available, {data_start.date()} to {data_end.date()}")
    insufficient_history = data_start > launch_start
    if insufficient_history:
        print(f"  WARNING - insufficient history: requested LAUNCH_START {launch_start.date()} predates "
              f"available data ({data_start.date()}); launches before {data_start.date()} were NOT tested.")
    if data_end < launch_end:
        print(f"  WARNING: available data ends {data_end.date()}, before requested LAUNCH_END "
              f"{launch_end.date()}; tested launch period was clipped to the data actually available.")

    print_diagnostics(f"{basket_desc}{worst_of_note}", price_df, tickers, terms, launch_start, launch_end, data_as_of)

    results, coverage = run_backtest(price_df, terms, launch_start, launch_end, data_as_of)
    n_total = len(results)

    audit_df = build_audit_dataframe(results, terms, obs_offsets)
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in name)
    audit_csv = os.path.join(SCRIPT_DIR, f"{safe_name} - audit.csv")
    audit_df.to_csv(audit_csv, index=False)
    print(f"\nPer-launch audit table ({n_total} rows) saved to:\n  {audit_csv}")

    cross_check_rate = cross_check_autocall_rate(audit_df, terms, obs_offsets)

    counts_all = Counter(r["outcome"] for r in results)
    n_autocall = counts_all.get("AUTOCALL", 0)
    n_maturity_cash = counts_all.get("MATURITY_CASH", 0)
    n_delivery = counts_all.get("PHYSICAL_DELIVERY", 0)
    n_outstanding = counts_all.get("OUTSTANDING", 0)

    # --- Reconciliation assertions: outcome counts and percentages ---
    assert n_autocall + n_maturity_cash + n_delivery + n_outstanding == n_total, (
        "outcome counts do not reconcile to the total launch count"
    )
    reported_rate = n_autocall / n_total if n_total else float("nan")
    assert abs(reported_rate - cross_check_rate) < 1e-12, (reported_rate, cross_check_rate)
    pct_sum_all = (n_autocall + n_maturity_cash + n_delivery + n_outstanding) / n_total
    assert abs(pct_sum_all - 1.0) < 1e-9, "all-launches percentages do not sum to 100%"

    n_reached_maturity = n_maturity_cash + n_delivery
    n_repaid_at_par = n_autocall + n_maturity_cash

    month_counts = Counter(r["autocall_month"] for r in results if r["outcome"] == "AUTOCALL")
    assert sum(month_counts.values()) == n_autocall, "per-month first-autocall counts do not sum to total autocalls"

    completed = [r for r in results if r["full_tenor_observable"]]
    n_completed = len(completed)
    counts_completed = Counter(r["outcome"] for r in completed)
    assert counts_completed.get("OUTSTANDING", 0) == 0, "completed-tenor cohort must never contain OUTSTANDING launches"
    if n_completed:
        assert (counts_completed.get("AUTOCALL", 0) + counts_completed.get("MATURITY_CASH", 0)
                + counts_completed.get("PHYSICAL_DELIVERY", 0)) == n_completed, (
            "completed-tenor cohort outcome counts do not reconcile"
        )
        pct_sum_completed = (counts_completed.get("AUTOCALL", 0) + counts_completed.get("MATURITY_CASH", 0)
                              + counts_completed.get("PHYSICAL_DELIVERY", 0)) / n_completed
        assert abs(pct_sum_completed - 1.0) < 1e-9, "completed-tenor cohort percentages do not sum to 100%"

    print(f"\n{'-' * 72}")
    print(f"Strike: {strike:.0%}  |  Autocall trigger: {trigger:.0%}  (both vs. each underlying's own launch spot)")
    print(f"Tenor: {tenor_months}m  |  Autocall observations: every {freq}m (lockout {lockout}m), months {obs_offsets}")
    print(f"Launches tested: {n_total} daily, {results[0]['launch_date'].date()} to {results[-1]['launch_date'].date()}")
    print(f"Requested vs. actual coverage: requested {coverage['requested_launch_start'].date()} to "
          f"{coverage['requested_launch_end'].date()}; data available {coverage['data_start'].date()} to "
          f"{coverage['data_end'].date()}; insufficient history: {coverage['insufficient_history']}")
    print(f"{'-' * 72}")
    print(f"ALL LAUNCHES (status as of {data_as_of.date()}, includes notes still outstanding):")
    print(f"  AUTOCALL             {n_autocall:>6} / {n_total}  ({n_autocall / n_total:.1%})")
    print(f"  MATURITY_CASH        {n_maturity_cash:>6} / {n_total}  ({n_maturity_cash / n_total:.1%})")
    print(f"  PHYSICAL_DELIVERY    {n_delivery:>6} / {n_total}  ({n_delivery / n_total:.1%})")
    print(f"  OUTSTANDING          {n_outstanding:>6} / {n_total}  ({n_outstanding / n_total:.1%})")
    print(f"  Reached maturity (MATURITY_CASH + PHYSICAL_DELIVERY): {n_reached_maturity} ({n_reached_maturity / n_total:.1%})")
    print(f"  Repaid at par, subtotal (AUTOCALL + MATURITY_CASH):   {n_repaid_at_par} ({n_repaid_at_par / n_total:.1%})"
          f"  <- a subtotal of two outcomes, not a capital-protection guarantee")
    print(f"{'-' * 72}")
    print(f"First autocall by observation month (must sum to total AUTOCALL = {n_autocall}):")
    for m in obs_offsets:
        c = month_counts.get(m, 0)
        of_autocall = f", {c / n_autocall:.1%} of autocalls" if n_autocall else ""
        print(f"  Month {m:>2}: {c:>6}  ({c / n_total:.1%} of all launches{of_autocall})")
    assert sum(month_counts.get(m, 0) for m in obs_offsets) == n_autocall
    print(f"{'-' * 72}")
    if n_completed:
        print(f"COMPLETED-TENOR COHORT ONLY (nominal maturity <= {data_as_of.date()}; n={n_completed} of "
              f"{n_total}; excludes still-outstanding and still-open-early-autocall launches - see README "
              f"\"The four outcomes reported\"):")
        for label in ("AUTOCALL", "MATURITY_CASH", "PHYSICAL_DELIVERY"):
            c = counts_completed.get(label, 0)
            print(f"  {label:<18}{c:>6} / {n_completed}  ({c / n_completed:.1%})")
    else:
        print("COMPLETED-TENOR COHORT: empty - every launch in this window is still OUTSTANDING.")
    print(f"{'-' * 72}")
    if n_delivery:
        avg_recovery = np.mean([r["recovery_fraction"] for r in results if r["outcome"] == "PHYSICAL_DELIVERY"])
        print(f"Average recovery on physical delivery (as % of par): {avg_recovery:.1%}")
    print(f"Cross-check: audit-table-reconstructed AUTOCALL rate = {cross_check_rate:.1%}  "
          f"(matches the reported rate {reported_rate:.1%} exactly)")
    print(f"For a percentages-only quick report with no audit CSV/charts, run \"FCN Condensed.py\".")

    # --- By launch-year cohort ---
    cohorts = yearly_breakdown(results)
    print(f"\n{'-' * 72}\nBy launch-year cohort (a year straddling or after {data_as_of.date()} will show a "
          f"nonzero OUTSTANDING share - expected, not a bug):")
    print(f"  Outcome mix is a % of that year's launches; the two return columns are the underlying's OWN "
          f"return (not the note's payoff) - cross-check the outcome mix against them directly.")
    print(f"  {'Year':<6}{'n':>6}{'Autocall':>10}{'MatCash':>10}{'Delivery':>10}{'Outst.':>9}   |{'Avg max ret.':>14}{'Avg term. ret.':>16}")
    for year, c in cohorts.items():
        avg_max = c["sum_max_return_pct"] / c["n"]
        avg_term = c["sum_terminal_return_pct"] / c["n"]
        print(f"  {year:<6}{c['n']:>6}{c['autocall'] / c['n']:>9.1%} {c['maturity_cash'] / c['n']:>9.1%} "
              f"{c['delivery'] / c['n']:>9.1%} {c['outstanding'] / c['n']:>8.1%}   |{avg_max:>+13.1f}%{avg_term:>+13.1f}%")

    bar_png = plot_outcome_mix_by_year(cohorts, terms, name, basket_desc, worst_of_note, n_total, results,
                                        data_as_of, safe_name)
    growth_png = plot_growth_with_launch_outcomes(price_df, tickers, results, terms, name, basket_desc,
                                                   worst_of_note, data_as_of, safe_name)
    print(f"Charts saved to {bar_png}\n  and {growth_png}")

    return {
        "name": name, "n_total": n_total, "n_autocall": n_autocall, "n_maturity_cash": n_maturity_cash,
        "n_delivery": n_delivery, "n_outstanding": n_outstanding, "n_completed": n_completed,
        "counts_completed": counts_completed,
    }


if __name__ == "__main__":
    try:
        run_synthetic_tests()
    except AssertionError as exc:
        print(f"\n{'!' * 72}\nSYNTHETIC TEST FAILURE - aborting before running the real backtest.\n{'!' * 72}")
        print(f"  {exc}")
        raise

    print(
        f"\n{'!' * 72}\n"
        "OVERLAPPING-LAUNCH CAVEAT: launches happen every trading day and overlap almost entirely, so\n"
        "they are NOT independent trials - treat any 'N launches, X% autocalled' figure as a description\n"
        "of this one historical path, not a frequentist probability estimate. See README.md \"Known\n"
        "biases\" and \"SEPARATE CAVEAT\" sections before trusting any percentage below.\n"
        f"{'!' * 72}\n"
    )

    summaries = [backtest_product(terms) for terms in PRODUCTS]

    print(f"\n{'=' * 72}\nSUMMARY ACROSS PRODUCTS\n{'=' * 72}")
    for s in summaries:
        n = s["n_total"]
        print(f"{s['name']}: autocall {s['n_autocall'] / n:.1%}  |  maturity_cash {s['n_maturity_cash'] / n:.1%}  "
              f"|  delivery {s['n_delivery'] / n:.1%}  |  outstanding {s['n_outstanding'] / n:.1%}  ({n} launches)")
        cc = s["counts_completed"]
        nc = s["n_completed"]
        if nc:
            print(f"  completed-tenor cohort (n={nc}): autocall {cc.get('AUTOCALL', 0) / nc:.1%}  |  "
                  f"maturity_cash {cc.get('MATURITY_CASH', 0) / nc:.1%}  |  "
                  f"delivery {cc.get('PHYSICAL_DELIVERY', 0) / nc:.1%}")
