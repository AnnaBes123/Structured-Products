"""
Standalone FCN percentage backtest - prints the outcome percentages, nothing else. Deliberately
self-contained - no import from FCN.py or anything else in this repo - so this one file can be
copied to any machine and run with nothing but `pip install pandas yfinance`. It reimplements a
compact version of FCN.py's classification rules rather than sharing that code path; see this
folder's README.md / DEVLOG.md for the full, audited version and its test suite - use that one if
correctness under edge cases (data gaps, basket calendar mismatches, etc.) matters more than
portability.

This is meant to eventually run FROM inside Excel, not write an .xlsx file - see DEVLOG.md
"Excel-hosted, not Excel-output" for why that's a different problem (yfinance needs internet
access, which Excel's own built-in Python sandbox doesn't allow) and what's still undecided about
how to get there.
"""

import pandas as pd
import yfinance as yf

# --- Edit these ---
TICKERS = ["^SPX"]             # 1 ticker = single-underlying; 2+ = worst-of basket
STRIKE = 0.99                   # fraction of launch level, checked ONLY at maturity
AUTOCALL_TRIGGER = 1.00         # fraction of launch level, checked at each observation
TENOR_MONTHS = 12
AUTOCALL_FREQUENCY_MONTHS = None   # None disables autocall entirely
AUTOCALL_LOCKOUT_MONTHS = 0

LAUNCH_START = "2000-01-01"     # first candidate launch date (inclusive)
LAUNCH_END = None                # None -> DATA_AS_OF
DATA_AS_OF = None                # None -> most recent complete calendar day before now
MAX_OBSERVATION_GAP_DAYS = 10    # calendar-day tolerance before a rolled observation is unresolved


def resolve_dates():
    data_as_of = pd.Timestamp(DATA_AS_OF) if DATA_AS_OF else pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    launch_start = pd.Timestamp(LAUNCH_START)
    launch_end = pd.Timestamp(LAUNCH_END) if LAUNCH_END else data_as_of
    return launch_start, launch_end, data_as_of


def fetch_prices(tickers, start, end):
    """Daily closes, inner-joined across tickers (a worst-of basket only keeps dates every ticker
    has a price for)."""
    closes = {}
    for t in tickers:
        data = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
        closes[t] = data["Close"].iloc[:, 0] if hasattr(data["Close"], "columns") else data["Close"]
    df = pd.concat(closes, axis=1)
    df.columns = tickers
    return df.dropna(how="any")


def observation_offsets():
    if AUTOCALL_FREQUENCY_MONTHS is None:
        return []
    start = AUTOCALL_LOCKOUT_MONTHS + AUTOCALL_FREQUENCY_MONTHS
    return list(range(start, TENOR_MONTHS, AUTOCALL_FREQUENCY_MONTHS))


def map_observation(target_date, all_dates):
    """Next available trading day on/after target_date, or None if too far away/unavailable."""
    pos = int(all_dates.searchsorted(target_date))
    if pos >= len(all_dates) or (all_dates[pos] - target_date).days > MAX_OBSERVATION_GAP_DAYS:
        return None
    return pos


def classify_launch(prices, all_dates, i, data_as_of, obs_offsets):
    """One launch -> (outcome, is_completed_tenor). Same rules as FCN.py's classify_launch: no
    month-zero autocall, an observation must resolve strictly before maturity to count as an early
    check, both boundaries (trigger/strike) are inclusive."""
    s0 = prices[i]
    launch_date = all_dates[i]
    maturity_calendar = launch_date + pd.DateOffset(months=TENOR_MONTHS)

    j = map_observation(maturity_calendar, all_dates) if maturity_calendar <= data_as_of else None

    for m in obs_offsets:
        obs_calendar = launch_date + pd.DateOffset(months=m)
        if obs_calendar > data_as_of:
            break
        pos = map_observation(obs_calendar, all_dates)
        if pos is None or pos <= i or (j is not None and pos >= j):
            continue
        if (prices[pos] / s0).min() >= AUTOCALL_TRIGGER:
            return "AUTOCALL", j is not None

    if j is not None:
        worst_at_maturity = (prices[j] / s0).min()
        return ("MATURITY_CASH" if worst_at_maturity >= STRIKE else "PHYSICAL_DELIVERY"), True
    return "OUTSTANDING", False


if __name__ == "__main__":
    launch_start, launch_end, data_as_of = resolve_dates()
    price_df = fetch_prices(TICKERS, launch_start, data_as_of + pd.Timedelta(days=1))
    price_df = price_df[price_df.index <= data_as_of]

    all_dates = price_df.index
    prices = price_df.to_numpy()
    obs_offsets = observation_offsets()
    launch_positions = [i for i, d in enumerate(all_dates) if launch_start <= d <= launch_end]

    outcomes = []
    completed_outcomes = []
    for i in launch_positions:
        outcome, is_completed = classify_launch(prices, all_dates, i, data_as_of, obs_offsets)
        outcomes.append(outcome)
        if is_completed:
            completed_outcomes.append(outcome)

    n = len(outcomes)
    n_completed = len(completed_outcomes)

    def pct(bucket, of):
        return of.count(bucket) / len(of) if of else None

    print(f"\n{'/'.join(TICKERS)} FCN  ({n} launches, data as-of {data_as_of.date()})")
    print(f"  All launches:      AUTOCALL {pct('AUTOCALL', outcomes):.1%}   "
          f"MATURITY_CASH {pct('MATURITY_CASH', outcomes):.1%}   "
          f"PHYSICAL_DELIVERY {pct('PHYSICAL_DELIVERY', outcomes):.1%}   "
          f"OUTSTANDING {pct('OUTSTANDING', outcomes):.1%}")
    if n_completed:
        print(f"  Completed cohort:  AUTOCALL {pct('AUTOCALL', completed_outcomes):.1%}   "
              f"MATURITY_CASH {pct('MATURITY_CASH', completed_outcomes):.1%}   "
              f"PHYSICAL_DELIVERY {pct('PHYSICAL_DELIVERY', completed_outcomes):.1%}")
    else:
        print("  Completed cohort:  n/a (every launch still OUTSTANDING)")
