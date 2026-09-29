"""Shared rolling-launch engine for every Product Backtest script. Each product supplies only its own
`classify(w, mat, obs_at)` rule; fetching, date mapping, reporting and charts live here once.
See FCN/README.md for the methodology and each folder's DEVLOG.md for decision history."""
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

plt.rcParams["font.family"] = "Arial"
MAX_GAP_DAYS = 10  # a roll longer than this is a data hole, not a holiday


def fetch(tickers, start, as_of=None):
    """Worst-of-ready close matrix. as_of=None -> yesterday (last complete session)."""
    as_of = pd.Timestamp(as_of) if as_of else pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    # auto_adjust=False: split- but not dividend-adjusted, matching the note's contractual levels
    df = yf.download(tickers, start=start, end=as_of + pd.Timedelta(days=1), progress=False, auto_adjust=False)
    df = df["Close"][tickers]
    n = len(df)
    df = df.dropna()  # inner join: a basket date missing on any ticker is dropped from all
    if n > len(df):
        print(f"  basket alignment dropped {n - len(df)} of {n} dates")
    if df.empty or not df.index.is_unique or not (df.to_numpy() > 0).all():
        raise ValueError("price data empty, duplicated, or non-positive")
    return df[df.index <= as_of], as_of


def map_obs(dates, target):
    """Following-business-day roll; None if past the data or the roll exceeds MAX_GAP_DAYS."""
    pos = int(dates.searchsorted(target))
    return pos if pos < len(dates) and (dates[pos] - target).days <= MAX_GAP_DAYS else None


def run(prices, tenor_months, classify, launch_start, launch_end=None):
    """One launch per trading day. classify(w, mat, obs_at) sees the worst-of path w (w[0] == 1),
    the maturity index `mat` into w (None = not yet observable -> not completed), and obs_at(m),
    the index of the m-month observation (None if unobservable)."""
    dates, px = prices.index, prices.to_numpy()
    launch_end = pd.Timestamp(launch_end) if launch_end else dates[-1]
    rows = []
    for i in np.where((dates >= pd.Timestamp(launch_start)) & (dates <= launch_end))[0]:
        j = map_obs(dates, dates[i] + pd.DateOffset(months=tenor_months))
        w = (px[i:(len(dates) if j is None else j + 1)] / px[i]).min(axis=1)

        def obs_at(m, i=i):
            k = map_obs(dates, dates[i] + pd.DateOffset(months=m))
            return None if k is None else k - i

        mat = None if j is None else j - i
        rows.append((dates[i], classify(w, mat, obs_at), mat is not None, w[-1] - 1))
    return pd.DataFrame(rows, columns=["Launch", "Outcome", "Completed", "UnderlyingReturn"])


def report(res, prices, buckets, title, script_file):
    """Prints all-launch + completed-cohort + by-year mix; saves audit CSV and two charts beside
    the calling script (gitignored, run-specific)."""
    print("\nCAVEAT: daily launches overlap almost entirely - these describe one historical path, "
          "not independent trials.")
    print(f"\n{title}\n  {len(res)} launches {res.Launch.iloc[0].date()} to {res.Launch.iloc[-1].date()}, "
          f"data {prices.index[0].date()} to {prices.index[-1].date()}")
    done = res[res.Completed]
    for label, df in (("All launches", res), (f"Completed cohort (n={len(done)})", done)):
        c = Counter(df.Outcome)
        print(f"  {label:<26}" + "   ".join(f"{b} {c[b] / len(df):.1%}" for b in buckets if len(df)))
    by_year = pd.crosstab(res.Launch.dt.year, res.Outcome, normalize="index").reindex(columns=buckets, fill_value=0)
    by_year["AvgUnderlyingRet"] = res.groupby(res.Launch.dt.year).UnderlyingReturn.mean()
    print("\nBy launch year (outcome mix; last col = underlying's own avg return to maturity/as-of):")
    print(by_year.to_string(float_format=lambda x: f"{x:.1%}"))

    stem = os.path.join(os.path.dirname(os.path.abspath(script_file)), title.split("  ")[0])
    res.to_csv(f"{stem} - audit.csv", index=False)
    ax = by_year[list(buckets)].mul(100).plot.bar(stacked=True, color=list(buckets.values()),
                                                   width=0.7, figsize=(max(8, len(by_year) * 0.4), 5.5))
    ax.set(ylabel="Share of that year's daily launches (%)", xlabel="", ylim=(0, 100))
    _finish(ax, title + "\nOutcome mix by launch year", f"{stem}.png")

    growth = (prices / prices.iloc[0]).min(axis=1)
    _, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(growth.index, growth, color="black", lw=0.8, alpha=0.6)
    for b, color in buckets.items():
        d = res.Launch[res.Outcome == b]
        ax.scatter(d, growth.loc[d], s=8, color=color, alpha=0.5, lw=0, label=f"{b} launch")
    ax.set_ylabel("Worst-of growth vs. start of history (x)")
    _finish(ax, title + "\nUnderlying growth with launch outcomes", f"{stem} - growth.png")
    print(f"\nSaved audit CSV + charts to {stem}*")


def _finish(ax, title, path):
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=8)
    ax.grid(True, axis="y", color="lightgrey", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
