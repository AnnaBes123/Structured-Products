"""
NOT run via `python3` - body of a single Excel `=PY()` cell (Microsoft 365 "Python in Excel").
Reverse Convertible: same worst-of terminal test as FCN's PHYSICAL_DELIVERY path, but no autocall -
the underlying is compared to STRIKE once, at maturity, nothing else. See Product Backtest/FCN's
README.md/DEVLOG.md for the shared conventions (gap tolerance, oversized price range, etc.).

Everything lives on one sheet, "RC":
  A:B  price pull - header row 1 ("Date", ticker), data below. A1:B100000 is deliberately oversized
       (dropna() strips the unused tail - never needs resizing for a longer/shorter date range).
  D:E  params, one per row, label col D / value col E:
         Ticker         PFE
         Strike         0.70
         Tenor Months   12
         Launch Start   2001-09-18
         Launch End     (blank -> last price date)
         Max Gap Days   10
  G1   this =PY() formula. Ctrl+Shift+Alt+M ("Output As > Excel Value") once to spill it.
"""

params_raw = xl("RC!D1:E6")
p = dict(zip(params_raw.iloc[:, 0], params_raw.iloc[:, 1]))
TICKER, STRIKE = p["Ticker"], float(p["Strike"])
TENOR_MONTHS, MAX_GAP_DAYS = int(p["Tenor Months"]), int(p["Max Gap Days"])

prices_df = xl("RC!A1:B100000", headers=True).dropna()
prices_df = prices_df.set_index(prices_df.columns[0]).sort_index()
all_dates, prices = prices_df.index, prices_df.to_numpy()
launch_start = pd.Timestamp(p["Launch Start"])
data_as_of = all_dates.max()
launch_end = data_as_of if pd.isna(p["Launch End"]) else pd.Timestamp(p["Launch End"])


def map_obs(target):
    pos = int(all_dates.searchsorted(target))
    return None if pos >= len(all_dates) or (all_dates[pos] - target).days > MAX_GAP_DAYS else pos


def classify(i):
    s0 = prices[i]
    j = map_obs(all_dates[i] + pd.DateOffset(months=TENOR_MONTHS))
    if j is None:
        return "OUTSTANDING", False
    worst = (prices[j] / s0).min()
    return ("MATURITY_CASH" if worst >= STRIKE else "PHYSICAL_DELIVERY"), True


rows = [classify(i) for i, d in enumerate(all_dates) if launch_start <= d <= launch_end]
labels, completed = [o for o, c in rows], [o for o, c in rows if c]


def fmt_pct(bucket, of):
    return f"{of.count(bucket) / len(of):.3%}" if of else "n/a"


result = pd.DataFrame([
    ("Product", f"{TICKER} RC"),
    ("Launches", len(labels)),
    ("Maturity Cash %", fmt_pct("MATURITY_CASH", labels)),
    ("Physical Delivery %", fmt_pct("PHYSICAL_DELIVERY", labels)),
    ("Outstanding %", fmt_pct("OUTSTANDING", labels)),
    ("Completed Launches", len(completed)),
    ("Completed Maturity Cash %", fmt_pct("MATURITY_CASH", completed)),
    ("Completed Physical Delivery %", fmt_pct("PHYSICAL_DELIVERY", completed)),
], columns=["Metric", "Value"])
result
