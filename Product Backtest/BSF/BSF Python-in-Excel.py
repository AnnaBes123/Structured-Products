"""
NOT run via `python3` - body of a single Excel `=PY()` cell (Microsoft 365 "Python in Excel").
Bullish Shark Fin: upside participation note capped by an up-and-out knock-out barrier - no
downside/delivery risk at all (principal-protected), unlike FCN/RC/BRC. Barrier is continuously
(daily) monitored, worst-of across tickers - same convention as this repo's other Python-in-Excel
snippets:
  - Barrier never touched (worst-of ratio stays below BARRIER the whole life) -> full upside
    participation realized at maturity (bucket: PARTICIPATION).
  - Barrier touched at any point -> upside knocked out, investor gets principal back (+ any fixed
    rebate - not modeled, this reports outcome buckets only, not $ payoffs) (bucket: KNOCKED_OUT).

Everything lives on one sheet, "BSF":
  A:B  price pull - header row 1 ("Date", ticker), data below. A1:B100000 oversized on purpose
       (dropna() strips the unused tail - see Product Backtest/RC's DEVLOG.md).
  D:E  params, one per row, label col D / value col E:
         Ticker         PFE
         Barrier        1.15
         Tenor Months   12
         Launch Start   2001-09-18
         Launch End     (blank -> last price date)
         Max Gap Days   10
  G1   this =PY() formula. Ctrl+Shift+Alt+M ("Output As > Excel Value") once to spill it.
"""

params_raw = xl("BSF!D1:E6")
p = dict(zip(params_raw.iloc[:, 0].str.strip(), params_raw.iloc[:, 1]))
TICKER, BARRIER = p["Ticker"], float(p["Barrier"])
TENOR_MONTHS, MAX_GAP_DAYS = int(p["Tenor Months"]), int(p["Max Gap Days"])

prices_df = xl("BSF!A1:B100000", headers=True).dropna()
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
    knocked_out = ((prices[i:j + 1] / s0).min(axis=1) >= BARRIER).any()
    return ("KNOCKED_OUT" if knocked_out else "PARTICIPATION"), True


rows = [classify(i) for i, d in enumerate(all_dates) if launch_start <= d <= launch_end]
labels, completed = [o for o, c in rows], [o for o, c in rows if c]


def fmt_pct(bucket, of):
    return f"{of.count(bucket) / len(of):.3%}" if of else "n/a"


result = pd.DataFrame([
    ("Product", f"{TICKER} Bullish Shark Fin"),
    ("Launches", len(labels)),
    ("Participation %", fmt_pct("PARTICIPATION", labels)),
    ("Knocked Out %", fmt_pct("KNOCKED_OUT", labels)),
    ("Outstanding %", fmt_pct("OUTSTANDING", labels)),
    ("Completed Launches", len(completed)),
    ("Completed Participation %", fmt_pct("PARTICIPATION", completed)),
    ("Completed Knocked Out %", fmt_pct("KNOCKED_OUT", completed)),
], columns=["Metric", "Value"])
result
