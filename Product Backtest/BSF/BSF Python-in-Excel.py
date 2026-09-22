"""
NOT run via `python3` - body of a single Excel `=PY()` cell (Microsoft 365 "Python in Excel").
Bullish Shark Fin: upside participation note capped by an up-and-out knock-out barrier - no
downside/delivery risk at all (principal-protected), unlike FCN/RC/BRC. Worst-of across tickers,
same convention as this repo's other Python-in-Excel snippets:
  - Barrier touched (per the monitoring mode below) -> upside knocked out, investor gets principal
    back (+ any fixed rebate - not modeled, this reports outcome buckets only, not $ payoffs)
    (bucket: KNOCKED_OUT). STRIKE is irrelevant once knocked out - the rebate doesn't depend on it.
  - Barrier never touched, finishes >= STRIKE at maturity -> participation payoff is positive
    (bucket: PARTICIPATION_ITM).
  - Barrier never touched, finishes < STRIKE at maturity -> never knocked out, but the
    participation payoff is zero - same as principal back with no gain (bucket: PARTICIPATION_OTM).
Two barrier-monitoring modes (param below), same idea as BRC: European = barrier checked ONLY at
the single maturity observation; American = checked on every trading day from launch through
maturity (a running-min touch test). STRIKE should be set <= BARRIER (not enforced - barebones).

Everything lives on one sheet, "BSF":
  A:B  price pull - header row 1 ("Date", ticker), data below. A1:B100000 oversized on purpose
       (dropna() strips the unused tail - see Product Backtest/RC's DEVLOG.md).
  D:E  params, one per row, label col D / value col E:
         Ticker              PFE
         Strike              1.00
         Barrier             1.15
         Barrier Monitoring  European   (or American)
         Tenor Months        12
         Launch Start        2001-09-18
         Launch End          (blank -> last price date)
         Max Gap Days        10
  G1   this =PY() formula. Ctrl+Shift+Alt+M ("Output As > Excel Value") once to spill it.
"""

params_raw = xl("BSF!D1:E8")
p = dict(zip(params_raw.iloc[:, 0].str.strip(), params_raw.iloc[:, 1]))
TICKER, STRIKE, BARRIER = p["Ticker"], float(p["Strike"]), float(p["Barrier"])
AMERICAN = str(p["Barrier Monitoring"]).strip().upper().startswith("A")
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
    worst = (prices[j] / s0).min()
    touched = ((prices[i:j + 1] / s0).min(axis=1) >= BARRIER).any() if AMERICAN else worst >= BARRIER
    if touched:
        return "KNOCKED_OUT", True
    return ("PARTICIPATION_ITM" if worst >= STRIKE else "PARTICIPATION_OTM"), True


rows = [classify(i) for i, d in enumerate(all_dates) if launch_start <= d <= launch_end]
labels, completed = [o for o, c in rows], [o for o, c in rows if c]


def fmt_pct(bucket, of):
    return f"{of.count(bucket) / len(of):.3%}" if of else "n/a"


result = pd.DataFrame([
    ("Product", f"{TICKER} Bullish Shark Fin ({'American' if AMERICAN else 'European'})"),
    ("Launches", len(labels)),
    ("Participation (ITM) %", fmt_pct("PARTICIPATION_ITM", labels)),
    ("Participation (OTM) %", fmt_pct("PARTICIPATION_OTM", labels)),
    ("Knocked Out %", fmt_pct("KNOCKED_OUT", labels)),
    ("Outstanding %", fmt_pct("OUTSTANDING", labels)),
    ("Completed Launches", len(completed)),
    ("Completed Participation (ITM) %", fmt_pct("PARTICIPATION_ITM", completed)),
    ("Completed Participation (OTM) %", fmt_pct("PARTICIPATION_OTM", completed)),
    ("Completed Knocked Out %", fmt_pct("KNOCKED_OUT", completed)),
], columns=["Metric", "Value"])
result
