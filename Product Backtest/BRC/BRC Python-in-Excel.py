"""
NOT run via `python3` - body of a single Excel `=PY()` cell (Microsoft 365 "Python in Excel").
Barrier Reverse Convertible: no autocall, plus a down-and-in barrier on top of RC's terminal test -
same terminal payoff convention as `Product MtM/Yield/Barrier Reverse Convertible/README.md`:
  - Barrier never touched -> full principal back, REGARDLESS of where the underlying ends up.
  - Barrier touched at some point, finishes >= STRIKE at maturity -> still full principal back.
  - Barrier touched at some point, finishes < STRIKE at maturity -> physical delivery (loss).
Two barrier-monitoring modes (param below): European = barrier checked ONLY at the single maturity
observation; American = checked on every trading day from launch through maturity (a running-min
touch test - see Product Backtest/RC's DEVLOG.md for the shared Python-in-Excel conventions this
reuses). BARRIER should be set <= STRIKE (not enforced here - keep it barebones).

Everything lives on one sheet, "BRC":
  A:B  price pull - header row 1 ("Date", ticker), data below. A1:B100000 is deliberately oversized
       (dropna() strips the unused tail - never needs resizing for a longer/shorter date range).
  D:E  params, one per row, label col D / value col E:
         Ticker              PFE
         Strike              1.00
         Barrier             0.70
         Barrier Monitoring  European   (or American)
         Tenor Months        12
         Launch Start        2001-09-18
         Launch End          (blank -> last price date)
         Max Gap Days        10
  G1   this =PY() formula. Ctrl+Shift+Alt+M ("Output As > Excel Value") once to spill it.
"""

params_raw = xl("BRC!D1:E8")
p = dict(zip(params_raw.iloc[:, 0].str.strip(), params_raw.iloc[:, 1]))
TICKER, STRIKE, BARRIER = p["Ticker"], float(p["Strike"]), float(p["Barrier"])
AMERICAN = str(p["Barrier Monitoring"]).strip().upper().startswith("A")
TENOR_MONTHS, MAX_GAP_DAYS = int(p["Tenor Months"]), int(p["Max Gap Days"])

prices_df = xl("BRC!A1:B100000", headers=True).dropna()
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
    touched = ((prices[i:j + 1] / s0).min(axis=1) <= BARRIER).any() if AMERICAN else worst <= BARRIER
    return ("PHYSICAL_DELIVERY" if touched and worst < STRIKE else "MATURITY_CASH"), True


rows = [classify(i) for i, d in enumerate(all_dates) if launch_start <= d <= launch_end]
labels, completed = [o for o, c in rows], [o for o, c in rows if c]


def fmt_pct(bucket, of):
    return f"{of.count(bucket) / len(of):.3%}" if of else "n/a"


result = pd.DataFrame([
    ("Product", f"{TICKER} BRC ({'American' if AMERICAN else 'European'})"),
    ("Launches", len(labels)),
    ("Maturity Cash %", fmt_pct("MATURITY_CASH", labels)),
    ("Physical Delivery %", fmt_pct("PHYSICAL_DELIVERY", labels)),
    ("Outstanding %", fmt_pct("OUTSTANDING", labels)),
    ("Completed Launches", len(completed)),
    ("Completed Maturity Cash %", fmt_pct("MATURITY_CASH", completed)),
    ("Completed Physical Delivery %", fmt_pct("PHYSICAL_DELIVERY", completed)),
], columns=["Metric", "Value"])
result
