"""
NOT run via `python3` - this is the body of a single Excel `=PY()` cell (Microsoft 365 "Python in
Excel"). Copy everything below the next line into that cell's formula bar (Alt+Enter for new
lines inside the formula bar), then Ctrl+Shift+Alt+M ("Output As > Excel Value") on the cell so the
result spills as a normal range instead of showing as a single Python-object card.

Setup this expects on the sheet (adjust the two xl(...) range references below to match):
  1. A FactSet-pulled daily price history somewhere on the sheet, laid out as a plain table with a
     header row: "Date" in the first column, one column per ticker after it (e.g. "Prices!A1:B6300").
     Use FactSet's Formula Builder (ribbon > FactSet > Formula Builder > search "price history" /
     "time series") to build that pull - the exact function name (e.g. FG_PRICE) and argument order
     depend on your FactSet entitlements, so generate it there rather than trust a hardcoded example.
  2. A one-row parameter table, headers in row 1 and values in row 2, one column per term - e.g. on
     a "Params" sheet:
       Strike | Autocall Trigger | Tenor Months | Autocall Frequency Months | Autocall Lockout Months | Launch Start | Launch End | Max Gap Days
        0.90  |       1.00       |      12      |             3             |            0            |  2001-09-18  |  (blank)   |     10
     Column names below must match these headers exactly. Autocall Frequency Months = 0 disables
     autocall (Excel has no blank-means-None the way Python does - 0 is the sentinel). Launch End
     left blank -> resolves to the last date in the price table.

`xl()` is a Python-in-Excel builtin (not usable/importable outside Excel) - it reads a sheet range
and, with headers=True, returns it as a pandas DataFrame. pandas is preloaded as `pd` in this
environment; no other import is available (no internet access, no yfinance - see DEVLOG.md
"Excel-hosted, not Excel-output" for why the price fetch has to happen via FactSet instead). Because
every term below is read via xl(...), the cell recalculates automatically whenever a param or the
price table changes - no button, no reopening the formula bar.

Ported from, and verified to match exactly (see DEVLOG.md), FCN.py's classify_launch - this is a
compact reimplementation, not a shared code path, same caveat as FCN XLSX.py.
"""

# ---8<--- everything from here down goes into the =PY() cell ---8<---

params = xl("Params!A1:H2", headers=True)                # <-- adjust range to your Params table
STRIKE = float(params["Strike"].iloc[0])
AUTOCALL_TRIGGER = float(params["Autocall Trigger"].iloc[0])
TENOR_MONTHS = int(params["Tenor Months"].iloc[0])
AUTOCALL_FREQUENCY_MONTHS = int(params["Autocall Frequency Months"].iloc[0])   # 0 = disabled
AUTOCALL_LOCKOUT_MONTHS = int(params["Autocall Lockout Months"].iloc[0])
MAX_OBSERVATION_GAP_DAYS = int(params["Max Gap Days"].iloc[0])
launch_start_raw = params["Launch Start"].iloc[0]
launch_end_raw = params["Launch End"].iloc[0]

prices_df = xl("Prices!A1:B6300", headers=True)          # <-- adjust range to your table
prices_df = prices_df.dropna().set_index(prices_df.columns[0]).sort_index()
tickers = list(prices_df.columns)
all_dates = prices_df.index
prices = prices_df.to_numpy()

launch_start = pd.Timestamp(launch_start_raw)
data_as_of = all_dates.max()
launch_end = data_as_of if pd.isna(launch_end_raw) else pd.Timestamp(launch_end_raw)


def obs_offsets():
    if not AUTOCALL_FREQUENCY_MONTHS:
        return []
    start = AUTOCALL_LOCKOUT_MONTHS + AUTOCALL_FREQUENCY_MONTHS
    return list(range(start, TENOR_MONTHS, AUTOCALL_FREQUENCY_MONTHS))


def map_obs(target, dates):
    pos = int(dates.searchsorted(target))
    if pos >= len(dates) or (dates[pos] - target).days > MAX_OBSERVATION_GAP_DAYS:
        return None
    return pos


def classify(i):
    s0 = prices[i]
    launch_date = all_dates[i]
    maturity_cal = launch_date + pd.DateOffset(months=TENOR_MONTHS)
    j = map_obs(maturity_cal, all_dates) if maturity_cal <= data_as_of else None
    for m in obs_offsets():
        obs_cal = launch_date + pd.DateOffset(months=m)
        if obs_cal > data_as_of:
            break
        pos = map_obs(obs_cal, all_dates)
        if pos is None or pos <= i or (j is not None and pos >= j):
            continue
        if (prices[pos] / s0).min() >= AUTOCALL_TRIGGER:
            return "AUTOCALL", j is not None
    if j is not None:
        worst = (prices[j] / s0).min()
        return ("MATURITY_CASH" if worst >= STRIKE else "PHYSICAL_DELIVERY"), True
    return "OUTSTANDING", False


launch_positions = [i for i, d in enumerate(all_dates) if launch_start <= d <= launch_end]
outcomes, completed = [], []
for i in launch_positions:
    outcome, is_completed = classify(i)
    outcomes.append(outcome)
    if is_completed:
        completed.append(outcome)


def pct(bucket, of):
    return of.count(bucket) / len(of) if of else None


result = pd.DataFrame([{
    "Product": "/".join(tickers) + " FCN",
    "Launches": len(outcomes),
    "Autocall %": pct("AUTOCALL", outcomes),
    "Maturity Cash %": pct("MATURITY_CASH", outcomes),
    "Physical Delivery %": pct("PHYSICAL_DELIVERY", outcomes),
    "Outstanding %": pct("OUTSTANDING", outcomes),
    "Completed Launches": len(completed),
    "Completed Autocall %": pct("AUTOCALL", completed),
    "Completed Maturity Cash %": pct("MATURITY_CASH", completed),
    "Completed Physical Delivery %": pct("PHYSICAL_DELIVERY", completed),
}])
result
