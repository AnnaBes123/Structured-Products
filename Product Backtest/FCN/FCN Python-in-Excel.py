"""
NOT run via `python3` - this is the body of a single Excel `=PY()` cell (Microsoft 365 "Python in
Excel"). Copy everything below the next line into that cell's formula bar (Alt+Enter for new
lines inside the formula bar), then Ctrl+Shift+Alt+M ("Output As > Excel Value") on the cell so the
result spills as a normal range instead of showing as a single Python-object card.

Setup this expects on the sheet (adjust the two xl(...) range references below to match):
  1. A FactSet-pulled daily price history somewhere on the sheet, laid out as a plain table with a
     header row: "Date" in the first column, one column per ticker after it (e.g. "Prices!A1:B100000").
     Use FactSet's Formula Builder (ribbon > FactSet > Formula Builder > search "price history" /
     "time series") to build that pull - the exact function name (e.g. FG_PRICE) and argument order
     depend on your FactSet entitlements, so generate it there rather than trust a hardcoded example.
     The range below is deliberately oversized (100,000 rows - centuries of daily data) rather than
     sized to match exactly: `.dropna()` already strips the unused blank tail, so a range that's too
     BIG is harmless, only one that's too SMALL silently truncates data - pulling more history later
     (a bigger date range, in FactSet, not in this range string) never needs this number bumped
     again. It only needs widening (more columns, e.g. to column C/D/...) if you add more tickers
     for a worst-of basket.
  2. A two-column parameter list on a "Params" sheet, one term per row, label in column A and value
     in column B (A1:B8):
       Strike                        0.90
       Autocall Trigger              1.00
       Tenor Months                  12
       Autocall Frequency Months     3
       Autocall Lockout Months       0
       Launch Start                  2001-09-18
       Launch End                    (blank)
       Max Gap Days                  10
     The labels in column A below must match these exactly (spelling/spacing). Autocall Frequency
     Months = 0 disables autocall (Excel has no blank-means-None the way Python does - 0 is the
     sentinel). Launch End left blank -> resolves to the last date in the price table.

`xl()` is a Python-in-Excel builtin (not usable/importable outside Excel) - it reads a sheet range
and, with headers=True, returns it as a pandas DataFrame. pandas is preloaded as `pd` in this
environment; no other import is available (no internet access, no yfinance - see DEVLOG.md
"Excel-hosted, not Excel-output" for why the price fetch has to happen via FactSet instead). Because
every term below is read via xl(...), the cell recalculates automatically whenever a param or the
price table changes - no button, no reopening the formula bar.

Output: a two-column Metric/Value table (one row per number, not one wide row) with every percentage
pre-formatted as a "71.1%"-style string - so it reads correctly with no manual cell formatting step,
at the cost of those cells no longer being usable in further Excel arithmetic (they're text, by
design - see DEVLOG.md).

Ported from, and verified to match exactly (see DEVLOG.md), FCN.py's classify_launch - this is a
compact reimplementation, not a shared code path, same caveat as FCN XLSX.py.
"""

# ---8<--- everything from here down goes into the =PY() cell ---8<---

params_raw = xl("Params!A1:B8")                          # <-- adjust range to your Params table
p = dict(zip(params_raw.iloc[:, 0].str.strip(), params_raw.iloc[:, 1]))   # label (col A, whitespace-trimmed) -> value (col B)
STRIKE, AUTOCALL_TRIGGER = float(p["Strike"]), float(p["Autocall Trigger"])
TENOR_MONTHS = int(p["Tenor Months"])
AUTOCALL_FREQUENCY_MONTHS = int(p["Autocall Frequency Months"])   # 0 = disabled
AUTOCALL_LOCKOUT_MONTHS, MAX_GAP_DAYS = int(p["Autocall Lockout Months"]), int(p["Max Gap Days"])

prices_df = xl("Prices!A1:B100000", headers=True)        # <-- oversized on purpose, see docstring
prices_df = prices_df.dropna().set_index(prices_df.columns[0]).sort_index()
all_dates, prices = prices_df.index, prices_df.to_numpy()
ticker_label = xl("Prices!I2")   # <-- wherever you actually typed the ticker for FactSet's formula

launch_start = pd.Timestamp(p["Launch Start"])
data_as_of = all_dates.max()
launch_end = data_as_of if pd.isna(p["Launch End"]) else pd.Timestamp(p["Launch End"])

obs_offsets = ([] if not AUTOCALL_FREQUENCY_MONTHS else
               list(range(AUTOCALL_LOCKOUT_MONTHS + AUTOCALL_FREQUENCY_MONTHS, TENOR_MONTHS, AUTOCALL_FREQUENCY_MONTHS)))


def map_obs(target):
    pos = int(all_dates.searchsorted(target))
    return None if pos >= len(all_dates) or (all_dates[pos] - target).days > MAX_GAP_DAYS else pos


def classify(i):
    s0 = prices[i]
    j = map_obs(all_dates[i] + pd.DateOffset(months=TENOR_MONTHS))
    for m in obs_offsets:
        obs_cal = all_dates[i] + pd.DateOffset(months=m)
        if obs_cal > data_as_of:
            break
        pos = map_obs(obs_cal)
        if pos is None or pos <= i or (j is not None and pos >= j):
            continue
        if (prices[pos] / s0).min() >= AUTOCALL_TRIGGER:
            return "AUTOCALL", j is not None
    if j is None:
        return "OUTSTANDING", False
    worst = (prices[j] / s0).min()
    return ("MATURITY_CASH" if worst >= STRIKE else "PHYSICAL_DELIVERY"), True


rows = [classify(i) for i, d in enumerate(all_dates) if launch_start <= d <= launch_end]
outcomes, completed = [o for o, c in rows], [o for o, c in rows if c]


def fmt_pct(bucket, of):
    return f"{of.count(bucket) / len(of):.3%}" if of else "n/a"


result = pd.DataFrame([
    ("Product", f"{ticker_label} FCN"),
    ("Launches", len(outcomes)),
    ("Autocall %", fmt_pct("AUTOCALL", outcomes)),
    ("Maturity Cash %", fmt_pct("MATURITY_CASH", outcomes)),
    ("Physical Delivery %", fmt_pct("PHYSICAL_DELIVERY", outcomes)),
    ("Outstanding %", fmt_pct("OUTSTANDING", outcomes)),
    ("Completed Launches", len(completed)),
    ("Completed Autocall %", fmt_pct("AUTOCALL", completed)),
    ("Completed Maturity Cash %", fmt_pct("MATURITY_CASH", completed)),
    ("Completed Physical Delivery %", fmt_pct("PHYSICAL_DELIVERY", completed)),
], columns=["Metric", "Value"])
result
