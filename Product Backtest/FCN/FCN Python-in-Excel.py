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
       Strike                        90%          (130, "130%" and 1.3 all work)
       Autocall Trigger              100%
       Tenor Months                  12
       Autocall Frequency            Quarterly    (Daily, Monthly, Bi-Monthly, Quarterly, Semi-Annually,
                                                   Annually, or a number of months; 0 / blank = no autocall)
       Autocall Delay                No delay     (or whole months; first observation = delay + one period)
       Launch Start                  2001-09-18
       Launch End                    (blank)
       Max Gap Days                  10
     The labels in column A must match these (spelling/spacing). The older labels "Autocall Frequency
     Months" and "Autocall Lockout Months" are still accepted. Same input conventions as the General
     cells: 0 or blank frequency = no autocall; Daily is only ever selected by the word "Daily".
     Launch End left blank -> resolves to the last date in the price table.

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
import re

params_raw = xl("Params!A1:B8")                          # <-- adjust range to your Params table
p = dict(zip(params_raw.iloc[:, 0].str.strip(), params_raw.iloc[:, 1]))   # label (col A, whitespace-trimmed) -> value (col B)


def text(*labels):
    v = next((p[k] for k in labels if k in p), None)  # first label present: new name, then the old one
    return "" if v is None or pd.isna(v) else str(v).strip().upper()


def level(label):
    s = text(label)
    x = float(s.rstrip("%") or 0)
    return x / 100 if s.endswith("%") or x > 5 else x  # 130 / "130%" / 1.3 -> 1.3


def months(*labels):
    s = text(*labels).replace("MONTHS", "").replace("M", "").strip()
    return 0 if not s or s.startswith("NO") else int(float(s))


def frequency(*labels):
    s = re.sub("[^A-Z0-9.]", "", text(*labels))
    named = {"DAILY": "DAILY", "MONTHLY": 1, "BIMONTHLY": 2, "QUARTERLY": 3, "QUARTER": 3, "SEMIANNUALLY": 6,
             "SEMIANNUAL": 6, "ANNUALLY": 12, "ANNUAL": 12}
    if s in named:
        return named[s]
    s = s.replace("MONTHS", "").replace("MONTH", "").rstrip("M")
    return None if s in ("", "NONE", "NO", "NOAUTOCALL") or float(s) == 0 else int(float(s))  # 0 = no autocall


STRIKE, AUTOCALL_TRIGGER = level("Strike"), level("Autocall Trigger")
TENOR_MONTHS, MAX_GAP_DAYS = months("Tenor Months"), months("Max Gap Days")
AUTOCALL_FREQUENCY = frequency("Autocall Frequency", "Autocall Frequency Months")   # None = no autocall
AUTOCALL_DELAY = months("Autocall Delay", "Autocall Delay to First Observation (Months)", "Autocall Lockout Months")

prices_df = xl("Prices!A1:B100000", headers=True)        # <-- oversized on purpose, see docstring
prices_df = prices_df.dropna().set_index(prices_df.columns[0]).sort_index()
all_dates, prices = prices_df.index, prices_df.to_numpy()
ticker_label = xl("Prices!I2")   # <-- wherever you actually typed the ticker for FactSet's formula

launch_start = pd.Timestamp(p["Launch Start"])
data_as_of = all_dates.max()
launch_end = data_as_of if pd.isna(p["Launch End"]) else pd.Timestamp(p["Launch End"])

obs_offsets = (list(range(AUTOCALL_DELAY + AUTOCALL_FREQUENCY, TENOR_MONTHS, AUTOCALL_FREQUENCY))
               if isinstance(AUTOCALL_FREQUENCY, int) else [])


def map_obs(target):
    pos = int(all_dates.searchsorted(target))
    return None if pos >= len(all_dates) or (all_dates[pos] - target).days > MAX_GAP_DAYS else pos


def classify(i):
    s0 = prices[i]
    j = map_obs(all_dates[i] + pd.DateOffset(months=TENOR_MONTHS))
    if AUTOCALL_FREQUENCY == "DAILY":  # every trading day after the delay, strictly before maturity's day
        first = int(all_dates.searchsorted(all_dates[i] + pd.DateOffset(months=AUTOCALL_DELAY), side="right"))
        if ((prices[first:len(all_dates) if j is None else j] / s0).min(axis=1) >= AUTOCALL_TRIGGER).any():
            return "AUTOCALL", j is not None
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
    ("Terms", f"strike {STRIKE:.0%}"
              + (f", autocall {AUTOCALL_TRIGGER:.0%} "
                 + {"DAILY": "daily", 1: "monthly", 2: "bi-monthly", 3: "quarterly",
                    6: "semi-annually", 12: "annually"}.get(AUTOCALL_FREQUENCY, f"every {AUTOCALL_FREQUENCY}m")
                 + (f" after {AUTOCALL_DELAY}m delay" if AUTOCALL_DELAY else "")
                 if AUTOCALL_FREQUENCY is not None else ", no autocall")
              + f", tenor {TENOR_MONTHS}m"),
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
