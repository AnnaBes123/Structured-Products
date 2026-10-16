"""
Body of ONE Excel =PY() cell: rolling backtest of any covered product (FCN / RC / BRC / BSF).
Paste everything below the ---8<--- line into the cell, then Ctrl+Shift+Alt+M (Output As > Excel Value).

Sheet "General":
  A:E  price pull, header row 1: "Date", then one column per ticker - up to 4 (unused columns blank).
       2+ tickers = worst-of: every test (autocall, barrier, strike) looks at the worst performer vs. its
       own launch level, and a loss is settled on that worst performer.
  G:H  params, value in H, rows 1-13 in exactly this order (blank = not used by this product). Only the
       row order matters - the G labels are yours to word per product (formulas: see DEVLOG.md):
         Ticker                      AMD          (display name only; blank -> the price headers, e.g. AAPL/MSFT)
         Product                     FCN          (FCN, RC, BRC or BSF)
         Strike                      90%          (% of launch level; all three levels: enter as a
         Barrier                     70%           percentage - 130%, not 1.3. BRC: down-and-in; BSF: up-and-out)
         Barrier Monitoring          European     (or American = every trading day)
         Autocall Trigger            100%         (FCN only)
         Autocall Frequency          Quarterly    (FCN only: Daily, Monthly, Bi-Monthly = every 2 months,
                                                   Quarterly, Semi-Annually; blank or 0 = no autocall)
         Autocall Delay to First Observation (Months)
                                     No delay     (FCN only: No delay / 0, or whole months added before the
                                                   first observation - quarterly + 3 -> first observation month 6)
         Tenor Months                12
         Launch Start                2001-01-02
         Launch End                  (blank -> last price date)
         Max Gap Days                10
         Settlement                  Physical     (or Cash; blank = Physical. FCN/RC/BRC loss case only)

Code notes (kept out of the pasted part - Excel caps a =PY() cell at 8,192 characters, so the code
below the cut line has no comments; reasons live here and in DEVLOG.md):
  - Params are read by row position in G1:H13, not by label, so the G labels can be reworded freely.
    p.get() lets a sheet that stops at row 12 (no Settlement row) still work.
  - level(): Excel's 130% arrives as 1.3; a typed 130 or "130%" also works - anything above 5 is read
    as percentage points (no real strike/barrier is above 500%).
  - frequency(): named frequencies, or a bare number = months; blank/0 = no autocall. First
    observation = delay + one period, so never the launch day. Daily = every trading day after the
    delay. An observation must fall strictly before the maturity day (that day is the strike test).
  - 2+ price columns = worst-of: every test uses the worst performer vs. its own launch level, and a
    loss is settled on it (its shares, or their value in cash - same value, so Settlement only renames).
  - classify(): a BSF American knock-out is final the day it's hit, even before maturity is in the
    data. European = maturity day only. FCN/RC have no barrier (strike test alone); BRC barrier hit
    then recovered to >= strike = BREACHED_RECOVERED (cash).
  - Price range A1:E100000 is oversized on purpose; empty rows/columns are dropped.
"""

# ---8<--- everything from here down goes into the =PY() cell ---8<---
import re

raw = xl("General!G1:H13")
p = dict(zip(["Ticker", "Product", "Strike", "Barrier", "Barrier Monitoring", "Autocall Trigger",
              "Autocall Frequency", "Autocall Delay", "Tenor Months", "Launch Start",
              "Launch End", "Max Gap Days", "Settlement"], raw.iloc[:, 1]))


def text(label):
    v = p.get(label)
    return "" if v is None or pd.isna(v) else str(v).strip().upper()


def level(label):
    s = text(label)
    x = float(s.rstrip("%") or 0)
    return x / 100 if s.endswith("%") or x > 5 else x


def months(label):
    s = text(label).replace("MONTHS", "").replace("M", "").strip()
    return 0 if not s or s.startswith("NO") else int(float(s))


def frequency(label):
    s = re.sub("[^A-Z0-9.]", "", text(label))
    named = {"DAILY": 0, "MONTHLY": 1, "BIMONTHLY": 2, "QUARTERLY": 3, "QUARTER": 3, "SEMIANNUALLY": 6,
             "SEMIANNUAL": 6, "ANNUALLY": 12, "ANNUAL": 12}
    if s in named:
        return named[s]
    s = s.replace("MONTHS", "").replace("MONTH", "").rstrip("M")
    if s in ("", "NONE", "NO", "NOAUTOCALL") or float(s) == 0:
        return None
    return int(float(s))


PRODUCT = text("Product")
STRIKE, BARRIER, TRIGGER = level("Strike"), level("Barrier"), level("Autocall Trigger")
AMERICAN = text("Barrier Monitoring").startswith("A")
LOSS = "CASH_SETTLEMENT" if text("Settlement").startswith("C") else "PHYSICAL_DELIVERY"
TENOR, DELAY, MAX_GAP = months("Tenor Months"), months("Autocall Delay"), months("Max Gap Days")
FREQ = frequency("Autocall Frequency") if PRODUCT == "FCN" else None
OBS_MONTHS = range(DELAY + FREQ, TENOR, FREQ) if FREQ else []
BUCKETS = {"FCN": ["AUTOCALL", "MATURITY_CASH", LOSS],
           "RC": ["MATURITY_CASH", LOSS],
           "BRC": ["MATURITY_CASH", "BREACHED_RECOVERED", LOSS],
           "BSF": ["UPSIDE_PAID", "NO_UPSIDE_BELOW_STRIKE", "KNOCKED_OUT"]}[PRODUCT]

MODE = "American" if AMERICAN else "European"
FREQ_NAME = {0: "daily", 1: "monthly", 2: "bi-monthly", 3: "quarterly", 6: "semi-annually"}.get(FREQ, f"every {FREQ}m")
TERMS = f"strike {STRIKE:.0%}" + {
    "FCN": ", no autocall" if FREQ is None else
           f", autocall {TRIGGER:.0%} {FREQ_NAME}" + (f" after {DELAY}m delay" if DELAY else ""),
    "RC": "",
    "BRC": f", knock-in barrier {BARRIER:.0%} {MODE}",
    "BSF": f", knock-out barrier {BARRIER:.0%} {MODE}"}[PRODUCT] + f", tenor {TENOR}m" + (
    "" if PRODUCT == "BSF" else ", cash-settled" if LOSS == "CASH_SETTLEMENT" else ", physically settled")

prices_df = xl("General!A1:E100000", headers=True).dropna(axis=1, how="all").dropna()
prices_df = prices_df.set_index(prices_df.columns[0]).sort_index()
dates, px = prices_df.index, prices_df.to_numpy(dtype=float)
NAMES = [str(c).strip() for c in prices_df.columns]
UNDERLYING = (text("Ticker") or "/".join(NAMES)) + (" worst-of" if len(NAMES) > 1 else "")
launch_start = pd.Timestamp(p["Launch Start"])
launch_end = dates[-1] if pd.isna(p["Launch End"]) else pd.Timestamp(p["Launch End"])


def map_obs(target):
    pos = int(dates.searchsorted(target))
    return None if pos >= len(dates) or (dates[pos] - target).days > MAX_GAP else pos


def autocalled(i, j):
    s0, end = px[i], len(dates) if j is None else j
    if FREQ == 0:
        first = int(dates.searchsorted(dates[i] + pd.DateOffset(months=DELAY), side="right"))
        return bool(((px[first:end] / s0).min(axis=1) >= TRIGGER).any())
    for m in OBS_MONTHS:
        k = map_obs(dates[i] + pd.DateOffset(months=m))
        if k is not None and i < k < end and (px[k] / s0).min() >= TRIGGER:
            return True
    return False


def classify(i):
    s0 = px[i]
    j = map_obs(dates[i] + pd.DateOffset(months=TENOR))
    if FREQ is not None and autocalled(i, j):
        return "AUTOCALL", j is not None
    live = (px[i:None if j is None else j + 1] / s0).min(axis=1)
    if PRODUCT == "BSF" and AMERICAN and (live >= BARRIER).any():
        return "KNOCKED_OUT", j is not None
    if j is None:
        return "OUTSTANDING", False
    path = live if AMERICAN else live[-1:]
    end = path[-1]
    if PRODUCT == "BSF":
        return ("KNOCKED_OUT" if (path >= BARRIER).any() else
                "UPSIDE_PAID" if end >= STRIKE else "NO_UPSIDE_BELOW_STRIKE"), True
    breached = PRODUCT != "BRC" or (path <= BARRIER).any()
    if breached and end < STRIKE:
        return LOSS, True
    return ("BREACHED_RECOVERED" if PRODUCT == "BRC" and breached else "MATURITY_CASH"), True


rows = [classify(i) for i, d in enumerate(dates) if launch_start <= d <= launch_end]
every, completed = [o for o, c in rows], [o for o, c in rows if c]


def pct(bucket, of):
    return f"{of.count(bucket) / len(of):.3%}" if of else "n/a"


pd.DataFrame([("Product", f"{UNDERLYING} {PRODUCT}"), ("Terms", TERMS), ("Launches", len(every))]
             + [(f"{b} %", pct(b, every)) for b in BUCKETS + ["OUTSTANDING"]]
             + [("Completed Launches", len(completed))]
             + [(f"Completed {b} %", pct(b, completed)) for b in BUCKETS],
             columns=["Metric", "Value"])
