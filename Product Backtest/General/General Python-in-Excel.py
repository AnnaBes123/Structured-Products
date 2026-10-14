"""
Body of ONE Excel =PY() cell: rolling backtest of any covered product (FCN / RC / BRC / BSF).
Paste everything below the ---8<--- line into the cell, then Ctrl+Shift+Alt+M (Output As > Excel Value).

Sheet "General":
  A:E  price pull, header row 1: "Date", then one column per ticker (2+ = worst-of; unused columns blank).
  G:H  params, value in H, rows 1-12 in exactly this order (blank = not used by this product). Only the
       row order matters - the G labels are yours to word per product (formulas: see DEVLOG.md):
         Ticker                      AMD
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
"""

# ---8<--- everything from here down goes into the =PY() cell ---8<---

raw = xl("General!G1:H12")
# read by row, not by label, so the G labels are free to change wording with the product
p = dict(zip(["Ticker", "Product", "Strike", "Barrier", "Barrier Monitoring", "Autocall Trigger",
              "Autocall Frequency", "Autocall Delay", "Tenor Months", "Launch Start",
              "Launch End", "Max Gap Days"], raw.iloc[:, 1]))


def text(label):
    return "" if pd.isna(p[label]) else str(p[label]).strip().upper()


def level(label):
    """Level as a fraction of launch. Excel's 130% already arrives as 1.3; a typed 130 or '130%' also works."""
    s = text(label)
    x = float(s.rstrip("%") or 0)
    return x / 100 if s.endswith("%") or x > 5 else x  # no real strike/barrier is above 500%


def months(label):
    """Whole months; blank, 0 or 'No delay' -> 0; a stray 'm' (e.g. '6m') is tolerated."""
    s = text(label).replace("MONTHS", "").replace("M", "").strip()
    return 0 if not s or s.startswith("NO") else int(float(s))


def frequency(label):
    """None = no autocall, 0 = every trading day, otherwise months between observations."""
    s = text(label).replace("-", "").replace(" ", "")
    named = {"DAILY": 0, "MONTHLY": 1, "BIMONTHLY": 2, "QUARTERLY": 3, "SEMIANNUALLY": 6, "SEMIANNUAL": 6}
    if s in named:
        return named[s]
    if s in ("", "NONE", "NO", "NOAUTOCALL") or float(s) == 0:  # a bare number still means months
        return None
    return int(float(s))


PRODUCT = text("Product")
STRIKE, BARRIER, TRIGGER = level("Strike"), level("Barrier"), level("Autocall Trigger")
AMERICAN = text("Barrier Monitoring").startswith("A")
TENOR, DELAY, MAX_GAP = months("Tenor Months"), months("Autocall Delay"), months("Max Gap Days")
FREQ = frequency("Autocall Frequency") if PRODUCT == "FCN" else None
# first observation = delay + one period, so never the launch day itself
OBS_MONTHS = range(DELAY + FREQ, TENOR, FREQ) if FREQ else []
BUCKETS = {"FCN": ["AUTOCALL", "MATURITY_CASH", "PHYSICAL_DELIVERY"],
           "RC": ["MATURITY_CASH", "PHYSICAL_DELIVERY"],
           "BRC": ["MATURITY_CASH", "BREACHED_RECOVERED", "PHYSICAL_DELIVERY"],
           "BSF": ["PARTICIPATION_ITM", "PARTICIPATION_OTM", "KNOCKED_OUT"]}[PRODUCT]

MODE = "American" if AMERICAN else "European"
FREQ_NAME = {0: "daily", 1: "monthly", 2: "bi-monthly", 3: "quarterly", 6: "semi-annually"}.get(FREQ, f"every {FREQ}m")
TERMS = f"strike {STRIKE:.0%}" + {
    "FCN": ", no autocall" if FREQ is None else
           f", autocall {TRIGGER:.0%} {FREQ_NAME}" + (f" after {DELAY}m delay" if DELAY else ""),
    "RC": "",
    "BRC": f", knock-in barrier {BARRIER:.0%} {MODE}",
    "BSF": f", knock-out barrier {BARRIER:.0%} {MODE}"}[PRODUCT] + f", tenor {TENOR}m"

prices_df = xl("General!A1:E100000", headers=True).dropna(axis=1, how="all").dropna()  # oversized on purpose
prices_df = prices_df.set_index(prices_df.columns[0]).sort_index()
dates, px = prices_df.index, prices_df.to_numpy(dtype=float)
launch_start = pd.Timestamp(p["Launch Start"])
launch_end = dates[-1] if pd.isna(p["Launch End"]) else pd.Timestamp(p["Launch End"])


def map_obs(target):
    """Next trading day on/after target; None if past the data or a gap > MAX_GAP (data hole)."""
    pos = int(dates.searchsorted(target))
    return None if pos >= len(dates) or (dates[pos] - target).days > MAX_GAP else pos


def autocalled(i, j):
    """True if an observation strictly between launch day and maturity day closes at/above the trigger."""
    s0, end = px[i], len(dates) if j is None else j  # maturity's own day belongs to the strike test
    if FREQ == 0:  # daily: every trading day after the delay
        first = int(dates.searchsorted(dates[i] + pd.DateOffset(months=DELAY), side="right"))
        return bool(((px[first:end] / s0).min(axis=1) >= TRIGGER).any())
    for m in OBS_MONTHS:
        k = map_obs(dates[i] + pd.DateOffset(months=m))
        if k is not None and i < k < end and (px[k] / s0).min() >= TRIGGER:
            return True
    return False


def classify(i):
    """(outcome, completed) for the note launched on trading day i."""
    s0 = px[i]
    j = map_obs(dates[i] + pd.DateOffset(months=TENOR))
    if FREQ is not None and autocalled(i, j):
        return "AUTOCALL", j is not None
    live = (px[i:None if j is None else j + 1] / s0).min(axis=1)  # worst-of, launch -> maturity (or as-of)
    if PRODUCT == "BSF" and AMERICAN and (live >= BARRIER).any():
        return "KNOCKED_OUT", j is not None  # KO is final the day it's hit, even if maturity isn't in the data yet
    if j is None:
        return "OUTSTANDING", False
    path = live if AMERICAN else live[-1:]  # European = maturity day only
    end = path[-1]
    if PRODUCT == "BSF":
        return ("KNOCKED_OUT" if (path >= BARRIER).any() else
                "PARTICIPATION_ITM" if end >= STRIKE else "PARTICIPATION_OTM"), True
    breached = PRODUCT != "BRC" or (path <= BARRIER).any()  # FCN/RC have no barrier: strike test alone
    if breached and end < STRIKE:
        return "PHYSICAL_DELIVERY", True
    # BRC only: barrier was hit during the life but the note recovered to >= strike, so cash after all
    return ("BREACHED_RECOVERED" if PRODUCT == "BRC" and breached else "MATURITY_CASH"), True


rows = [classify(i) for i, d in enumerate(dates) if launch_start <= d <= launch_end]
every, completed = [o for o, c in rows], [o for o, c in rows if c]


def pct(bucket, of):
    return f"{of.count(bucket) / len(of):.3%}" if of else "n/a"


pd.DataFrame([("Product", f"{p['Ticker']} {PRODUCT}"), ("Terms", TERMS), ("Launches", len(every))]
             + [(f"{b} %", pct(b, every)) for b in BUCKETS + ["OUTSTANDING"]]
             + [("Completed Launches", len(completed))]
             + [(f"Completed {b} %", pct(b, completed)) for b in BUCKETS],
             columns=["Metric", "Value"])
