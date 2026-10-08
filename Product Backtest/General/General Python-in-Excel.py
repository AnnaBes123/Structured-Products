"""
Body of ONE Excel =PY() cell: rolling backtest of any covered product (FCN / RC / BRC / BSF).
Paste everything below the ---8<--- line into the cell, then Ctrl+Shift+Alt+M (Output As > Excel Value).

Sheet "General":
  A:E  price pull, header row 1: "Date", then one column per ticker (2+ = worst-of; unused columns blank).
  G:H  params, label in G / value in H (rows 1-12; blank = not used by this product):
         Ticker                      AMD
         Product                     FCN        (FCN, RC, BRC or BSF)
         Strike                      0.90
         Barrier                     0.70       (BRC: down-and-in; BSF: up-and-out)
         Barrier Monitoring          European   (or American = every trading day)
         Autocall Trigger            1.00       (FCN only)
         Autocall Frequency Months   3          (FCN only; 0 = no autocall)
         Autocall Lockout Months     0          (FCN only)
         Tenor Months                12
         Launch Start                2001-01-02
         Launch End                  (blank -> last price date)
         Max Gap Days                10
"""

# ---8<--- everything from here down goes into the =PY() cell ---8<---

raw = xl("General!G1:H12")
p = dict(zip(raw.iloc[:, 0].str.strip(), raw.iloc[:, 1]))


def num(label):
    return 0.0 if pd.isna(p[label]) else float(p[label])


PRODUCT = str(p["Product"]).strip().upper()
STRIKE, BARRIER, TRIGGER = num("Strike"), num("Barrier"), num("Autocall Trigger")
AMERICAN = str(p["Barrier Monitoring"]).strip().upper().startswith("A")
TENOR, FREQ, LOCKOUT, MAX_GAP = (int(num(k)) for k in
                                 ("Tenor Months", "Autocall Frequency Months", "Autocall Lockout Months", "Max Gap Days"))
OBS_MONTHS = range(LOCKOUT + FREQ, TENOR, FREQ) if PRODUCT == "FCN" and FREQ else []  # never month 0
BUCKETS = {"FCN": ["AUTOCALL", "MATURITY_CASH", "PHYSICAL_DELIVERY"],
           "RC": ["MATURITY_CASH", "PHYSICAL_DELIVERY"],
           "BRC": ["MATURITY_CASH", "PHYSICAL_DELIVERY"],
           "BSF": ["PARTICIPATION_ITM", "PARTICIPATION_OTM", "KNOCKED_OUT"]}[PRODUCT]

prices_df = xl("General!A1:E100000", headers=True).dropna(axis=1, how="all").dropna()  # oversized on purpose
prices_df = prices_df.set_index(prices_df.columns[0]).sort_index()
dates, px = prices_df.index, prices_df.to_numpy(dtype=float)
launch_start = pd.Timestamp(p["Launch Start"])
launch_end = dates[-1] if pd.isna(p["Launch End"]) else pd.Timestamp(p["Launch End"])


def map_obs(target):
    """Next trading day on/after target; None if past the data or a gap > MAX_GAP (data hole)."""
    pos = int(dates.searchsorted(target))
    return None if pos >= len(dates) or (dates[pos] - target).days > MAX_GAP else pos


def classify(i):
    """(outcome, completed) for the note launched on trading day i."""
    s0 = px[i]
    j = map_obs(dates[i] + pd.DateOffset(months=TENOR))
    for m in OBS_MONTHS:
        k = map_obs(dates[i] + pd.DateOffset(months=m))
        # must fall strictly before maturity's own day - that day belongs to the strike test
        if k is not None and i < k and (j is None or k < j) and (px[k] / s0).min() >= TRIGGER:
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
    delivered = end < STRIKE and (PRODUCT != "BRC" or (path <= BARRIER).any())
    return ("PHYSICAL_DELIVERY" if delivered else "MATURITY_CASH"), True


rows = [classify(i) for i, d in enumerate(dates) if launch_start <= d <= launch_end]
every, completed = [o for o, c in rows], [o for o, c in rows if c]


def pct(bucket, of):
    return f"{of.count(bucket) / len(of):.3%}" if of else "n/a"


mode = f" ({'American' if AMERICAN else 'European'})" if PRODUCT in ("BRC", "BSF") else ""
pd.DataFrame([("Product", f"{p['Ticker']} {PRODUCT}{mode}"), ("Launches", len(every))]
             + [(f"{b} %", pct(b, every)) for b in BUCKETS + ["OUTSTANDING"]]
             + [("Completed Launches", len(completed))]
             + [(f"Completed {b} %", pct(b, completed)) for b in BUCKETS],
             columns=["Metric", "Value"])
