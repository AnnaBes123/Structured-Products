"""
Body of ONE Excel =PY() cell: price path chart, with a vertical red line for every launch that hit the
product's marked outcome - at its launch date, except BSF, drawn at the date the knock-out occurred. Same "General" sheet and params (A:E prices, G:H params)
as `General Python-in-Excel.py` - edit the Product/terms there and this chart redraws.
Marked outcome: FCN/RC/BRC = PHYSICAL_DELIVERY; BSF (principal-protected, no delivery) = KNOCKED_OUT.
Paste everything below the ---8<--- line into the cell; the chart shows as an image.
"""

# ---8<--- everything from here down goes into the =PY() cell ---8<---
import matplotlib.pyplot as plt

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
BAD = {"FCN": "PHYSICAL_DELIVERY", "RC": "PHYSICAL_DELIVERY", "BRC": "PHYSICAL_DELIVERY",
       "BSF": "KNOCKED_OUT"}[PRODUCT]

prices_df = xl("General!A1:E100000", headers=True).dropna(axis=1, how="all").dropna()
prices_df = prices_df.set_index(prices_df.columns[0]).sort_index()
dates, px = prices_df.index, prices_df.to_numpy(dtype=float)
launch_start = pd.Timestamp(p["Launch Start"])
launch_end = dates[-1] if pd.isna(p["Launch End"]) else pd.Timestamp(p["Launch End"])


def map_obs(target):
    pos = int(dates.searchsorted(target))
    return None if pos >= len(dates) or (dates[pos] - target).days > MAX_GAP else pos


def classify(i):
    """(outcome, row to mark on the chart) for the note launched on trading day i - same rules as the
    backtest cell. Row = launch day, except a BSF knock-out = the day the barrier was hit."""
    s0 = px[i]
    j = map_obs(dates[i] + pd.DateOffset(months=TENOR))
    for m in OBS_MONTHS:
        k = map_obs(dates[i] + pd.DateOffset(months=m))
        if k is not None and i < k and (j is None or k < j) and (px[k] / s0).min() >= TRIGGER:
            return "AUTOCALL", i
    live = (px[i:None if j is None else j + 1] / s0).min(axis=1)  # worst-of, launch -> maturity (or as-of)
    if PRODUCT == "BSF" and AMERICAN and (live >= BARRIER).any():
        return "KNOCKED_OUT", i + int((live >= BARRIER).argmax())  # first hit; final even before maturity
    if j is None:
        return "OUTSTANDING", i
    path = live if AMERICAN else live[-1:]  # European = maturity day only
    end = path[-1]
    if PRODUCT == "BSF":
        if end >= BARRIER:  # American hits already returned above, so this is the European maturity test
            return "KNOCKED_OUT", j
        return ("PARTICIPATION_ITM" if end >= STRIKE else "PARTICIPATION_OTM"), i
    delivered = end < STRIKE and (PRODUCT != "BRC" or (path <= BARRIER).any())
    return ("PHYSICAL_DELIVERY" if delivered else "MATURITY_CASH"), i


# single ticker: actual price; basket: worst-of performance vs. first date
y = px[:, 0] if px.shape[1] == 1 else (px / px[0]).min(axis=1)
bad, n = [], 0
for i, d in enumerate(dates):
    if launch_start <= d <= launch_end:
        n += 1
        outcome, k = classify(i)
        if outcome == BAD:
            bad.append(dates[k])
days = sorted(set(bad))  # many launches knock out on the same day - draw each day once
when = "knock-out date" if PRODUCT == "BSF" else "launch date"

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(dates, y, color="black", lw=0.8)
ax.vlines(days, 0, 1, transform=ax.get_xaxis_transform(), colors="firebrick", lw=0.5,
          alpha=0.5 if PRODUCT == "BSF" else 0.15)  # full height; BSF days are de-duplicated, so less overlap
ax.plot([], [], color="firebrick", label=f"{BAD}: {when} ({len(bad)} of {n} launches"
                                             + (f", {len(days)} days)" if PRODUCT == "BSF" else ")"))  # solid legend swatch
barrier = (f", barrier {BARRIER:.0%} {'American' if AMERICAN else 'European'}"
           if PRODUCT in ("BRC", "BSF") else "")
ax.set(title=f"{p['Ticker']} {PRODUCT} - strike {STRIKE:.0%}{barrier}, tenor {TENOR}m",
       ylabel="Price" if px.shape[1] == 1 else "Worst-of vs. start (x)")
ax.legend(loc="upper left", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig
