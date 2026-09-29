"""
Body of ONE Excel =PY() cell: price path chart, with a red line from launch to maturity for every
launch that ended badly for the investor. Same "General" sheet and params (A:E prices, G:H params)
as `General Python-in-Excel.py` - edit the Product/terms there and this chart redraws.
Bad outcome: FCN/RC/BRC = PHYSICAL_DELIVERY; BSF (principal-protected, no delivery) = PARTICIPATION_OTM.
Paste everything below the ---8<--- line into the cell; the chart shows as an image.
"""

# ---8<--- everything from here down goes into the =PY() cell ---8<---
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection

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
       "BSF": "PARTICIPATION_OTM"}[PRODUCT]

prices_df = xl("General!A1:E100000", headers=True).dropna(axis=1, how="all").dropna()
prices_df = prices_df.set_index(prices_df.columns[0]).sort_index()
dates, px = prices_df.index, prices_df.to_numpy(dtype=float)
launch_start = pd.Timestamp(p["Launch Start"])
launch_end = dates[-1] if pd.isna(p["Launch End"]) else pd.Timestamp(p["Launch End"])


def map_obs(target):
    pos = int(dates.searchsorted(target))
    return None if pos >= len(dates) or (dates[pos] - target).days > MAX_GAP else pos


def classify(i):
    """(outcome, maturity row) for the note launched on trading day i - same rules as the backtest cell."""
    s0 = px[i]
    j = map_obs(dates[i] + pd.DateOffset(months=TENOR))
    for m in OBS_MONTHS:
        k = map_obs(dates[i] + pd.DateOffset(months=m))
        if k is not None and i < k and (j is None or k < j) and (px[k] / s0).min() >= TRIGGER:
            return "AUTOCALL", j
    if j is None:
        return "OUTSTANDING", j
    path = (px[i if AMERICAN else j:j + 1] / s0).min(axis=1)
    end = path[-1]
    if PRODUCT == "BSF":
        return ("KNOCKED_OUT" if (path >= BARRIER).any() else
                "PARTICIPATION_ITM" if end >= STRIKE else "PARTICIPATION_OTM"), j
    delivered = end < STRIKE and (PRODUCT != "BRC" or (path <= BARRIER).any())
    return ("PHYSICAL_DELIVERY" if delivered else "MATURITY_CASH"), j


# single ticker: actual price; basket: worst-of performance vs. first date
y = px[:, 0] if px.shape[1] == 1 else (px / px[0]).min(axis=1)
x = mdates.date2num(dates)
bad, n = [], 0
for i, d in enumerate(dates):
    if launch_start <= d <= launch_end:
        n += 1
        outcome, j = classify(i)
        if outcome == BAD:
            bad.append([(x[i], y[i]), (x[j], y[j])])  # launch -> maturity

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(dates, y, color="black", lw=0.8)
ax.add_collection(LineCollection(bad, colors="firebrick", lw=0.6, alpha=0.15))
ax.plot([], [], color="firebrick", label=f"{BAD}: launch -> maturity ({len(bad)} of {n} launches)")  # solid legend swatch
ax.set(title=f"{p['Ticker']} {PRODUCT} - strike {STRIKE:.0%}, tenor {TENOR}m",
       ylabel="Price" if px.shape[1] == 1 else "Worst-of vs. start (x)")
ax.legend(loc="upper left", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig
