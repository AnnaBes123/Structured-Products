"""
Body of ONE Excel =PY() cell: price path chart, with a vertical red line for every launch that hit the
product's marked outcome - at its launch date, except BSF, drawn at the date the knock-out occurred. Same "General" sheet and params (A:E prices, G:H params)
as `General Python-in-Excel.py` - edit the Product/terms there and this chart redraws.
Marked outcome: FCN/RC/BRC = PHYSICAL_DELIVERY or CASH_SETTLEMENT (per Settlement); BSF (principal-protected, no delivery) = KNOCKED_OUT.
Paste everything below the ---8<--- line into the cell; the chart shows as an image.
"""

# ---8<--- everything from here down goes into the =PY() cell ---8<---
import matplotlib.pyplot as plt

raw = xl("General!G1:H13")
# read by row, not by label, so the G labels are free to change wording with the product
p = dict(zip(["Ticker", "Product", "Strike", "Barrier", "Barrier Monitoring", "Autocall Trigger",
              "Autocall Frequency", "Autocall Delay", "Tenor Months", "Launch Start",
              "Launch End", "Max Gap Days", "Settlement"], raw.iloc[:, 1]))


def text(label):
    v = p.get(label)  # .get: a sheet without the newer rows (e.g. Settlement) still works
    return "" if v is None or pd.isna(v) else str(v).strip().upper()


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
# below-strike loss (FCN/RC/BRC) is settled on the worst performer: its shares, or their value in cash
LOSS = "CASH_SETTLEMENT" if text("Settlement").startswith("C") else "PHYSICAL_DELIVERY"
TENOR, DELAY, MAX_GAP = months("Tenor Months"), months("Autocall Delay"), months("Max Gap Days")
FREQ = frequency("Autocall Frequency") if PRODUCT == "FCN" else None
# first observation = delay + one period, so never the launch day itself
OBS_MONTHS = range(DELAY + FREQ, TENOR, FREQ) if FREQ else []
BAD = "KNOCKED_OUT" if PRODUCT == "BSF" else LOSS

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
# 2+ price columns = worst-of basket: every test uses the worst performer vs. its own launch level
UNDERLYING = (text("Ticker") or "/".join(NAMES)) + (" worst-of" if len(NAMES) > 1 else "")
launch_start = pd.Timestamp(p["Launch Start"])
launch_end = dates[-1] if pd.isna(p["Launch End"]) else pd.Timestamp(p["Launch End"])


def map_obs(target):
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
    """(outcome, row to mark on the chart) for the note launched on trading day i - same rules as the
    backtest cell. Row = launch day, except a BSF knock-out = the day the barrier was hit."""
    s0 = px[i]
    j = map_obs(dates[i] + pd.DateOffset(months=TENOR))
    if FREQ is not None and autocalled(i, j):
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
        return ("UPSIDE_PAID" if end >= STRIKE else "NO_UPSIDE_BELOW_STRIKE"), i
    breached = PRODUCT != "BRC" or (path <= BARRIER).any()  # FCN/RC have no barrier: strike test alone
    if breached and end < STRIKE:
        return LOSS, i
    # BRC only: barrier was hit during the life but the note recovered to >= strike, so cash after all
    return ("BREACHED_RECOVERED" if PRODUCT == "BRC" and breached else "MATURITY_CASH"), i


# single ticker: actual price; basket: worst-of performance vs. first date
y = px[:, 0] if px.shape[1] == 1 else (px / px[0]).min(axis=1)
bad, pending, n = [], [], 0
for i, d in enumerate(dates):
    if launch_start <= d <= launch_end:
        n += 1
        outcome, k = classify(i)
        if outcome == BAD:
            bad.append(dates[k])
        elif outcome == "OUTSTANDING":
            pending.append(dates[i])
days = sorted(set(bad))  # many launches knock out on the same day - draw each day once
when = "knock-out date" if PRODUCT == "BSF" else "launch date"

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(dates, y, color="black", lw=0.8)
# grey = launches with no outcome yet, so "no red here" isn't misread as "no bad outcomes here"
ax.vlines(pending, 0, 1, transform=ax.get_xaxis_transform(), colors="grey", lw=0.5, alpha=0.15)
ax.vlines(days, 0, 1, transform=ax.get_xaxis_transform(), colors="firebrick", lw=0.5,
          alpha=0.5 if PRODUCT == "BSF" else 0.15)  # full height; BSF days are de-duplicated, so less overlap
ax.plot([], [], color="firebrick", label=f"{BAD}: {when} ({len(bad)} of {n} launches"
                                             + (f", {len(days)} days)" if PRODUCT == "BSF" else ")"))  # solid legend swatch
if pending:
    ax.plot([], [], color="lightgrey", label=f"OUTSTANDING: launch date, maturity not reached yet ({len(pending)})")
ax.set(title=f"{UNDERLYING} {PRODUCT} - {TERMS}",
       ylabel="Price" if px.shape[1] == 1 else "Worst-of vs. start (x)")
ax.legend(loc="upper left", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig
