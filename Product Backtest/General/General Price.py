"""
Body of ONE Excel =PY() cell: price path chart - same params (R3:S15), same price
range (A:P, up to 15 tickers = worst-of) and same outcome rules as `General Python-in-Excel.py`, so every
line here is a launch that the table counts in that bucket. Percent inputs (130, "130%", 1.3) and named
autocall frequencies are read exactly as the table reads them.

One vertical line per launch, coloured by outcome, at its launch date - except KNOCKED_OUT, drawn at the
day the knock-out occurred. Which outcomes are drawn comes from a selection grid at R23:S30 on the same sheet (to move it,
edit the xl('R23:S30') address in wanted() - keep it a literal string, not a variable):
    R (Status)                 S (Show: TRUE/FALSE, checkbox, Y/N, x or 1)
    AUTOCALL                   TRUE
    MATURITY_CASH              FALSE
    BREACHED_RECOVERED         FALSE
    PHYSICAL_DELIVERY          TRUE      (also covers CASH_SETTLEMENT - either name works)
    UPSIDE_PAID                FALSE
    NO_UPSIDE_BELOW_STRIKE     FALSE
    KNOCKED_OUT                TRUE
    OUTSTANDING                TRUE
Spaces instead of underscores and any order are fine; rows that don't apply to the Product are ignored.
No grid, or nothing ticked -> the product's bad outcome (loss case; BSF: KNOCKED_OUT) plus OUTSTANDING.

2+ tickers: every underlying is drawn (rebased to 100% at the first date, log scale) plus the worst-of in
black - the line every test uses. Paste everything below the ---8<--- line; the chart shows as an image.

Code notes (kept out of the pasted part - Excel caps a =PY() cell at 8,192 characters, so the code
below the cut line has no comments; reasons live here and in DEVLOG.md):
  - Params are read by row position in R3:S15, not by label, so the R labels can be reworded freely.
  - Ranges have no sheet prefix: the cell reads the sheet it sits on (same sheet as prices, params, grid).
  - Tickers 5-15 take colours from matplotlib's tab20b palette after the 4 fixed ones.
    p.get() lets a sheet that stops at row 14 (no Settlement row) still work.
  - level(): Excel's 130% arrives as 1.3; a typed 130 or "130%" also works - anything above 5 is read
    as percentage points (no real strike/barrier is above 500%).
  - frequency(): named frequencies, or a bare number = months; blank/0 = no autocall. First
    observation = delay + one period, so never the launch day. Daily is stored as the word "DAILY" (never 0 - 0 means no
    autocall) = every trading day after the delay. An observation must fall strictly before the maturity day (that day is the strike test).
  - Price data is cleaned before use: dates may be real dates, text or Excel serial numbers; non-numeric
    prices (#N/A, text) and rows with any blank price are dropped. Blank Launch Start = first price date.
    An unknown Product raises a plain-English error instead of a KeyError.
  - 2+ price columns = worst-of: every test uses the worst performer vs. its own launch level, and a
    loss is settled on it (its shares, or their value in cash - same value, so Settlement only renames).
  - classify(): a BSF American knock-out is final the day it's hit, even before maturity is in the
    data. European = maturity day only. FCN/RC have no barrier (strike test alone); BRC barrier hit
    then recovered to >= strike = BREACHED_RECOVERED (cash).
  - Strings in the pasted code use single quotes: Excel stores each double quote as two characters
    toward the 8,192 limit.
  - Status colours: V23:W30 = status name | colour (name like 'orange' or hex '#1E90FF'), read in the
    same xl('R23:W30') call as the Show grid. They override the COLORS defaults for that run only - the
    code is never edited; blank/invalid -> default.
  - Chart: the status grid is read with a literal xl('R23:W30') because Excel tracks a cell's
    dependencies from the xl("...") text. Grey (outstanding) is drawn first so outcomes sit on top;
    knock-out days are de-duplicated (many launches share one). Red = the bad outcome only (loss, or
    BSF's knock-out - never on the same chart) and drawn stronger. Basket y-axis is log so one big
    winner (e.g. NVDA ~450x) doesn't flatten the worst-of line.
"""

# ---8<--- everything from here down goes into the =PY() cell ---8<---
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mc

raw = xl('R3:S15')
p = dict(zip(['Ticker', 'Product', 'Strike', 'Barrier', 'Barrier Monitoring', 'Autocall Trigger',
    'Autocall Frequency', 'Autocall Delay', 'Tenor Months', 'Launch Start',
    'Launch End', 'Max Gap Days', 'Settlement'], raw.iloc[:, 1]))

def text(label):
    v = p.get(label)
    return '' if v is None or pd.isna(v) else str(v).strip().upper()

def level(label):
    s = text(label)
    x = float(s.rstrip('%') or 0)
    return x / 100 if s.endswith('%') or x > 5 else x

def months(label):
    s = text(label).replace('MONTHS', '').replace('M', '').strip()
    return 0 if not s or s.startswith('NO') else int(float(s))

def frequency(label):
    s = re.sub('[^A-Z0-9.]', '', text(label))
    named = {'DAILY': 'DAILY', 'MONTHLY': 1, 'BIMONTHLY': 2, 'QUARTERLY': 3, 'QUARTER': 3, 'SEMIANNUALLY': 6,
        'SEMIANNUAL': 6, 'ANNUALLY': 12, 'ANNUAL': 12}
    if s in named:
        return named[s]
    s = s.replace('MONTHS', '').replace('MONTH', '').rstrip('M')
    if s in ('', 'NONE', 'NO', 'NOAUTOCALL') or float(s) == 0:
        return None
    return int(float(s))

PRODUCT = text('Product')
if PRODUCT not in ('FCN', 'RC', 'BRC', 'BSF'):
    raise ValueError(f'Product (S4) must be FCN, RC, BRC or BSF, not {PRODUCT}')
STRIKE, BARRIER, TRIGGER = level('Strike'), level('Barrier'), level('Autocall Trigger')
AMERICAN = text('Barrier Monitoring').startswith('A')
LOSS = 'CASH_SETTLEMENT' if text('Settlement').startswith('C') else 'PHYSICAL_DELIVERY'
TENOR, DELAY, MAX_GAP = months('Tenor Months'), months('Autocall Delay'), months('Max Gap Days')
FREQ = frequency('Autocall Frequency') if PRODUCT == 'FCN' else None
OBS_MONTHS = range(DELAY + FREQ, TENOR, FREQ) if isinstance(FREQ, int) else []
BAD = 'KNOCKED_OUT' if PRODUCT == 'BSF' else LOSS

MODE = 'American' if AMERICAN else 'European'
FREQ_NAME = {'DAILY': 'daily', 1: 'monthly', 2: 'bi-monthly', 3: 'quarterly', 6: 'semi-annually', 12: 'annually'}.get(FREQ, f'every {FREQ}m')
TERMS = f'strike {STRIKE:.0%}' + {
    'FCN': ', no autocall' if FREQ is None else
    f', autocall {TRIGGER:.0%} {FREQ_NAME}' + (f' after {DELAY}m delay' if DELAY else ''),
    'RC': '',
    'BRC': f', knock-in barrier {BARRIER:.0%} {MODE}',
    'BSF': f', knock-out barrier {BARRIER:.0%} {MODE}'}[PRODUCT] + f', tenor {TENOR}m' + (
    '' if PRODUCT == 'BSF' else ', cash-settled' if LOSS == 'CASH_SETTLEMENT' else ', physically settled')

d = xl('A1:P100000', headers=True)
c = d.iloc[:, 0]
c = pd.to_datetime(c, unit='D', origin='1899-12-30') if c.dtype.kind in 'if' else pd.to_datetime(c, errors='coerce')
prices_df = d.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').set_index(c).dropna(axis=1, how='all').dropna()
prices_df = prices_df[prices_df.index.notna()].sort_index()
if len(prices_df) < 2:
    raise ValueError('No usable prices: dates in A, prices in B:P')
dates, px = prices_df.index, prices_df.to_numpy(dtype=float)
NAMES = [str(c).strip() for c in prices_df.columns]
UNDERLYING = (text('Ticker') and str(p['Ticker']).strip() or '/'.join(NAMES)) + (' worst-of' if len(NAMES) > 1 else '')
launch_start = dates[0] if pd.isna(p['Launch Start']) else pd.Timestamp(p['Launch Start'])
launch_end = dates[-1] if pd.isna(p['Launch End']) else pd.Timestamp(p['Launch End'])

def map_obs(target):
    pos = int(dates.searchsorted(target))
    return None if pos >= len(dates) or (dates[pos] - target).days > MAX_GAP else pos

def autocalled(i, j):
    s0, end = px[i], len(dates) if j is None else j
    if FREQ == 'DAILY':
        first = int(dates.searchsorted(dates[i] + pd.DateOffset(months=DELAY), side='right'))
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
        return 'AUTOCALL', i
    live = (px[i:None if j is None else j + 1] / s0).min(axis=1)
    if PRODUCT == 'BSF' and AMERICAN and (live >= BARRIER).any():
        return 'KNOCKED_OUT', i + int((live >= BARRIER).argmax())
    if j is None:
        return 'OUTSTANDING', i
    path = live if AMERICAN else live[-1:]
    end = path[-1]
    if PRODUCT == 'BSF':
        if end >= BARRIER:
            return 'KNOCKED_OUT', j
        return ('UPSIDE_PAID' if end >= STRIKE else 'NO_UPSIDE_BELOW_STRIKE'), i
    breached = PRODUCT != 'BRC' or (path <= BARRIER).any()
    if breached and end < STRIKE:
        return LOSS, i
    return ('BREACHED_RECOVERED' if PRODUCT == 'BRC' and breached else 'MATURITY_CASH'), i

COLORS = {'AUTOCALL': 'dodgerblue', 'MATURITY_CASH': 'seagreen', 'BREACHED_RECOVERED': 'goldenrod',
    'UPSIDE_PAID': 'darkcyan', 'NO_UPSIDE_BELOW_STRIKE': 'mediumpurple', 'OUTSTANDING': 'grey',
    'KNOCKED_OUT': 'firebrick', LOSS: 'firebrick'}
TICKER_COLORS = ['navy', 'saddlebrown', 'deeppink', 'olive']
APPLIES = {'FCN': ['AUTOCALL', 'MATURITY_CASH', LOSS], 'RC': ['MATURITY_CASH', LOSS],
    'BRC': ['MATURITY_CASH', 'BREACHED_RECOVERED', LOSS],
    'BSF': ['UPSIDE_PAID', 'NO_UPSIDE_BELOW_STRIKE', 'KNOCKED_OUT']}[PRODUCT] + ['OUTSTANDING']

def wanted():
    try:
        g = xl('R23:W30').values.tolist()
    except Exception:
        g = []
    key = {re.sub('[^A-Z]', '', b): b for b in COLORS} | {'PHYSICALDELIVERY': LOSS, 'CASHSETTLEMENT': LOSS}
    show = set()
    for r in g:
        k, v = (key.get(t, key.get(t[:-2])) for t in (re.sub('[^A-Z]', '', str(x).upper()) for x in (r[0], r[4])))
        if k in APPLIES and str(r[1]).strip().upper() in ('TRUE', 'Y', 'YES', 'X', '1', '1.0'):
            show.add(k)
        if v and mc.is_color_like(str(r[5]).strip()):
            COLORS[v] = str(r[5]).strip()
    if show:
        return [b for b in APPLIES if b in show], ''
    return [BAD, 'OUTSTANDING'], '\nDefault view - nothing ticked in R23:S30'

SHOW, NOTE = wanted()
y = px[:, 0] if px.shape[1] == 1 else (px / px[0]).min(axis=1)
marks, n = {b: [] for b in SHOW}, 0
for i, d in enumerate(dates):
    if launch_start <= d <= launch_end:
        n += 1
        outcome, k = classify(i)
        if outcome in marks:
            marks[outcome].append(dates[k])

fig, ax = plt.subplots(figsize=(11, 5.5))
if px.shape[1] > 1:
    for k, name in enumerate(NAMES):
        ax.plot(dates, px[:, k] / px[0, k], lw=0.9, label=name, color=(TICKER_COLORS + list(plt.cm.tab20b.colors))[k], zorder=3)
ax.plot(dates, y, color='black', lw=1.2 if px.shape[1] > 1 else 0.8,
    label='Worst-of (what the note observes)' if px.shape[1] > 1 else None, zorder=3)
for b in sorted(SHOW, key=lambda b: b != 'OUTSTANDING'):
    ax.vlines(sorted(set(marks[b])), 0, 1, transform=ax.get_xaxis_transform(), colors=COLORS[b], lw=0.5,
        alpha=(0.5 if b == 'KNOCKED_OUT' and AMERICAN else 0.25) if b == BAD else 0.12)
for b in SHOW:
    ds, ko = marks[b], b == 'KNOCKED_OUT'
    when = ('knock-out date' if AMERICAN else 'maturity date') if ko else 'launch date'
    lab = f'{b}: {when} ({len(ds)} of {n} launches, {len(ds) / n:.1%}' + (f', on {len(set(ds))} days)' if ko else ')')
    if b == 'AUTOCALL' and not ds:
        lab = 'AUTOCALL: none (check frequency, delay, trigger vs. tenor)'
    ax.plot([], [], color=COLORS[b], label=lab)
ax.set(title=f'{UNDERLYING} {PRODUCT} - {TERMS}{NOTE}',
    ylabel='Price' if px.shape[1] == 1 else 'Performance vs. first date')
if px.shape[1] > 1:
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2 if len(NAMES) < 5 else 4, frameon=False, fontsize=8)
for h in leg.get_lines():
    h.set_linewidth(2)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig
