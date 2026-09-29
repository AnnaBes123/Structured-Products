"""Rolling historical backtest of a Bullish Shark Fin: principal-protected upside participation,
knocked out by an up-and-out barrier. Buckets only, no $ payoff (rebate/participation not modeled).
Standalone-Python companion to `BSF Python-in-Excel.py` (same rule). History: DEVLOG.md."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _backtest import fetch, run, report  # noqa: E402

TICKERS = ["PFE"]            # 2+ tickers = worst-of basket
STRIKE = 1.00                # should be <= BARRIER
BARRIER = 1.15
AMERICAN = False             # True: barrier checked every trading day; False: maturity only
TENOR_MONTHS = 12
LAUNCH_START, LAUNCH_END, DATA_AS_OF = "2001-09-18", None, None   # None -> latest available
BUCKETS = {"PARTICIPATION_ITM": "seagreen", "PARTICIPATION_OTM": "goldenrod",
           "KNOCKED_OUT": "firebrick", "OUTSTANDING": "lightgrey"}


def classify(w, mat, obs_at, american=AMERICAN):
    if mat is None:
        return "OUTSTANDING"
    if (w[:mat + 1] >= BARRIER).any() if american else w[mat] >= BARRIER:
        return "KNOCKED_OUT"
    return "PARTICIPATION_ITM" if w[mat] >= STRIKE else "PARTICIPATION_OTM"


def self_test():
    path = lambda *pts: np.array(pts, dtype=float)
    spike = path(1, BARRIER + 0.05, STRIKE + 0.01)   # knocked mid-life, ends just above strike
    assert classify(spike, 2, None, american=True) == "KNOCKED_OUT"
    assert classify(spike, 2, None, american=False) == "PARTICIPATION_ITM"    # European ignores the spike
    assert classify(path(1, 0.9, BARRIER), 2, None, american=False) == "KNOCKED_OUT"   # barrier inclusive
    assert classify(path(1, 0.9, STRIKE), 2, None, american=True) == "PARTICIPATION_ITM"  # strike inclusive
    assert classify(path(1, 1.1, 0.8), 2, None, american=True) == "PARTICIPATION_OTM"
    assert classify(path(1, 2.0), None, None) == "OUTSTANDING"


if __name__ == "__main__":
    self_test()
    prices, as_of = fetch(TICKERS, LAUNCH_START, DATA_AS_OF)
    res = run(prices, TENOR_MONTHS, classify, LAUNCH_START, LAUNCH_END)
    report(res, prices, BUCKETS, f"{'/'.join(TICKERS)} BSF  (strike {STRIKE:.0%}, barrier {BARRIER:.0%} "
           f"{'American' if AMERICAN else 'European'}, tenor {TENOR_MONTHS}m, as-of {as_of.date()})", __file__)
