"""Rolling historical backtest of a worst-of Barrier Reverse Convertible: no autocall; delivery only if
the down-and-in barrier was touched AND maturity finishes below strike (same terminal convention as
Product MtM's BRC). Standalone-Python companion to `BRC Python-in-Excel.py`. History: DEVLOG.md."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _backtest import fetch, run, report  # noqa: E402

TICKERS = ["PFE"]            # 2+ tickers = worst-of basket
STRIKE = 1.00
BARRIER = 0.70               # should be <= STRIKE
AMERICAN = False             # True: barrier checked every trading day; False: maturity only
TENOR_MONTHS = 12
LAUNCH_START, LAUNCH_END, DATA_AS_OF = "2001-09-18", None, None   # None -> latest available
BUCKETS = {"MATURITY_CASH": "seagreen", "PHYSICAL_DELIVERY": "firebrick", "OUTSTANDING": "lightgrey"}


def classify(w, mat, obs_at, american=AMERICAN):
    if mat is None:
        return "OUTSTANDING"
    touched = (w[:mat + 1] <= BARRIER).any() if american else w[mat] <= BARRIER
    return "PHYSICAL_DELIVERY" if touched and w[mat] < STRIKE else "MATURITY_CASH"


def self_test():
    path = lambda *pts: np.array(pts, dtype=float)
    dip = path(1, BARRIER - 0.05, STRIKE - 0.01)    # touched mid-life, ends just below strike
    assert classify(dip, 2, None, american=True) == "PHYSICAL_DELIVERY"
    assert classify(dip, 2, None, american=False) == "MATURITY_CASH"          # European ignores the dip
    assert classify(path(1, 1.2, BARRIER), 2, None, american=False) == "PHYSICAL_DELIVERY"   # barrier inclusive
    assert classify(path(1, BARRIER - 0.1, STRIKE), 2, None, american=True) == "MATURITY_CASH"  # strike inclusive
    assert classify(path(1, 0.5), None, None) == "OUTSTANDING"


if __name__ == "__main__":
    self_test()
    prices, as_of = fetch(TICKERS, LAUNCH_START, DATA_AS_OF)
    res = run(prices, TENOR_MONTHS, classify, LAUNCH_START, LAUNCH_END)
    report(res, prices, BUCKETS, f"{'/'.join(TICKERS)} BRC  (strike {STRIKE:.0%}, barrier {BARRIER:.0%} "
           f"{'American' if AMERICAN else 'European'}, tenor {TENOR_MONTHS}m, as-of {as_of.date()})", __file__)
