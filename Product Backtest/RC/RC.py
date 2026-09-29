"""Rolling historical backtest of a worst-of Reverse Convertible: no autocall, one strike test at
maturity. Standalone-Python companion to `RC Python-in-Excel.py` (same rule). History: DEVLOG.md."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _backtest import fetch, run, report  # noqa: E402

TICKERS = ["PFE"]            # 2+ tickers = worst-of basket
STRIKE = 0.70                # vs. launch level, tested only at maturity
TENOR_MONTHS = 12
LAUNCH_START, LAUNCH_END, DATA_AS_OF = "2001-09-18", None, None   # None -> latest available
BUCKETS = {"MATURITY_CASH": "seagreen", "PHYSICAL_DELIVERY": "firebrick", "OUTSTANDING": "lightgrey"}


def classify(w, mat, obs_at):
    if mat is None:
        return "OUTSTANDING"
    return "MATURITY_CASH" if w[mat] >= STRIKE else "PHYSICAL_DELIVERY"


def self_test():
    path = lambda *pts: np.array(pts, dtype=float)
    assert classify(path(1, 0.1, STRIKE), 2, None) == "MATURITY_CASH"          # strike inclusive, path ignored
    assert classify(path(1, 1.5, STRIKE - 0.01), 2, None) == "PHYSICAL_DELIVERY"
    assert classify(path(1, 0.1), None, None) == "OUTSTANDING"


if __name__ == "__main__":
    self_test()
    prices, as_of = fetch(TICKERS, LAUNCH_START, DATA_AS_OF)
    res = run(prices, TENOR_MONTHS, classify, LAUNCH_START, LAUNCH_END)
    report(res, prices, BUCKETS, f"{'/'.join(TICKERS)} RC  (strike {STRIKE:.0%}, tenor {TENOR_MONTHS}m, "
           f"as-of {as_of.date()})", __file__)
