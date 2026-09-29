"""Rolling historical backtest of a worst-of, physically-settled, autocallable Fixed Coupon Note with
a European (maturity-only) strike test. See README.md (methodology, biases) and DEVLOG.md (history)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _backtest import fetch, run, report  # noqa: E402

TICKERS = ["AMD"]            # 2+ tickers = worst-of basket
STRIKE = 0.90                # vs. launch level, tested only at maturity
AUTOCALL_TRIGGER = 1.00      # vs. launch level, tested at each observation
TENOR_MONTHS = 12
AUTOCALL_FREQUENCY_MONTHS = 3    # None disables autocall
AUTOCALL_LOCKOUT_MONTHS = 0
LAUNCH_START, LAUNCH_END, DATA_AS_OF = "2001-01-02", None, None   # None -> latest available

# Starts at lockout+freq, so a launch-day (month-0) autocall is impossible by construction.
OBS_MONTHS = (range(AUTOCALL_LOCKOUT_MONTHS + AUTOCALL_FREQUENCY_MONTHS, TENOR_MONTHS, AUTOCALL_FREQUENCY_MONTHS)
              if AUTOCALL_FREQUENCY_MONTHS else [])
BUCKETS = {"AUTOCALL": "dodgerblue", "MATURITY_CASH": "seagreen",
           "PHYSICAL_DELIVERY": "firebrick", "OUTSTANDING": "lightgrey"}


def classify(w, mat, obs_at):
    for m in OBS_MONTHS:
        k = obs_at(m)
        # must resolve strictly before maturity's own trading day - that day belongs to the strike test
        if k is not None and 0 < k and (mat is None or k < mat) and w[k] >= AUTOCALL_TRIGGER:
            return "AUTOCALL"
    if mat is None:
        return "OUTSTANDING"
    return "MATURITY_CASH" if w[mat] >= STRIKE else "PHYSICAL_DELIVERY"


def self_test():
    """Synthetic paths: 1 obs every 3 idx units, maturity at idx 12 (obs_at(m) == m)."""
    path = lambda *pts: np.array(pts, dtype=float)
    obs = lambda m: m
    flat = [1.0] * 13
    assert classify(path(*flat[:3], 1.1, *flat[4:]), 12, obs) == "AUTOCALL"
    assert classify(path(1, *[0.95] * 12), 12, obs) == "MATURITY_CASH"
    assert classify(path(1, *[0.95] * 11, STRIKE), 12, obs) == "MATURITY_CASH"          # strike inclusive
    assert classify(path(1, *[0.95] * 11, 0.5), 12, obs) == "PHYSICAL_DELIVERY"
    assert classify(path(1, *[0.95] * 2, AUTOCALL_TRIGGER, *[0.5] * 9), 12, obs) == "AUTOCALL"  # trigger inclusive
    assert classify(path(1, *[0.5] * 12), 12, obs) == "PHYSICAL_DELIVERY"              # no month-0 autocall
    assert classify(path(1, *[0.95] * 5), None, lambda m: m if m < 6 else None) == "OUTSTANDING"
    assert classify(path(1, *[0.95] * 5), 5, lambda m: 5) != "AUTOCALL"                # obs collides w/ maturity
    assert classify(path(1, *[0.95] * 12), 12, lambda m: None) == "MATURITY_CASH"      # gap -> skipped obs


if __name__ == "__main__":
    self_test()
    prices, as_of = fetch(TICKERS, LAUNCH_START, DATA_AS_OF)
    res = run(prices, TENOR_MONTHS, classify, LAUNCH_START, LAUNCH_END)
    report(res, prices, BUCKETS, f"{'/'.join(TICKERS)} FCN  (strike {STRIKE:.0%}, autocall {AUTOCALL_TRIGGER:.0%} "
           f"every {AUTOCALL_FREQUENCY_MONTHS}m, tenor {TENOR_MONTHS}m, as-of {as_of.date()})", __file__)
