"""
Condensed FCN backtest: percentages only, no audit CSV, no charts, no diagnostics. Reuses FCN.py's
own fetching/classification (imported below, not duplicated) so the two never drift apart. Run
FCN.py itself first if you want its synthetic correctness test suite - this script assumes that
logic is already trustworthy and only prints the headline numbers. See README.md / DEVLOG.md.
"""

from collections import Counter

import pandas as pd

from FCN import PRODUCTS, product_label, resolve_run_dates, fetch_multi_asset_path, validate_price_df, run_backtest


def condensed_report(terms):
    name = product_label(terms)
    launch_start, launch_end, data_as_of = resolve_run_dates()
    price_df = fetch_multi_asset_path(terms["TICKERS"], launch_start, data_as_of + pd.Timedelta(days=1))
    price_df = price_df[price_df.index <= data_as_of]
    validate_price_df(price_df, terms["TICKERS"])

    results, _coverage = run_backtest(price_df, terms, launch_start, launch_end, data_as_of)
    n_total = len(results)
    counts = Counter(r["outcome"] for r in results)
    completed = [r for r in results if r["full_tenor_observable"]]
    n_completed = len(completed)
    counts_completed = Counter(r["outcome"] for r in completed)

    print(f"\n{name}  ({n_total} launches, data as-of {data_as_of.date()})")
    print(f"  All launches:      AUTOCALL {counts.get('AUTOCALL', 0) / n_total:.1%}   "
          f"MATURITY_CASH {counts.get('MATURITY_CASH', 0) / n_total:.1%}   "
          f"PHYSICAL_DELIVERY {counts.get('PHYSICAL_DELIVERY', 0) / n_total:.1%}   "
          f"OUTSTANDING {counts.get('OUTSTANDING', 0) / n_total:.1%}")
    if n_completed:
        print(f"  Completed cohort:  AUTOCALL {counts_completed.get('AUTOCALL', 0) / n_completed:.1%}   "
              f"MATURITY_CASH {counts_completed.get('MATURITY_CASH', 0) / n_completed:.1%}   "
              f"PHYSICAL_DELIVERY {counts_completed.get('PHYSICAL_DELIVERY', 0) / n_completed:.1%}")
    else:
        print("  Completed cohort:  n/a (every launch still OUTSTANDING)")


if __name__ == "__main__":
    for _terms in PRODUCTS:
        condensed_report(_terms)
