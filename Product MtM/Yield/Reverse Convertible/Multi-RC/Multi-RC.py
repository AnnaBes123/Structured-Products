import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql

plt.rcParams["font.family"] = "Arial"

import sys

_PRODUCT_MTM_ROOT = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(_PRODUCT_MTM_ROOT) != "Product MtM":
    _PRODUCT_MTM_ROOT = os.path.dirname(_PRODUCT_MTM_ROOT)
if _PRODUCT_MTM_ROOT not in sys.path:
    sys.path.insert(0, _PRODUCT_MTM_ROOT)

from _common import (
    fetch_daily_closes,
    fetch_dividend_yield,
    fetch_underlying_name,
    _quantlib_process,
    zcb_price_and_greeks,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".png")

# --- Product terms (same base terms as "Reverse Convertible.py", one folder
# up, plus the worst-of basket feature - see README for the full rationale) ---
STRIKE = 0.90                  # worst-of put strike, as a fraction of each name's OWN entry level
RISK_FREE_RATE = 0.04          # SOFR proxy - used for BOTH the ZCB leg and the option leg
GS_CDS_SPREAD = 0.005308       # Goldman Sachs 5y CDS, 53.08 bps - issuer credit spread, ZCB leg only

ENTRY_DATE = "2025-01-02"
TENOR = 1

TICKERS = ["AAPL", "JPM", "XOM"]  # 3 sectors on purpose - see README "dispersion" discussion
CORRELATION_LOOKBACK_YEARS = 2   # trailing window ending at ENTRY_DATE, no look-ahead

N_MC_PATHS = 50000
MC_SEED = 42

SPOT_SCENARIO_RANGE = np.arange(0.60, 1.41, 0.05)  # 60% to 140% of S0, in 5pt steps


def fetch_multi_asset_path(tickers, start, end):
    """Daily closes for every basket name, aligned on shared trading
    dates (inner join) to guard against one name's missing print."""
    series = {ticker: fetch_daily_closes(ticker, start, end) for ticker in tickers}
    df = pd.concat(series, axis=1)
    df = df.dropna(how="any")
    if df.index[-1] < end - pd.Timedelta(days=10):
        raise RuntimeError(
            f"Requested window {start.date()} to {end.date()} extends past the last available "
            f"trading day ({df.index[-1].date()}) - this script models a COMPLETED historical "
            f"window, not a live in-progress note. Pick an ENTRY_DATE/TENOR combination that ends "
            f"on or before today."
        )
    return df


def estimate_vols_and_correlation(calibration_price_df):
    """Realized annualized vol per name plus the full correlation
    matrix, both from the same trailing real daily-close history."""
    log_returns = np.log(calibration_price_df / calibration_price_df.shift(1)).dropna()
    vols = log_returns.std() * np.sqrt(252)
    corr = log_returns.corr()
    return vols.to_dict(), corr


def black_scholes_put(S, K, T, r, sigma, q=0.0):
    """Plain European put via QuantLib's AnalyticEuropeanEngine - the N=1
    degenerate case, used only in verify_against_closed_form."""
    if T <= 0:
        return {"price": max(K - S, 0.0)}

    today = ql.Date(1, 1, 2000)
    ql.Settings.instance().evaluationDate = today
    process = _quantlib_process(S, r, q, sigma, today)

    days = max(int(round(T * 365)), 1)
    exercise = ql.EuropeanExercise(today + ql.Period(days, ql.Days))
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return {"price": option.NPV()}


# REPLICATION: ZCB(Principal) - Short Put on worst-of relative performance,
# priced NATIVELY via QuantLib's basket-option machinery (no hand-rolled
# Monte Carlo needed - see README.md and MATHEMATICS.md section 3).

def worst_of_put_price(relative_spots, K, T, r, sigmas, qs, corr_matrix, n_paths=N_MC_PATHS, seed=MC_SEED):
    """MC price of a worst-of put via QuantLib's native basket engine
    (StochasticProcessArray + MinBasketPayoff + MCEuropeanBasketEngine).
    `relative_spots[i]` is name i's performance relative to its own
    entry level."""
    n = len(relative_spots)
    today = ql.Date(1, 1, 2000)
    ql.Settings.instance().evaluationDate = today

    processes = [_quantlib_process(relative_spots[i], r, qs[i], sigmas[i], today) for i in range(n)]

    corr = ql.Matrix(n, n)
    for i in range(n):
        for j in range(n):
            corr[i][j] = float(corr_matrix[i][j])

    process_array = ql.StochasticProcessArray(processes, corr)
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    basket_payoff = ql.MinBasketPayoff(payoff)

    days = max(int(round(T * 365)), 1)
    exercise = ql.EuropeanExercise(today + ql.Period(days, ql.Days))
    option = ql.BasketOption(basket_payoff, exercise)

    engine = ql.MCEuropeanBasketEngine(process_array, "PseudoRandom", timeStepsPerYear=1,
                                        requiredSamples=n_paths, seed=seed)
    option.setPricingEngine(engine)
    return option.NPV()


def stulz_put_price(relative_spot_1, relative_spot_2, K, T, r, sigma_1, sigma_2, q_1, q_2, rho):
    """Closed-form (Stulz 1982) 2-asset worst-of put via QuantLib's
    StulzEngine - a convergence benchmark for worst_of_put_price at N=2
    only, never the live pricer."""
    today = ql.Date(1, 1, 2000)
    ql.Settings.instance().evaluationDate = today
    p1 = _quantlib_process(relative_spot_1, r, q_1, sigma_1, today)
    p2 = _quantlib_process(relative_spot_2, r, q_2, sigma_2, today)

    payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    basket_payoff = ql.MinBasketPayoff(payoff)
    days = max(int(round(T * 365)), 1)
    exercise = ql.EuropeanExercise(today + ql.Period(days, ql.Days))
    option = ql.BasketOption(basket_payoff, exercise)
    option.setPricingEngine(ql.StulzEngine(p1, p2, rho))
    return option.NPV()


def multi_rc_price(relative_spots, T_remaining, sigmas, qs, corr_matrix, r=RISK_FREE_RATE,
                     credit_spread=GS_CDS_SPREAD, n_paths=N_MC_PATHS, seed=MC_SEED):
    """Principal = 1.0 ("par" units) - no single dollar S0 to anchor to
    with more than one underlying."""
    zcb = zcb_price_and_greeks(1.0, T_remaining, r + credit_spread)
    put_quantity = 1.0 / STRIKE

    if T_remaining <= 0:
        worst_of = min(relative_spots)
        put_payoff = max(STRIKE - worst_of, 0.0)
        return {"price": zcb["price"] - put_quantity * put_payoff, "worst_of": worst_of}

    put_price = worst_of_put_price(relative_spots, STRIKE, T_remaining, r, sigmas, qs, corr_matrix, n_paths, seed)
    return {"price": zcb["price"] - put_quantity * put_price, "worst_of": min(relative_spots)}


def verify_against_closed_form(sigmas, qs, corr_matrix, T, r=RISK_FREE_RATE, n_paths=N_MC_PATHS):
    """N=1 and N=2 reduction checks against independent QuantLib
    closed-form benchmarks - see MATHEMATICS.md section 4."""
    single_relative = [1.0]
    mc_single = worst_of_put_price(single_relative, STRIKE, T, r, sigmas[:1], qs[:1],
                                     np.array([[1.0]]), n_paths=n_paths, seed=MC_SEED)
    vanilla_single = black_scholes_put(1.0, STRIKE, T, r, sigmas[0], qs[0])["price"]
    n1_rel_diff = abs(mc_single - vanilla_single) / vanilla_single

    two_relative = [1.0, 1.0]
    rho_12 = float(corr_matrix[0][1])
    mc_two = worst_of_put_price(two_relative, STRIKE, T, r, sigmas[:2], qs[:2],
                                  np.array(corr_matrix)[:2, :2], n_paths=200_000, seed=MC_SEED)
    stulz_two = stulz_put_price(1.0, 1.0, STRIKE, T, r, sigmas[0], sigmas[1], qs[0], qs[1], rho_12)
    n2_rel_diff = abs(mc_two - stulz_two) / stulz_two

    return {
        "mc_single": mc_single, "vanilla_single": vanilla_single, "n1_rel_diff": n1_rel_diff,
        "mc_two": mc_two, "stulz_two": stulz_two, "n2_rel_diff": n2_rel_diff,
    }


def worst_of_running_return(S0_list, price_df):
    """Redemption Payoff (Relative to Par), worst-of - see README
    "Reading the charts"."""
    relative = price_df / pd.Series(S0_list, index=price_df.columns)
    worst_of = relative.min(axis=1)
    running = pd.Series(0.0, index=price_df.index)
    below_strike = worst_of < STRIKE
    running[below_strike] = worst_of[below_strike] / STRIKE - 1.0
    return running, worst_of


def multi_rc_mtm_price_series(S0_list, price_df, sigmas, qs, corr_matrix, r=RISK_FREE_RATE,
                                credit_spread=GS_CDS_SPREAD, n_paths=N_MC_PATHS):
    maturity_date = price_df.index[-1]
    relative_path = price_df / pd.Series(S0_list, index=price_df.columns)

    prices = []
    for date, row in relative_path.iterrows():
        T_remaining = (maturity_date - date).days / 365.25
        result = multi_rc_price(row.values.tolist(), T_remaining, sigmas, qs, corr_matrix, r, credit_spread, n_paths)
        prices.append(result["price"])
    return pd.Series(prices, index=price_df.index)


def finite_difference_greeks(relative_spots, T, sigmas, qs, corr_matrix, r=RISK_FREE_RATE,
                              credit_spread=GS_CDS_SPREAD, n_paths=N_MC_PATHS, seed=MC_SEED):
    """Bump-and-reprice Greeks, one Delta/Vega per name (risk is a
    vector). Common random numbers (MC_SEED) - bump sizes confirmed
    stable in the README's bump-size scan."""
    n = len(relative_spots)
    bump_S, bump_sigma, bump_r, bump_T = 0.01, 0.005, 0.0001, 7 / 365

    base = multi_rc_price(relative_spots, T, sigmas, qs, corr_matrix, r, credit_spread, n_paths, seed)

    deltas = []
    for i in range(n):
        up = list(relative_spots); up[i] += bump_S
        down = list(relative_spots); down[i] -= bump_S
        price_up = multi_rc_price(up, T, sigmas, qs, corr_matrix, r, credit_spread, n_paths, seed)["price"]
        price_down = multi_rc_price(down, T, sigmas, qs, corr_matrix, r, credit_spread, n_paths, seed)["price"]
        deltas.append((price_up - price_down) / (2 * bump_S))

    vegas = []
    for i in range(n):
        sig_up = list(sigmas); sig_up[i] += bump_sigma
        sig_down = list(sigmas); sig_down[i] -= bump_sigma
        price_up = multi_rc_price(relative_spots, T, sig_up, qs, corr_matrix, r, credit_spread, n_paths, seed)["price"]
        price_down = multi_rc_price(relative_spots, T, sig_down, qs, corr_matrix, r, credit_spread, n_paths, seed)["price"]
        vegas.append((price_up - price_down) / (2 * bump_sigma))

    price_up_r = multi_rc_price(relative_spots, T, sigmas, qs, corr_matrix, r + bump_r, credit_spread, n_paths, seed)["price"]
    price_down_r = multi_rc_price(relative_spots, T, sigmas, qs, corr_matrix, r - bump_r, credit_spread, n_paths, seed)["price"]
    rho = (price_up_r - price_down_r) / (2 * bump_r)

    price_less_T = multi_rc_price(relative_spots, max(T - bump_T, 0.0), sigmas, qs, corr_matrix, r, credit_spread, n_paths, seed)["price"]
    theta = (price_less_T - base["price"]) / bump_T

    return {"price": base["price"], "deltas": deltas, "vegas": vegas, "rho": rho, "theta": theta}


def plot_path(price_df, S0_list, tickers, underlying_names, mtm_enabled=True, sigmas=None, qs=None,
              corr_matrix=None, greeks=None):
    index_return_pct = (price_df / pd.Series(S0_list, index=price_df.columns) - 1) * 100
    tracker, worst_of = worst_of_running_return(S0_list, price_df)
    tracker_pct = tracker * 100

    _, ax = plt.subplots(figsize=(18, 8))
    asset_colors = ["steelblue", "darkorange", "seagreen", "mediumorchid", "peru", "slategray"]
    for i, ticker in enumerate(tickers):
        ax.plot(index_return_pct.index, index_return_pct[ticker].values, color=asset_colors[i % len(asset_colors)],
                linewidth=1.1, alpha=0.85, label=underlying_names[i])

    ax.plot(tracker_pct.index, tracker_pct.values, color="indianred", linewidth=1.8,
            linestyle="dashed", label="Multi-RC Redemption Payoff (Relative to Par) (worst-of)")

    ax.grid(True, which="major", color="lightgrey", linewidth=0.6)
    ax.axhline(0, color="lightgrey", linewidth=0.8)
    strike_pct = (STRIKE - 1) * 100
    ax.axhline(strike_pct, color="dodgerblue", linewidth=0.8, linestyle="dotted")
    ax.text(0.01, strike_pct, f"Strike: {strike_pct:.1f}%", transform=ax.get_yaxis_transform(),
            color="dodgerblue", fontsize=9, va="bottom", ha="left")

    ax.set_xlabel("Date")
    ax.set_ylabel("Return from Entry (%)")
    lines, labels = ax.get_legend_handles_labels()
    ax.margins(x=0, y=0.05)

    if greeks is not None:
        delta_lines = "\n".join(f"  {t}: {d:.3f}" for t, d in zip(tickers, greeks["deltas"]))
        vega_lines = "\n".join(f"  {t}: {v:.4f}" for t, v in zip(tickers, greeks["vegas"]))
        greeks_text = (
            "Greeks at Inception:\n"
            f"Δ (Delta) per name:\n{delta_lines}\n"
            f"ν (Vega) per name (per 1% vol):\n{vega_lines}\n"
            f"ρ (Rho): {greeks['rho'] * 0.01:.4f} per 1% change in SOFR\n"
            f"θ (Theta): {greeks['theta']:.4f} per year"
        )
        ax.text(1.16, 0.5, greeks_text, transform=ax.transAxes,
                fontsize=9, fontweight="light", color="black", va="center", ha="left")

    combined_min = min(index_return_pct.min().min(), tracker_pct.min(), strike_pct)
    combined_max = max(index_return_pct.max().max(), tracker_pct.max())

    if mtm_enabled:
        mtm_price = multi_rc_mtm_price_series(S0_list, price_df, sigmas, qs, corr_matrix)
        mtm_gain_over_par_pct = (mtm_price - 1.0) * 100

        ax2 = ax.twinx()
        mtm_line, = ax2.plot(
            mtm_gain_over_par_pct.index, mtm_gain_over_par_pct.values, color="darkred", linewidth=1.8,
            linestyle="solid", label="Multi-RC — Model Value of Redemption Component (% of Par, excl. coupons, QuantLib basket MC)")
        ax2.set_ylabel("Redemption Component Value vs. Par (%)", color="darkred", rotation=270, labelpad=10)
        ax2.tick_params(axis="y", labelcolor="darkred")

        combined_min = min(combined_min, mtm_gain_over_par_pct.min())
        combined_max = max(combined_max, mtm_gain_over_par_pct.max())
        pad = (combined_max - combined_min) * 0.05
        ax.set_ylim(combined_min - pad, combined_max + pad)
        ax2.set_ylim(combined_min - pad, combined_max + pad)

        lines.append(mtm_line)
        labels.append(mtm_line.get_label())
    else:
        pad = (combined_max - combined_min) * 0.05
        ax.set_ylim(combined_min - pad, combined_max + pad)

    basket_label = " / ".join(underlying_names)
    end_date = index_return_pct.index[-1]
    ax.set_title(f"Worst-of Basket ({basket_label}) Price Return Path from {price_df.index[0].date()} to {end_date.date()} \n"
                 f"vs Model Value of Redemption Component (Excludes Coupons) — Multi-RC")
    ax.legend(lines, labels, loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    underlying_names = [fetch_underlying_name(t) for t in TICKERS]
    print(f"Basket: {', '.join(f'{n} ({t})' for n, t in zip(underlying_names, TICKERS))}")

    entry_ts = pd.Timestamp(ENTRY_DATE)
    maturity_ts = entry_ts + pd.Timedelta(days=round(TENOR * 365.25))
    calibration_start = entry_ts - pd.Timedelta(days=round(CORRELATION_LOOKBACK_YEARS * 365.25))

    print(f"\nFetching {CORRELATION_LOOKBACK_YEARS}y TRAILING history (ending {ENTRY_DATE}, before the note")
    print(f"starts) to estimate vol and correlation from real data - no look-ahead into the backtest window...")
    calibration_df = fetch_multi_asset_path(TICKERS, calibration_start, entry_ts)
    vols_dict, corr_df = estimate_vols_and_correlation(calibration_df)
    sigmas = [vols_dict[t] for t in TICKERS]
    corr_matrix = corr_df.loc[TICKERS, TICKERS].values

    print(f"\nRealized annualized vol (from the {CORRELATION_LOOKBACK_YEARS}y trailing window):")
    for t, n, s in zip(TICKERS, underlying_names, sigmas):
        print(f"  {n} ({t}): {s:.2%}")

    print(f"\nRealized correlation matrix (from the same trailing window):")
    print(corr_df.loc[TICKERS, TICKERS].to_string(float_format=lambda x: f"{x:.3f}"))

    dividend_yields = [fetch_dividend_yield(t) for t in TICKERS]
    print(f"\nDividend yields (flat, continuous):")
    for t, n, q in zip(TICKERS, underlying_names, dividend_yields):
        print(f"  {n} ({t}): {q:.2%}")

    print(f"\nFetching {ENTRY_DATE} (entry) to +{TENOR}y (maturity) backtest path...")
    price_df = fetch_multi_asset_path(TICKERS, entry_ts, maturity_ts)
    S0_list = [float(price_df[t].iloc[0]) for t in TICKERS]
    S_T_list = [float(price_df[t].iloc[-1]) for t in TICKERS]

    print(f"\nVerifying the QuantLib basket-option wiring (N=1 reduces to a plain vanilla put; N=2")
    print(f"converges onto QuantLib's closed-form StulzEngine as path count grows):")
    check = verify_against_closed_form(sigmas, dividend_yields, corr_matrix, TENOR)
    print(f"  N=1: MC basket price={check['mc_single']:.5f}  vanilla put (closed form)={check['vanilla_single']:.5f}"
          f"  rel diff={check['n1_rel_diff']:.3%}  "
          f"({'PASS' if check['n1_rel_diff'] < 0.01 else 'FAIL - investigate before trusting results'})")
    print(f"  N=2: MC basket (200k paths)={check['mc_two']:.5f}  Stulz closed form={check['stulz_two']:.5f}"
          f"  rel diff={check['n2_rel_diff']:.3%}  "
          f"({'PASS' if check['n2_rel_diff'] < 0.02 else 'FAIL - investigate before trusting results'})")

    tracker, worst_of = worst_of_running_return(S0_list, price_df)
    note_return = tracker.iloc[-1]
    worst_performer_idx = int(np.argmin([S_T_list[i] / S0_list[i] for i in range(len(TICKERS))]))

    summary_rows = {
        "Entry Date": price_df.index[0].date(),
        "Maturity Date": price_df.index[-1].date(),
        "Strike": f"{STRIKE:.0%} of each name's own entry level",
    }
    for t, n, s0, st in zip(TICKERS, underlying_names, S0_list, S_T_list):
        summary_rows[f"{n} ({t}) Return"] = f"{st / s0 - 1:+.2%}"
    summary_rows["Worst Performer"] = f"{underlying_names[worst_performer_idx]} ({TICKERS[worst_performer_idx]})"
    summary_rows["Worst-of Relative Performance (T)"] = f"{worst_of.iloc[-1]:.2%}"
    summary_rows["Multi-RC Return"] = f"{note_return:.2%}"

    print("\n" + pd.Series(summary_rows).to_string())

    greeks = finite_difference_greeks([1.0] * len(TICKERS), TENOR, sigmas, dividend_yields, corr_matrix)
    fair_value_pct_of_par = greeks["price"]

    print(f"\nMulti-RC — model value of redemption component at inception (excludes coupons)")
    print(f"(QuantLib MCEuropeanBasketEngine, {N_MC_PATHS:,} paths, ZCB discounted at SOFR")
    print(f"{RISK_FREE_RATE:.2%} + Goldman Sachs CDS {GS_CDS_SPREAD:.2%}, worst-of put at SOFR alone,")
    print(f"T={TENOR}y, strike={STRIKE:.0%}):")
    print(f"  {fair_value_pct_of_par:.2%} of par ({fair_value_pct_of_par - 1:+.2%} vs. par)")

    for t, n, d, v in zip(TICKERS, underlying_names, greeks["deltas"], greeks["vegas"]):
        print(f"  Delta ({n}): {d:.3f}   Vega ({n}, per 1% vol): {v * 0.01:.4f}")
    print(f"  Rho: {greeks['rho'] * 0.01:.4f}  (per 1% change in SOFR)")
    print(f"  Theta: {greeks['theta']:.4f} per year / {greeks['theta'] / 365:.5f} per day")

    print(f"\nFor comparison, a SINGLE-name Reverse Convertible at the same {STRIKE:.0%} strike on just the worst")
    print(f"performer alone would be worth (closed form, ignoring the other two names entirely):")
    single_name_zcb = zcb_price_and_greeks(1.0, TENOR, RISK_FREE_RATE + GS_CDS_SPREAD)
    for t, n, s, q in zip(TICKERS, underlying_names, sigmas, dividend_yields):
        single_put = black_scholes_put(1.0, STRIKE, TENOR, RISK_FREE_RATE, s, q)["price"]
        single_price = single_name_zcb["price"] - (1.0 / STRIKE) * single_put
        print(f"  {n} alone: {single_price:.2%} of par")
    print(f"  Multi-RC (worst-of all three): {fair_value_pct_of_par:.2%} of par  <- should be LOWER than")
    print(f"  every single-name figure above: the worst-of feature means there are three independent")
    print(f"  chances for one name to breach the strike, not just one - this is the real cost of")
    print(f"  dispersion, and it should show up here as a real, computed number, not an assertion.")

    plot_path(price_df, S0_list, TICKERS, underlying_names, mtm_enabled=True,
              sigmas=sigmas, qs=dividend_yields, corr_matrix=corr_matrix, greeks=greeks)
