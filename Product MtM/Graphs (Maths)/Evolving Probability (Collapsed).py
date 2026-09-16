import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm, multivariate_normal
from scipy.integrate import quad

# 3D evolving probability P3(t) = P(S_t1<=K1, S_t2<=K2, S_t3<=K3), collapsed
# into a (K1,K2) surface (K3 fixed) since a genuine 3D object can't be a
# single time-series line. Closed form Phi3 drives the live animation;
# sequential_triple_integral below is a one-time offline validation of it.
# See README.md for the full Phi2/Phi3 derivation and validation approach.

ENTRY_DATE = "2025-01-02"
OBS_DATE_1 = "2025-06-30"  # t1: end of Q2
OBS_DATE_2 = "2025-09-30"  # t2: end of Q3
OBS_DATE_3 = "2025-12-31"  # t3: end of Q4
K3_PCT = 1.00               # K3 held fixed; surface is over (K1, K2)
MU = 0.04

K_GRID_PCT = np.linspace(0.70, 1.30, 26)  # 70% to 130% of S0

TICKER = "^GSPC"
FRED_SERIES = "SP500"
VOL_TERM_STRUCTURE_TICKERS = {"^VIX": 30, "^VIX3M": 93, "^VIX6M": 182}
VIX_FRED_SERIES = "VIXCLS"


def fetch_daily_closes(ticker, start, end, fred_series=None):
    series = None
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if data is not None and not data.empty:
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            series = close.dropna()
    except Exception as exc:
        print(f"  yfinance failed for {ticker} ({exc})" + (" ; trying FRED fallback..." if fred_series else ""))

    if (series is None or series.empty) and fred_series:
        try:
            import pandas_datareader.data as web
            data = web.DataReader(fred_series, "fred", start, end)
            series = data[fred_series].dropna()
        except Exception as exc:
            raise RuntimeError(f"Could not fetch {ticker} data from either yfinance or FRED: {exc}")

    if series is None or series.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    return series


def interpolate_implied_vol(vols_by_tenor, T_years):
    points = sorted(vols_by_tenor.items())
    if T_years <= points[0][0]:
        return points[0][1]
    if T_years >= points[-1][0]:
        return points[-1][1]
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if t0 <= T_years <= t1:
            var0, var1 = v0 ** 2 * t0, v1 ** 2 * t1
            var_T = var0 + (var1 - var0) * (T_years - t0) / (t1 - t0)
            return np.sqrt(var_T / T_years)
    return points[-1][1]


def d_barrier(S_t, K, T, sigma_t, mu):
    v = mu - 0.5 * sigma_t ** 2
    return (np.log(K / S_t) - v * T) / (sigma_t * np.sqrt(T))


def sequential_triple_integral(S_t, K1, K2, K3, T1, T2, T3, sigma_t, mu):
    """
    Literal triple integral: P3 = int_{-inf}^{lnK1} phi1(y1|St)
                                   [ int_{-inf}^{lnK2} phi2(y2|y1)
                                     [ int_{-inf}^{lnK3} phi3(y3|y2) dy3 ]
                                   dy2 ] dy1
    The innermost integral over y3 is a Gaussian CDF (closed form); the
    other two are done by genuine numerical quadrature (scipy.integrate.quad,
    nested), not by re-deriving the joint multivariate-normal formula.
    """
    v = mu - 0.5 * sigma_t ** 2
    lnS = np.log(S_t)
    m1, s1 = lnS + v * T1, sigma_t * np.sqrt(T1)
    dt2, dt3 = T2 - T1, T3 - T2

    def inner_cdf_y3_given_y2(y2):
        m3, s3 = y2 + v * dt3, sigma_t * np.sqrt(dt3)
        return norm.cdf((np.log(K3) - m3) / s3)

    def middle_integral_given_y1(y1):
        m2, s2 = y1 + v * dt2, sigma_t * np.sqrt(dt2)
        integrand = lambda y2: inner_cdf_y3_given_y2(y2) * norm.pdf((y2 - m2) / s2) / s2
        value, _ = quad(integrand, -np.inf, np.log(K2))
        return value

    outer_integrand = lambda y1: middle_integral_given_y1(y1) * norm.pdf((y1 - m1) / s1) / s1
    result, _ = quad(outer_integrand, -np.inf, np.log(K1))
    return result


print(f"Fetching S&P 500 path from {ENTRY_DATE} to {OBS_DATE_1}...")
close = fetch_daily_closes(TICKER, ENTRY_DATE, OBS_DATE_1, fred_series=FRED_SERIES)
close = close[close.index < pd.Timestamp(OBS_DATE_1)]  # stop the day before t1 (T1 must stay > 0)
S0 = float(close.iloc[0])
K3 = K3_PCT * S0
K_grid = K_GRID_PCT * S0

print("Fetching VIX / VIX3M / VIX6M term structure for the same window...")
vol_term_structure = {}
for ticker, tenor_days in VOL_TERM_STRUCTURE_TICKERS.items():
    fred_series = VIX_FRED_SERIES if ticker == "^VIX" else None
    series = fetch_daily_closes(ticker, ENTRY_DATE, OBS_DATE_1, fred_series=fred_series)
    vol_term_structure[tenor_days / 365.25] = (series / 100.0).reindex(close.index).ffill().bfill()

t1_date, t2_date, t3_date = pd.Timestamp(OBS_DATE_1), pd.Timestamp(OBS_DATE_2), pd.Timestamp(OBS_DATE_3)
dates = close.index
T1_years = np.array([(t1_date - d).days / 365.25 for d in dates])
T2_years = np.array([(t2_date - d).days / 365.25 for d in dates])
T3_years = np.array([(t3_date - d).days / 365.25 for d in dates])
sigma_t_series = np.array([
    interpolate_implied_vol({tenor: s.loc[d] for tenor, s in vol_term_structure.items()}, T1_years[i])
    for i, d in enumerate(dates)
])
S_t_series = close.values

# --- One-time offline validation: sequential triple integral vs. closed-form Phi3 ---
i_check = len(dates) // 2
S_chk, sig_chk = S_t_series[i_check], sigma_t_series[i_check]
T1_chk, T2_chk, T3_chk = T1_years[i_check], T2_years[i_check], T3_years[i_check]
K1_chk, K2_chk = S0, S0  # 100% of S0, matching the grid's center

literal = sequential_triple_integral(S_chk, K1_chk, K2_chk, K3, T1_chk, T2_chk, T3_chk, sig_chk, MU)

d1_chk = d_barrier(S_chk, K1_chk, T1_chk, sig_chk, MU)
d2_chk = d_barrier(S_chk, K2_chk, T2_chk, sig_chk, MU)
d3_chk = d_barrier(S_chk, K3, T3_chk, sig_chk, MU)
R_chk = np.sqrt(np.minimum.outer([T1_chk, T2_chk, T3_chk], [T1_chk, T2_chk, T3_chk]) /
                np.maximum.outer([T1_chk, T2_chk, T3_chk], [T1_chk, T2_chk, T3_chk]))
closed_form = multivariate_normal(mean=[0, 0, 0], cov=R_chk).cdf([d1_chk, d2_chk, d3_chk])

print(f"\nValidating the literal sequential triple integral against the closed-form Phi3")
print(f"(at {dates[i_check].date()}, K1=K2=K3=100% of S0):")
print(f"  Sequential triple integral (numerical): {literal:.6f}")
print(f"  Closed-form Phi3 (joint trivariate CDF): {closed_form:.6f}")
print(f"  Relative difference: {abs(literal - closed_form) / closed_form:.4%}  "
      f"({'PASS' if abs(literal - closed_form) / closed_form < 0.01 else 'FAIL - investigate'})")

# --- Animated (K1, K2) surface, K3 fixed, using the closed-form Phi3 for speed ---
K1_mesh, K2_mesh = np.meshgrid(K_grid, K_grid)


def p3_surface(S_t, T1, T2, T3, sigma_t):
    d1 = d_barrier(S_t, K1_mesh, T1, sigma_t, MU)
    d2 = d_barrier(S_t, K2_mesh, T2, sigma_t, MU)
    d3 = d_barrier(S_t, K3, T3, sigma_t, MU)
    R = np.sqrt(np.minimum.outer([T1, T2, T3], [T1, T2, T3]) / np.maximum.outer([T1, T2, T3], [T1, T2, T3]))
    rv = multivariate_normal(mean=[0, 0, 0], cov=R)

    Z = np.empty_like(d1)
    it = np.nditer(d1, flags=["multi_index"])
    for _ in it:
        i, j = it.multi_index
        Z[i, j] = rv.cdf([d1[i, j], d2[i, j], d3])
    return Z


print(f"\nPrecomputing the (K1,K2) probability surface for {len(dates)} frames "
      f"({len(K_grid)}x{len(K_grid)} grid each - this is the slow step)...")
Z_frames = [p3_surface(S_t_series[i], T1_years[i], T2_years[i], T3_years[i], sigma_t_series[i])
            for i in range(len(dates))]
print("Done precomputing.")

fig, ax = plt.subplots(figsize=(9, 8))
mesh = ax.pcolormesh(K1_mesh / S0 * 100, K2_mesh / S0 * 100, Z_frames[0], cmap="viridis",
                      vmin=0, vmax=max(z.max() for z in Z_frames), shading="auto")
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("P(S_t1<=K1, S_t2<=K2, S_t3<=K3)")
ax.set_xlabel("K1 (% of S0)")
ax.set_ylabel("K2 (% of S0)")
title = ax.set_title("")


def update(frame):
    mesh.set_array(Z_frames[frame].ravel())
    S_t, T1, T2, T3, sigma_t, date = (S_t_series[frame], T1_years[frame], T2_years[frame],
                                       T3_years[frame], sigma_t_series[frame], dates[frame])
    title.set_text(
        f"{date.date()}  |  S_t = {S_t:,.0f}   sigma_t = {sigma_t:.1%}   "
        f"T1={T1*365:.0f}d T2={T2*365:.0f}d T3={T3*365:.0f}d\n"
        f"P3(t) surface over (K1,K2), K3 fixed at {K3_PCT:.0%} of S0"
    )
    return mesh, title


anim = animation.FuncAnimation(fig, update, frames=len(dates), interval=80, blit=False)

OUTPUT_GIF = __file__ + ".gif"
print(f"Rendering {len(dates)} frames to {OUTPUT_GIF} ...")
anim.save(OUTPUT_GIF, writer=animation.PillowWriter(fps=12))
print("Done.")

plt.show()
