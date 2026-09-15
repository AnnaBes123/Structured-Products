# Graphs (Maths)

Standalone maths exercises behind the barrier/autocall probability logic used elsewhere in
`Product MtM/` — not historical approximations of a specific product, but the underlying multivariate-normal
machinery (correlation between observation dates on one Brownian path, `Phi2`/`Phi3` closed
forms, Monte Carlo cross-checks) worked out and visualized on its own. Two static charts, three
animated ones, all self-contained scripts (no cross-imports between them, matching the rest of
this repo).

## `Autocall Density` — P(autocall at t=3), analytic surface

**What it's asking**: for a single underlying observed at three dates (t=1, t=2, t=3), what's the
probability it *first* crosses above its barrier exactly at t=3 — i.e. it stayed at/below its
barrier at t=1 **and** t=2, then broke above it at t=3?

```
P(autocall at t=3) = Phi2(k1, k2; rho12) - Phi3(k1, k2, k3; R)
```

`R` is the correlation matrix between the underlying's value at t=1/t=2/t=3 on one Brownian path
(`rho_ij = sqrt(min(ti,tj)/max(ti,tj))`), `k3` is held fixed at 0.0 (barrier = starting level, in
standardized units), and the surface sweeps `k1`, `k2` from -3 to +3.

**Reading the chart**: X/Y axes are the barrier levels k1 and k2 (standardized - 0 means "at the
starting level", positive means "above it"); the Z axis (and color) is the resulting probability.
The surface rises toward the back-right corner (permissive k1, k2 → the note almost never calls
early, so nearly all its eventual "cross the barrier" probability lands at t=3) and falls to
~0 toward the front-left (a low k1 or k2 means it's already almost certainly called earlier, so
there's nothing left over for t=3).

## `Autocall Monte Carlo` — simulation cross-check

Simulates 100,000 correlated paths `(X1, X2, X3)` from the same `R`, classifies each one with a
plain `if/elif/else` (called at t=1 / t=2 / t=3 / never), and scatter-plots a 10,000-point
subsample in 3D, colored by outcome (red / cyan / chartreuse / black). The title prints both the
Monte Carlo estimate and the analytic `Phi2 - Phi3` value for the same point side by side — they
should agree to within ordinary sampling noise. This is the "does the closed form actually match
reality" check for the formula used in `Autocall Density`.

**Reading the chart**: each axis is one date's standardized outcome (X1, X2, X3); each dot is one
simulated path. A green (chartreuse) dot sitting in the region X1≤0, X2≤0, X3>0 is exactly a path
that "called at t=3" - count what fraction of all dots are that color and it should land close to
the printed MC/Analytic probabilities.

## The three `Evolving Probability` animations

These extend the same idea in a different direction: instead of one static snapshot, they walk
day-by-day through **real 2025 S&P 500 data** (spot + VIX/VIX3M/VIX6M implied vol, Jan 2 – Jun 27
2025) and show how a barrier probability actually evolves as the market moves and time-to-observation
shrinks. Observation dates: t1 = 2025-06-30 (Q2), t2 = 2025-09-30 (Q3), t3 = 2025-12-31 (Q4) — a
1-quarter lockout, matching the rest of this repo's autocall convention. The animation stops the
day before t1, since the underlying formulas need `T1 = t1 - t > 0` to stay well-defined.

Shared setup across all three: `mu = 4%` (risk-free drift), `sigma_t` = that day's real implied
vol interpolated to `T1`, and under GBM `ln(S_ti) | S_t ~ Normal(ln(S_t) + v·Ti, sigma_t²·Ti)`
with `v = mu - 0.5·sigma_t²`.

### `Evolving Probability (1D)` — one barrier, one density curve

Shows the actual lognormal density of `S_t1` (the index level at the Q2 observation) reshaping
day by day, with the region `S_t1 ≤ K1` (K1 = 100% of the Jan-2 entry level) shaded red - **the
shaded area is P1(t), read directly off the plot.**

**Reading it**: the black dashed vertical line is today's actual spot; the blue dotted line is
the fixed barrier K1. As the animation plays, watch the bell curve narrow (variance shrinks as
`T1 → 0`) and drift with the real market - in the 2025 window it moves right (index rallies) so
the shaded probability collapses from ~21% (Jan 2) toward 0% (late June) as spot pulls decisively
clear of the barrier. Title shows the exact P1 value, spot, vol, and days remaining every frame.

### `Evolving Probability (2D)` — two barriers, a joint density contour

Extends to `(S_t1, S_t2)` jointly - since both come from the same Brownian path, they're
correlated (`rho12 = sqrt(T1/T2)`), so the bivariate lognormal density is an **elongated diagonal
blob**, not a circular one. `P2(t) = P(S_t1≤K1, S_t2≤K2) = Phi2(d1, d2; rho12)` is the probability
mass inside the dotted box (bottom-left quadrant formed by the K1/K2 dotted lines).

**Reading it**: the black dot is today's actual spot (plotted at (S_t, S_t) as a reference point);
the red contour is the joint density. Watch the ellipse both **tighten** (both legs' uncertainty
shrinks as time passes) and **swing** along the diagonal as spot moves - the diagonal elongation
itself is the visual signature of the correlation between the two dates. Title shows P2(t) exactly.

### `Evolving Probability (3D)` — a single smooth density surface, colored by the third leg

All three barriers are fixed at 100% of S0 (`K1=K2=K3`), same as K1 in the 1D file and K1/K2 in
the 2D file. Several representations were tried and dropped before this one: a 3D scatter cloud
(matplotlib's 3D projection loses depth cues, overlapping points blend together), a bulging
surface over hypothetical `(K1,K2)` choices (doesn't match the maths once all three barriers are
fixed constants, not free axes), and a parallel coordinates plot (correct, but too dense to read
cleanly). This is a genuinely continuous **surface — "like water"**, not points or lines:

- **Height** = the actual bivariate lognormal joint density of `(S_t1, S_t2)` — the same object
  as the 2D file's contour, now rendered as a real 3D surface instead of a flat one.
- **Color** = `P(S_t3 ≤ K3 | S_t2)` — the closed-form probability that the *third* leg also
  clears its barrier, given where the surface currently sits. This is exact, not simulated or
  approximated — and by the **Markov property** of Brownian motion, it depends *only* on `S_t2`,
  not on `S_t1` at all (the future increment from t2 to t3 doesn't care how the path got to t2).
  That's visible directly in the picture: the color transition (matplotlib's `magma` — dark
  purple/black = unlikely to clear the t3 barrier, bright yellow = likely to clear it) runs in
  bands that are roughly constant along the `S_t1` axis and vary only along `S_t2` — the Markov
  property, made visible rather than asserted.

**Reading it**: the dashed black lines mark the `S_t1≤K1, S_t2≤K2` quadrant — `P3(t)`, printed in
the title, is the density-times-color volume integrated over *only* that quadrant (the surface
still shows the whole domain for context, same convention as the 1D/2D companions shading only
part of a fully-drawn curve/contour). Watch the bump narrow and its color band sharpen from dark
to bright as the animation advances toward t1 — tighter time-to-observation means both less
positional uncertainty (taller, narrower bump) and a sharper verdict on whether the third leg
will clear (a crisper color transition).

**Triple integral, as requested, validated two independent ways**:
1. `sequential_triple_integral()` implements the literal `∫∫∫ φ1(y1|St)·φ2(y2|y1)·φ3(y3|y2) dy1
   dy2 dy3` (innermost step closed-form, the other two genuine numerical quadrature), checked
   once against the closed-form `Phi3` (matched to 0.0000%–0.0004% in testing).
2. **Law of total probability**, checked on the *exact same grid the file renders*: summing
   `density(S1,S2) × P(S_t3≤K3|S_t2)` over the `S_t1≤K1, S_t2≤K2` quadrant is literally the outer
   two integrals of the same triple integral, done via the rendered grid instead of `scipy.quad`
   — an independent numerical proof that the picture on screen is mathematically consistent with
   the number in the title (matched to ~0.26%, essentially all grid-discretization noise).

## A note on log prices vs. price-space axes

Every formula above is computed in **log-price space internally** (that's what GBM/`Phi`
requires) - the price-space vs. log-price question is purely about how the *axes* are displayed.
Barriers here are shown as "% of S0" (K1/K2/K3), matching how every other product in this repo
(and every real term sheet) quotes a barrier - never as a log-difference. The 1D/2D density
shapes are correctly transformed back from log-space via the appropriate Jacobian (`f_S(s) =
f_Y(ln s)/s`), so what you see is the true (skewed) lognormal shape of the actual index level,
not an approximation.

## Usage

```
pip install -r ../../requirements.txt
python3 "Autocall Density"
python3 "Autocall Monte Carlo"
python3 "Evolving Probability (1D)"
python3 "Evolving Probability (2D)"
python3 "Evolving Probability (3D)"
```

The three `Evolving Probability` scripts fetch real 2025 data via `yfinance` (falling back to
FRED for the S&P 500 series only - FRED has no VIX3M/VIX6M equivalent, so the vol fallback would
need a different source if yfinance is unavailable), precompute every frame, then save an
animated `.gif` next to the script (same name + `.gif`) via `matplotlib.animation.PillowWriter`.
The 3D file's precompute step is the slowest (~120 frames x a 22x22 grid of 3D Gaussian CDF
evaluations) - expect a couple of minutes.
