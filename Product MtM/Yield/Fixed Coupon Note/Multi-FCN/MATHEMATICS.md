# The Maths, Fully Decomposed

The mechanical companion to `README.md`. Builds directly on `../MATHEMATICS.md` (single-name
Autocallable FCN) and `../../Reverse Convertible/Multi-RC/MATHEMATICS.md` (worst-of basket
mechanics) - this file only covers what's genuinely new when the two are combined.

Notation: `n` names, each tracked as **relative performance** `x_i(t) = S_i(t)/S0_i` (1.0 at
inception - there's no single dollar `S0` to anchor multiple names to, so `Principal = 1.0` "par
units" throughout). `ρ` = the `n×n` correlation matrix estimated from trailing real returns.

---

## 1. Why no closed form, and no QuantLib basket engine

The single-name Autocallable FCN's closed-form check needs a joint multivariate normal over
**time** for one asset (`n_obs` dimensions). Extending that to a worst-of basket needs a joint
distribution over **time and assets at once** (`n_assets × n_obs` dimensions - 12 for this file's
default 3 names × 4 dates), which pushes `scipy.stats.multivariate_normal.cdf` into territory
where it falls back to its own internal Monte Carlo estimate - slower and noisier than just
simulating the GBM paths directly. QuantLib also has no basket-autocallable engine. So the live
pricer is a hand-rolled correlated multi-step Monte Carlo (below), generalizing the single-name
engine with an added asset axis.

## 2. Monte Carlo engine: correlated multi-asset, multi-date

Draw independent standard normals `Z_indep`, shape `(n_paths, n_steps, n_assets)`, and correlate
**across assets within each time step** via the Cholesky factor `L` of `ρ` (`ρ = L·Lᵀ`):

```
Z_corr[:, t, :] = Z_indep[:, t, :] @ Lᵀ
```

Each asset's log-increment at step `t` uses its own drift and vol:

```
Δlog x_i(t) = (r - q_i - σ_i²/2)·Δt_t + σ_i·√Δt_t · Z_corr[:, t, i]
```

At every grid date, `worst_of(t) = min_i x_i(t)`. The first date (if any) where `worst_of(t) ≥
TRIGGER` is that path's `settle_time`; otherwise `settle_time = T`. Per path:

```
principal_pv = 1.0 · e^(-(r + c)·settle_time)
put_pv       = -(1/Strike)·max(Strike - worst_of(T), 0) · e^(-r·T)   if never triggered, else 0
price        = mean(principal_pv + put_pv)
```

Identical structure to the single-name engine (`../MATHEMATICS.md` section 5.2), with `S_T/S0`
replaced by `worst_of(T)` and an added Cholesky step for the cross-asset correlation.

## 3. Verification: two nested reductions

Both push the autocall trigger to an unreachable level (no path can ever call) and check the
engine collapses onto an independently-computed benchmark:

1. **N=1 reduction** - a single synthetic name (correlation matrix `[[1.0]]`) must match that
   name's plain closed-form FCN price, `ZCB(1.0) - (1/Strike)·Put` (QuantLib
   `AnalyticEuropeanEngine`). Validates the time-stepping/discounting machinery alone (the
   worst-of-across-assets reduction is trivial with one asset).
2. **N=3 reduction** - the real 3-name basket must match Multi-RC's own terminal-only worst-of
   price, `ZCB(1.0) - (1/Strike)·worst_of_put_price` (QuantLib's `MCEuropeanBasketEngine`, the
   same pricer Multi-RC itself uses). Validates the worst-of-across-assets reduction specifically,
   now happening *inside* the time-stepped engine rather than at a single terminal date.

Both print a relative difference and a PASS/FAIL at runtime (observed ~0.004% and ~0.09%
respectively) before the actual autocall-enabled price is trusted.

## 4. Dispersion cuts both ways

Multi-RC's own math (`../../Reverse Convertible/Multi-RC/MATHEMATICS.md`) shows dispersion (low
pairwise correlation) makes the **downside** worse for a worst-of put - more independent chances
for one name to breach the strike. The autocall condition here is `worst_of ≥ TRIGGER`, i.e.
*every* name simultaneously at/above it - the same dispersion that hurts the downside also makes
this **upside** condition harder to satisfy, so a worst-of autocallable calls less often than any
single-name autocallable at the same trigger. Both effects share the same root cause and the same
sign for the investor; the script's own printed "for comparison, no-autocall" price is the
computed check that the autocall feature is still worth more than not having it (early-redemption
optionality can only help), not an assumed conclusion.
