# =============================================================================
# P(Breach) Estimator.R
# =============================================================================
# R port of "P(Breach) Estimator.py" (this file's companion, same folder).
# Same real-world (physical-measure) breach-probability question, the same
# STRIKE-relative-to-entry rebasing, the same calendar-to-trading-day horizon
# conversion, and the same walk-forward-safe naive baseline as the Python
# version - see README.md / MATHEMATICS.md, which document both.
#
# The entire reason this port exists: the BASKET case here uses a genuine
# DCC-GARCH with a fitted MULTIVARIATE Student's t (rmgarch), not Python's
# hand-rolled CCC-GARCH + Gaussian-copula-onto-univariate-t approximation.
# Python's `arch` package has no multivariate GARCH; R's rugarch/rmgarch do.
#
# Scope differences from the Python version:
#  - No Markov-switching candidates. Python has this off by default anyway
#    (INCLUDE_MARKOV_SWITCHING = False there); R's MSGARCH package (already
#    installed alongside rugarch/rmgarch on this machine) would be the
#    natural way to add genuine Markov-switching GARCH later, if wanted.
#  - No charts - console output only, per request.
#  - Because rmgarch's DCC framework fits each name's own univariate model
#    independently (unlike Python's hand-rolled one-step basket recursion,
#    which only understands one variance-recursion shape), EVERY basket name
#    gets the FULL 6-spec volatility grid here, not just the 3 that were
#    basket-safe in Python (see BASKET_INCOMPATIBLE_VOL_SPECS there).
#  - Walk-forward validation is single-asset only, the same scope cut Python
#    makes (a joint, DCC-level walk-forward validation is a natural but
#    unbuilt extension in both versions).

suppressMessages({
  library(rugarch)
  library(rmgarch)
  library(quantmod)
  library(xts)
})

# --- Product terms (same convention as P(Breach) Estimator.py) ---
STRIKE <- 0.5                  # breach level, as a fraction of each name's own entry level
ENTRY_DATE <- as.Date("2026-09-17")
TENOR <- 1                      # years

# 1 ticker = single-asset mode. 2+ = basket (worst-of) mode via DCC-GARCH.
TICKERS <- c("CL=F", "BZ=F")

N_SIMULATIONS <- 10000           # Monte Carlo paths per rolling origin date
RNG_SEED <- 42
set.seed(RNG_SEED)

# Same reasoning as MAX_SAFE_SIMULATIONS in the Python version: a basket sim
# allocates an (n.sim x m.sim x n_tickers)-ish state at every rolling date,
# so it's capped harder than single-asset.
MAX_SAFE_SIMULATIONS <- 20000

safe_n_sims <- function(requested, basket_mode = FALSE) {
  cap <- if (basket_mode) MAX_SAFE_SIMULATIONS %/% 4 else MAX_SAFE_SIMULATIONS
  if (requested > cap) {
    cat(sprintf("    [n_sims clamped from %s to %s - see MAX_SAFE_SIMULATIONS]\n",
                format(requested, big.mark=","), format(cap, big.mark=",")))
    return(cap)
  }
  requested
}

TRADING_DAYS_PER_YEAR <- 252
CALENDAR_DAYS_PER_YEAR <- 365.25

calendar_to_trading_days <- function(calendar_days) {
  # Both the rolling and walk-forward simulations advance one FITTED
  # trading-day return per simulated step, so a horizon expressed in
  # calendar days (as maturity/entry/target dates naturally are) would
  # simulate ~365/252 too many daily shocks - this converts to the
  # equivalent trading-day count instead.
  max(1, round(calendar_days * TRADING_DAYS_PER_YEAR / CALENDAR_DAYS_PER_YEAR))
}

# Walk-forward validation / "self-learning" - see README for the full design.
FULL_HISTORY_YEARS <- 15
VALIDATION_YEARS <- 3
REFIT_FREQUENCY_DAYS <- 252
EVAL_FREQUENCY_DAYS <- 42

# -----------------------------------------------------------------------
# Data fetching
# -----------------------------------------------------------------------

fetch_daily_closes <- function(ticker, from, to, fred_series = NULL) {
  cl <- tryCatch({
    x <- suppressWarnings(getSymbols(ticker, src = "yahoo", from = from, to = to,
                                      auto.assign = FALSE))
    na.omit(Cl(x))
  }, error = function(e) NULL, warning = function(w) NULL)

  if ((is.null(cl) || nrow(cl) == 0) && !is.null(fred_series)) {
    cl <- tryCatch({
      x <- suppressWarnings(getSymbols(fred_series, src = "FRED", from = from, to = to,
                                        auto.assign = FALSE))
      na.omit(x)
    }, error = function(e) NULL)
  }

  if (is.null(cl) || nrow(cl) == 0) {
    stop(sprintf("No data returned for %s", ticker))
  }
  colnames(cl) <- ticker
  cl
}

fetch_multi_asset_path <- function(tickers, from, to) {
  series_list <- lapply(tickers, function(t) {
    fred <- if (t == "^GSPC") "SP500" else NULL
    fetch_daily_closes(t, from, to, fred_series = fred)
  })
  merged <- do.call(merge, series_list)
  colnames(merged) <- tickers
  na.omit(merged)
}

log_returns_pct <- function(price_xts) {
  # Log returns in percent - matches the Python version's scaling (numerical
  # stability for the optimizer), and keeps both files' formulas comparable.
  na.omit(diff(log(price_xts)) * 100)
}

# -----------------------------------------------------------------------
# Model selection: grid of ARMA-mean x GARCH-family-variance candidates,
# picked by BIC (AIC also reported) - see README for the AIC/BIC/SBIC and
# GJR-vs-TGARCH notes, which apply identically here.
# -----------------------------------------------------------------------

GARCH_MEAN_SPECS <- list(
  list(name = "Zero",     armaOrder = c(0, 0), include.mean = FALSE),
  list(name = "Constant", armaOrder = c(0, 0), include.mean = TRUE),
  list(name = "AR(1)",    armaOrder = c(1, 0), include.mean = TRUE)
)

GARCH_VOL_SPECS <- list(
  list(name = "ARCH(1)",          model = "sGARCH",   garchOrder = c(1, 0)),
  list(name = "GARCH(1,1)",       model = "sGARCH",   garchOrder = c(1, 1)),
  list(name = "GJR-GARCH(1,1,1)", model = "gjrGARCH",  garchOrder = c(1, 1)),
  list(name = "EGARCH(1,1,1)",    model = "eGARCH",   garchOrder = c(1, 1)),
  list(name = "APARCH(1,1,1)",    model = "apARCH",   garchOrder = c(1, 1)),
  list(name = "FIGARCH(1,1)",     model = "fiGARCH",  garchOrder = c(1, 1))
)

build_uspec <- function(mean_spec, vol_spec, fixed.pars = list()) {
  ugarchspec(
    variance.model = list(model = vol_spec$model, garchOrder = vol_spec$garchOrder),
    mean.model = list(armaOrder = mean_spec$armaOrder, include.mean = mean_spec$include.mean),
    distribution.model = "std",       # Student's t, fitted shape - same as Python's dist="t"
    fixed.pars = fixed.pars
  )
}

fit_garch_family_candidates <- function(returns_pct, fit_end) {
  # Estimation uses only data through fit_end - no look-ahead.
  data_fit <- returns_pct[paste0("/", fit_end)]
  candidates <- list()
  for (mean_spec in GARCH_MEAN_SPECS) {
    for (vol_spec in GARCH_VOL_SPECS) {
      label <- sprintf("%s + %s (t)", mean_spec$name, vol_spec$name)
      spec <- build_uspec(mean_spec, vol_spec)
      fit <- tryCatch(
        ugarchfit(spec, data = data_fit, solver = "hybrid",
                  fit.control = list(stationarity = 1)),
        error = function(e) NULL
      )
      if (is.null(fit) || fit@fit$convergence != 0) {
        cat(sprintf("    [skip] %s: failed to fit\n", label))
        next
      }
      ic <- infocriteria(fit)
      candidates[[length(candidates) + 1]] <- list(
        label = label, mean_spec = mean_spec, vol_spec = vol_spec, fit = fit,
        aic = ic["Akaike", 1], bic = ic["Bayes", 1], loglik = as.numeric(likelihood(fit)),
        nparams = length(coef(fit))
      )
    }
  }
  candidates
}

select_best_model <- function(candidates, underlying_label, verbose = TRUE) {
  bics <- sapply(candidates, function(c) c$bic)
  aics <- sapply(candidates, function(c) c$aic)
  best_idx <- which.min(bics)
  best_aic_idx <- which.min(aics)

  if (verbose) {
    tbl <- data.frame(
      Model = sapply(candidates, function(c) c$label),
      Params = sapply(candidates, function(c) c$nparams),
      LogLik = sapply(candidates, function(c) c$loglik),
      AIC = aics, BIC = bics
    )
    tbl <- tbl[order(tbl$BIC), ]
    cat(sprintf("\nModel comparison for %s (sorted by BIC, per-observation - lower is better):\n",
                underlying_label))
    print(tbl, row.names = FALSE)
    if (best_idx != best_aic_idx) {
      cat(sprintf("  Note: AIC would have picked a different model ('%s') than BIC ('%s').\n",
                  candidates[[best_aic_idx]]$label, candidates[[best_idx]]$label))
    } else {
      cat(sprintf("  AIC and BIC agree: '%s'\n", candidates[[best_idx]]$label))
    }
  }
  candidates[[best_idx]]
}

# -----------------------------------------------------------------------
# SINGLE-ASSET forward simulation
# -----------------------------------------------------------------------

simulate_garch_forward <- function(best, returns_pct_full, origin_date, horizon_trading_days,
                                    n_sims = N_SIMULATIONS, seed = RNG_SEED) {
  # Same truncate-then-fix-then-simulate design as the Python version's
  # simulate_garch_forward: rebuild the spec with the ALREADY-FITTED
  # parameters (fixed.pars, no re-estimation), then filter it through real
  # returns up to and including origin_date to get the correct starting
  # conditional variance/residual state, then simulate forward from there.
  spec_fixed <- build_uspec(best$mean_spec, best$vol_spec, fixed.pars = as.list(coef(best$fit)))
  data_through_origin <- returns_pct_full[paste0("/", origin_date)]
  filt <- ugarchfilter(spec_fixed, data = data_through_origin)

  pre_sigma <- tail(as.numeric(sigma(filt)), 1)
  pre_resid <- tail(as.numeric(residuals(filt)), 1)
  pre_return <- tail(as.numeric(data_through_origin), 1)

  sim <- ugarchpath(spec_fixed, n.sim = horizon_trading_days, m.sim = n_sims,
                     presigma = pre_sigma, prereturns = pre_return, preresiduals = pre_resid,
                     rseed = seed)
  sim_returns_pct <- fitted(sim)                       # (horizon_trading_days x n_sims)
  cum_path_pct <- apply(sim_returns_pct, 2, cumsum)     # cumulative log-return path, per sim
  terminal_relative <- exp(cum_path_pct[nrow(cum_path_pct), ] / 100)   # S_H / S_origin
  path_min_relative <- exp(apply(cum_path_pct, 2, min) / 100)          # min over the path
  list(terminal = terminal_relative, path_min = path_min_relative)
}

# -----------------------------------------------------------------------
# BASKET mode: genuine DCC-GARCH (Engle 2002) with a fitted multivariate
# Student's t (rmgarch) - each name keeps its own independently best-fit
# univariate model (any of the 6 volatility specs; rmgarch's DCC framework
# doesn't care what each one's own recursion looks like, unlike Python's
# hand-rolled one-step basket recursion), and a genuinely time-varying,
# jointly-estimated correlation ties them together.
# -----------------------------------------------------------------------

fit_dcc_basket <- function(best_models, tickers, returns_pct_df, fit_end) {
  uspecs <- lapply(tickers, function(t) build_uspec(best_models[[t]]$mean_spec,
                                                      best_models[[t]]$vol_spec))
  mspec <- multispec(uspecs)
  data_fit <- returns_pct_df[paste0("/", fit_end), tickers]
  dspec <- dccspec(uspec = mspec, dccOrder = c(1, 1), distribution = "mvt")
  dccfit(dspec, data = data_fit, fit.control = list(eval.se = FALSE))
}

# Splits a dccfit's flat coef() vector into per-ticker univariate param lists
# plus the joint DCC-level params - dccspec's fixed.pars mechanism requires
# each piece supplied separately (see rmgarch's dccsim-methods docs).
.split_dcc_coefs <- function(dfit, tickers) {
  allc <- coef(dfit)
  uni_pars <- lapply(tickers, function(t) {
    prefix <- paste0("[", t, "].")
    idx <- startsWith(names(allc), prefix)
    vals <- allc[idx]
    names(vals) <- substring(names(vals), nchar(prefix) + 1)
    as.list(vals)
  })
  names(uni_pars) <- tickers
  joint_names <- names(allc)[startsWith(names(allc), "[Joint]")]
  joint_pars <- as.list(setNames(allc[joint_names], sub("^\\[Joint\\]", "", joint_names)))
  list(uni = uni_pars, joint = joint_pars)
}

simulate_dcc_forward <- function(dfit, best_models, tickers, returns_pct_df, origin_date,
                                  horizon_trading_days, n_sims = N_SIMULATIONS, seed = RNG_SEED) {
  coefs <- .split_dcc_coefs(dfit, tickers)
  uspecs_fixed <- lapply(tickers, function(t) build_uspec(best_models[[t]]$mean_spec,
                                                            best_models[[t]]$vol_spec,
                                                            fixed.pars = coefs$uni[[t]]))
  mspec_fixed <- multispec(uspecs_fixed)
  dspec_fixed <- dccspec(uspec = mspec_fixed, dccOrder = c(1, 1), distribution = "mvt",
                          fixed.pars = coefs$joint)

  data_through_origin <- returns_pct_df[paste0("/", origin_date), tickers]
  filt <- dccfilter(dspec_fixed, data = data_through_origin)

  n <- length(tickers)
  pre_sigma <- matrix(tail(sigma(filt), 1), nrow = 1)
  pre_resid <- matrix(tail(residuals(filt), 1), nrow = 1)
  pre_return <- matrix(tail(as.matrix(data_through_origin), 1), nrow = 1)
  preQ <- filt@mfilter$Q[[length(filt@mfilter$Q)]]
  preZ <- matrix(tail(filt@mfilter$stdresid, 1), nrow = 1)

  set.seed(seed)
  sim <- dccsim(dspec_fixed, n.sim = horizon_trading_days, m.sim = n_sims, startMethod = "sample",
                presigma = pre_sigma, prereturns = pre_return, preresiduals = pre_resid,
                preQ = preQ, preZ = preZ, Qbar = filt@mfilter$Qbar,
                rseed = seed + seq_len(n_sims))

  terminal <- matrix(NA_real_, n_sims, n)
  path_min <- matrix(NA_real_, n_sims, n)
  for (s in seq_len(n_sims)) {
    sim_returns_pct <- fitted(sim, sim = s)             # (horizon_trading_days x n_names)
    cum_path_pct <- apply(sim_returns_pct, 2, cumsum)
    terminal[s, ] <- exp(cum_path_pct[nrow(cum_path_pct), ] / 100)
    path_min[s, ] <- exp(apply(cum_path_pct, 2, min) / 100)
  }
  colnames(terminal) <- tickers
  colnames(path_min) <- tickers
  list(terminal = terminal, path_min = path_min)
}

# -----------------------------------------------------------------------
# Rolling probability-of-breach: at each historical date, simulate forward
# to the fixed maturity date using only information available up to that
# date, and read off the empirical fraction of paths that breach - both at
# maturity and ever (touch) along the path.
# -----------------------------------------------------------------------

rolling_breach_probabilities <- function(price_df, S0, best_models, returns_pct_df, fit_end,
                                          maturity_date, dcc_fit = NULL, n_sims = N_SIMULATIONS) {
  tickers <- colnames(price_df)
  basket_mode <- length(tickers) > 1
  n_sims <- safe_n_sims(n_sims, basket_mode)
  dates <- index(price_df)
  S0 <- setNames(as.numeric(S0), tickers)

  terminal_probs <- numeric(length(dates))
  touch_probs <- numeric(length(dates))

  for (i in seq_along(dates)) {
    date <- dates[i]
    # STRIKE is fixed relative to S0 (entry), but every simulation below
    # produces returns relative to TODAY's spot at `date`. relative_today
    # re-bases the simulated outcome back onto S0, so a name that has
    # already moved is judged against its REMAINING distance to STRIKE, not
    # a fresh full STRIKE move measured from today (see README/MATHEMATICS).
    relative_today <- as.numeric(price_df[date, ]) / S0
    names(relative_today) <- tickers

    horizon_calendar <- as.numeric(maturity_date - date)
    if (horizon_calendar <= 0) {
      worst_today <- min(relative_today)
      terminal_probs[i] <- if (worst_today < STRIKE) 1.0 else 0.0
      touch_probs[i] <- terminal_probs[i]
      next
    }

    horizon <- calendar_to_trading_days(horizon_calendar)

    if (!basket_mode) {
      ticker <- tickers[1]
      sim <- simulate_garch_forward(best_models[[ticker]], returns_pct_df[, ticker], date,
                                     horizon, n_sims,
                                     seed = RNG_SEED + (as.integer(date) %% 10000))
      terminal_from_entry <- sim$terminal * relative_today[ticker]
      path_min_from_entry <- sim$path_min * relative_today[ticker]
      terminal_probs[i] <- mean(terminal_from_entry < STRIKE)
      touch_probs[i] <- mean(path_min_from_entry < STRIKE)
    } else {
      sim <- simulate_dcc_forward(dcc_fit, best_models, tickers, returns_pct_df, date, horizon,
                                   n_sims, seed = RNG_SEED + (as.integer(date) %% 10000))
      terminal_from_entry <- sweep(sim$terminal, 2, relative_today[tickers], `*`)
      path_min_from_entry <- sweep(sim$path_min, 2, relative_today[tickers], `*`)
      worst_terminal <- apply(terminal_from_entry, 1, min)
      worst_path_min <- apply(path_min_from_entry, 1, min)
      terminal_probs[i] <- mean(worst_terminal < STRIKE)
      touch_probs[i] <- mean(worst_path_min < STRIKE)
    }
  }

  list(dates = dates, terminal = terminal_probs, touch = touch_probs)
}

# -----------------------------------------------------------------------
# Walk-forward validation: does the predicted probability actually mean
# anything? Single-asset only - same scope cut as the Python version (a
# joint, DCC-level walk-forward validation is a natural but unbuilt
# extension in both).
# -----------------------------------------------------------------------

naive_historical_baseline <- function(price_series, as_of_date, horizon_trading_days, strike) {
  # Walk-forward-safe naive baseline: the empirical breach frequency among
  # ALL horizon_trading_days-ahead returns observable using ONLY price
  # history up to and including as_of_date - a forecast that could actually
  # have been made AT that checkpoint, not the mean of the validation set's
  # own (still-future, as of that checkpoint) outcomes.
  history <- as.numeric(price_series[paste0("/", as_of_date)])
  n <- length(history)
  if (n <= horizon_trading_days) return(NA_real_)
  fwd_relative <- history[(horizon_trading_days + 1):n] / history[1:(n - horizon_trading_days)]
  mean(fwd_relative < strike)
}

brier_score <- function(predicted, actual) {
  mean((predicted - actual)^2, na.rm = TRUE)
}

calibration_table <- function(predicted, actual, n_buckets = 5) {
  # Same "degrade gracefully with too few distinct values" behavior as the
  # Python version's pd.qcut(..., q=min(n_buckets, n_unique), duplicates="drop"):
  # with too little walk-forward data (or many tied predictions), fall back
  # to fewer buckets - or a single bucket - rather than erroring.
  n_unique <- length(unique(predicted))
  qs <- unique(quantile(predicted, probs = seq(0, 1, length.out = min(n_buckets, n_unique) + 1)))
  bucket <- if (length(qs) < 2) {
    factor(rep(sprintf("[%.3f, %.3f]", min(predicted), max(predicted)), length(predicted)))
  } else {
    cut(predicted, breaks = qs, include.lowest = TRUE)
  }
  data.frame(
    bucket = levels(bucket),
    n = as.numeric(table(bucket)),
    mean_predicted = as.numeric(tapply(predicted, bucket, mean)),
    empirical_frequency = as.numeric(tapply(actual, bucket, mean))
  )
}

walk_forward_validate <- function(ticker, full_returns_pct, full_price_series, validation_start,
                                   validation_end, tenor_years, strike,
                                   refit_every = REFIT_FREQUENCY_DAYS, eval_every = EVAL_FREQUENCY_DAYS,
                                   n_sims = N_SIMULATIONS) {
  n_sims <- safe_n_sims(n_sims, basket_mode = FALSE)
  horizon_calendar_days <- round(tenor_years * CALENDAR_DAYS_PER_YEAR)
  horizon_trading_days <- calendar_to_trading_days(horizon_calendar_days)

  trading_dates <- index(full_price_series)
  in_window <- trading_dates >= validation_start & trading_dates <= validation_end
  checkpoints <- trading_dates[in_window][seq(1, sum(in_window), by = eval_every)]

  current_model <- NULL
  last_refit_date <- NULL
  records <- list()

  for (date in checkpoints) {
    date <- as.Date(date, origin = "1970-01-01")
    if (is.null(last_refit_date) || as.numeric(date - last_refit_date) >= refit_every / 252 * 365.25) {
      candidates <- fit_garch_family_candidates(full_returns_pct, date)
      current_model <- select_best_model(candidates, sprintf("as of %s", date), verbose = FALSE)
      last_refit_date <- date
      cat(sprintf("  [refit @ %s, %d obs] -> %s\n", date,
                  nrow(full_returns_pct[paste0("/", date)]), current_model$label))
    }

    target_date_calendar <- date + horizon_calendar_days
    future_dates <- trading_dates[trading_dates > date]
    if (length(future_dates) == 0) next
    target_date <- future_dates[which.min(abs(as.numeric(future_dates - target_date_calendar)))]
    if (target_date < target_date_calendar - 10) next  # not enough real data yet to score this checkpoint

    sim <- simulate_garch_forward(current_model, full_returns_pct, date, horizon_trading_days, n_sims)
    predicted_prob <- mean(sim$terminal < strike)
    naive_predicted <- naive_historical_baseline(full_price_series, date, horizon_trading_days, strike)

    actual_relative <- as.numeric(full_price_series[target_date]) / as.numeric(full_price_series[date])
    actual_breach <- if (actual_relative < strike) 1.0 else 0.0

    records[[length(records) + 1]] <- list(
      date = date, target_date = target_date, predicted = predicted_prob,
      naive_predicted = naive_predicted, actual = actual_breach, model = current_model$label
    )
  }

  do.call(rbind, lapply(records, as.data.frame))
}

# =============================================================================
# Main
# =============================================================================

basket_mode <- length(TICKERS) > 1
cat(sprintf("%s: %s\n", if (basket_mode) "Basket" else "Underlying", paste(TICKERS, collapse = ", ")))

entry_ts <- ENTRY_DATE
maturity_ts <- entry_ts + round(TENOR * 365.25)
history_start <- entry_ts - round(FULL_HISTORY_YEARS * 365.25)
validation_start <- entry_ts - round(VALIDATION_YEARS * 365.25)
validation_end <- entry_ts - round(TENOR * 365.25)

cat(sprintf("\nFetching %dy of history ending %s...\n", FULL_HISTORY_YEARS, ENTRY_DATE))
full_price_df <- fetch_multi_asset_path(TICKERS, history_start, maturity_ts)
full_returns_df <- log_returns_pct(full_price_df)

last_available_date <- max(index(full_price_df))
note_matured <- maturity_ts <= last_available_date
if (entry_ts > last_available_date) {
  gap_days <- as.numeric(entry_ts - last_available_date)
  if (gap_days > 10) {
    stop(sprintf("ENTRY_DATE (%s) is %d days after the last available real trading day (%s).",
                 ENTRY_DATE, gap_days, last_available_date))
  }
  cat(sprintf("\nNote: ENTRY_DATE (%s) is a weekend/holiday or newer than the most recently ",
              ENTRY_DATE))
  cat(sprintf("available close - using %s instead.\n", last_available_date))
  entry_ts <- last_available_date
  maturity_ts <- entry_ts + round(TENOR * 365.25)
  note_matured <- maturity_ts <= last_available_date
}

path_price_df <- full_price_df[paste0(entry_ts, "/", min(maturity_ts, last_available_date))]
if (nrow(path_price_df) == 0) {
  stop(sprintf("No trading days found between %s and %s.", entry_ts, min(maturity_ts, last_available_date)))
}
S0_list <- as.numeric(path_price_df[1, ])
names(S0_list) <- TICKERS

if (!note_matured) {
  cat(sprintf("\nNote: maturity (%s) is AFTER the last available real trading day (%s) - this note ",
              maturity_ts, last_available_date))
  cat("hasn't matured yet. The rolling estimate below runs through the last available date only.\n")
}

cat(sprintf("\n%s\nWALK-FORWARD VALIDATION (held out %s to %s, never overlapping the live %s forecast)\n%s\n",
            strrep("#", 70), validation_start, validation_end, ENTRY_DATE, strrep("#", 70)))

for (ticker in TICKERS) {
  cat(sprintf("\n--- %s ---\n", ticker))
  results_df <- walk_forward_validate(ticker, full_returns_df[, ticker], full_price_df[, ticker],
                                       validation_start, validation_end, TENOR, STRIKE)

  if (is.null(results_df) || nrow(results_df) == 0) {
    cat("  No out-of-sample checkpoints produced - skipping.\n")
    next
  }

  b <- brier_score(results_df$predicted, results_df$actual)
  b_naive <- brier_score(results_df$naive_predicted, results_df$actual)

  cat(sprintf("\n  %d out-of-sample (predicted, actual) pairs, %.1f%% actually breached\n",
              nrow(results_df), 100 * mean(results_df$actual)))
  cat(sprintf("  Brier score - this model:             %.4f\n", b))
  cat(sprintf("  Brier score - naive (historical rate): %.4f\n", b_naive))
  cat("  (lower is better; 0.25 = the 'always guess 50%' baseline under class balance)\n")
  cat("\n  Calibration table (predicted probability bucket vs. what actually happened):\n")
  print(calibration_table(results_df$predicted, results_df$actual), row.names = FALSE)
}

cat(sprintf("\n%s\nLIVE MODEL (fit on all data through %s)\n%s\n", strrep("#", 70), ENTRY_DATE, strrep("#", 70)))
fit_end <- entry_ts
best_models <- list()
for (ticker in TICKERS) {
  cat(sprintf("\n%s\nFitting candidate models for %s\n%s\n", strrep("=", 70), ticker, strrep("=", 70)))
  candidates <- fit_garch_family_candidates(full_returns_df[, ticker], fit_end)
  best <- select_best_model(candidates, ticker, verbose = TRUE)
  best_models[[ticker]] <- best
  cat(sprintf("  >>> Selected: %s <<<\n", best$label))
}

dcc_fit <- NULL
if (basket_mode) {
  cat(sprintf("\n%s\nFitting joint DCC-GARCH (multivariate Student's t) across the basket\n%s\n",
              strrep("=", 70), strrep("=", 70)))
  dcc_fit <- fit_dcc_basket(best_models, TICKERS, full_returns_df, fit_end)
  cat("DCC fit complete. Coefficients:\n")
  print(coef(dcc_fit))
}

cat(sprintf("\n%s\nRunning the sequential rolling Monte Carlo (%s paths each)\n%s\n",
            strrep("=", 70), format(N_SIMULATIONS, big.mark = ","), strrep("=", 70)))
result <- rolling_breach_probabilities(path_price_df, S0_list, best_models, full_returns_df,
                                        fit_end, maturity_ts, dcc_fit = dcc_fit)

cat(sprintf("\n%s\nRESULTS\n%s\n", strrep("=", 70), strrep("=", 70)))
cat(sprintf("Entry Date: %s   Maturity Date: %s   Strike: %.0f%%\n", ENTRY_DATE, maturity_ts, STRIKE * 100))
cat(sprintf("\nDay-1 estimated P(breach at maturity): %.2f%%\n", 100 * result$terminal[1]))
cat(sprintf("Day-1 estimated P(ever touches strike): %.2f%%\n", 100 * result$touch[1]))
cat(sprintf("\nMost recent rolling estimate (%s): %.2f%%\n",
            result$dates[length(result$dates)], 100 * result$terminal[length(result$terminal)]))

if (note_matured) {
  actual_relative <- as.numeric(path_price_df[nrow(path_price_df), ]) / S0_list
  actual_worst_of_T <- min(actual_relative)
  actually_breached <- actual_worst_of_T < STRIKE
  cat(sprintf("What ACTUALLY happened in this historical path: worst-of relative performance at maturity = %.2f%% (%s the %.0f%% strike)\n",
              100 * actual_worst_of_T, if (actually_breached) "BREACHED" else "did not breach", STRIKE * 100))
} else {
  cat(sprintf("Note matures %s - still %d days out, actual outcome not yet known.\n",
              maturity_ts, as.numeric(maturity_ts - last_available_date)))
}
