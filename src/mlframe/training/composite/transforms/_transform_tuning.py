"""Numeric defaults shared by the composite transforms and their sibling modules.

Leaf module: no imports from this package. Siblings read these at top level, and the facade re-exports them.
"""

# Multiplier for MAD-soft-cap on T_hat (logratio in particular).
_MAD_SOFT_CAP_K: float = 10.0
# ----------------------------------------------------------------------
# Parent-resident threshold constants referenced as default-arg values
# by sibling-modules at their top-level (linear / nonlinear). They MUST
# stay in the parent because the siblings import them at module load
# time. The Pack-N additions (multi/grouped/robust gates) live here for
# the same reason.
# ----------------------------------------------------------------------

_LINRES_ROBUST_MAD_K: float = 3.0
"""Hard threshold: rows with |residual| > k * sigma_MAD are dropped before the second-pass OLS. k=3 keeps ~99.7% of inliers under Normality and drops outliers heavier than 3-sigma."""

_LINRES_ROBUST_MIN_KEEP_FRAC: float = 0.5
"""Safety: if the MAD-trim would drop more than (1 - this) of rows, fall back to plain OLS. Protects against pathological all-outlier targets where MAD is too small and trimming becomes destructive."""

_THEILSEN_MAX_PAIRS: int = 1_000_000
"""Cap on the number of (i, j) point pairs used by the Theil-Sen slope estimate. Theil-Sen is O(n^2) in the number of pairs; above this cap a random subsample of pairs is drawn (seeded for determinism) so the fit stays bounded on large-n targets while keeping the breakdown-point robustness."""

# Condition-number gate above which joint OLS is considered unstable
# and the multi-base transform falls back to zero-alpha + intercept.
# 30 is the conventional threshold (Belsley/Kuh/Welsch); above that,
# multicollinearity inflates standard errors enough that the alpha
# estimates carry no useful information. Exposed as module-level so
# tests can monkey-patch without recompiling.
_MULTI_BASE_COND_NUMBER_MAX: float = 30.0

# Minimum rows per group below which the group falls back to the global
# fit. Belsley/Kuh/Welsch rule of thumb is ">= 10 x predictors" for stable
# OLS; for 1-base + intercept that's 20-30. We use 30 as a safe default;
# exposed as module-level so tests can monkey-patch.
_GROUPED_MIN_GROUP_SIZE: int = 30

_QUANTILE_RESIDUAL_DEFAULT_N_BINS: int = 10
_QUANTILE_RESIDUAL_DEFAULT_MIN_BIN_N: int = 50

_MONOTONIC_RESIDUAL_DEFAULT_N_KNOTS: int = 12
_MONOTONIC_RESIDUAL_DEFAULT_MIN_KNOT_N: int = 30

# Pack D 2026-05-18: degeneracy threshold for the fitted PCHIP g(base).
# When ``(knots_y.max() - knots_y.min()) / std(y_train) < this``, the
# spline is essentially a constant -- the transform T = y - g(base)
# collapses to ``T = y - const`` (identity up to a global shift) and
# downstream models on T produce SAME predictions as on raw y. 0.05
# = the spline must capture at least 5% of y's std to count as non-
# trivial; below that the discovery loop must reject the spec.
_MONOTONIC_DEGENERACY_RATIO: float = 0.05

_EWMA_RESIDUAL_DEFAULT_K: int = 7  # half-life-like span; alpha = 2 / (k + 1) ~= 0.25
_FRAC_DIFF_DEFAULT_D: float = 0.5  # Lopez de Prado standard order
_FRAC_DIFF_DEFAULT_LAGS: int = 30  # maximum weight tail used in the truncated series
