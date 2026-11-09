# Wave 2 audit: NUMERICS (2026-11-05)

Scope: `src/mlframe` (benchmarks excluded). Searched for raw-moment variance, unguarded divisions by std/sigma, log of unclipped values, float32 accumulation, NaN swallowed by comparisons, absolute epsilons. CHANGELOG checked: neither finding below is listed.

Sites read and found correct (no finding): `calibration/_independence_check.py:49-62` (columns centred before the moment form), `metrics/regression/_regression_metrics.py` and `_regression_extras.py` (centred/Welford ss_tot, zero-ss_tot guarded), `preprocessing/rare_count_pruning.py:43-52` (alpha-smoothed cells, `where=se>0`), `feature_selection/wrappers/_univariate_ht.py:283` (`sigma<=0` guard), `data_valuation/_knn_shapley.py:102-106` and `_training_weight_adapter.py:64-68` (constant column gives identical z, harmless), `votenrank/geometric_weight_blend.py:79`, `training/_classif_helpers.py:38`, `training/_mc_dropout.py:69` (clipped before log), `metrics/_log_loss_and_separation.py:57-67` (NaN/out-of-range rejected explicitly), `training/targets/_target_distribution_analyzer_stats.py:28-33`, `training/loss_recommendation.py:100-113` (centred moments, var/sigma guarded).

## NUM-W2-1 (Medium) - `src/mlframe/calibration/quality.py:619-641` and `:644-666`

Statement: `_anderson_darling_kernel` (and its `prange` twin) are compiled with `fastmath=True`, which includes the no-NaN flag, so a NaN PIT value is not propagated; the clamp branches may fold it into `eps` / `1-eps` and the statistic comes back finite and wrong.

Scenario: `anderson_darling_statistic` (`:679-686`) only guards `n == 0`; it sorts the input (NaN to the end) and passes the full `n`. With 1000 PIT values of which 10 are NaN (e.g. a CDF evaluated on a NaN prediction), LLVM may treat `a < eps` / `a > 1-eps` on NaN as ordinary branches and compute `log(eps) = -27.6` for each, adding roughly `-(2k+1) * 27.6 / n` per NaN term: a large, finite, plausible-looking A^2 instead of NaN. Without fastmath the result is NaN, which callers can detect. The exact folding is compiler-dependent, which is itself the problem (serial and parallel kernels may disagree).

Disposition: RESOLVED-needed. Either drop `fastmath=True` (or use `fastmath={"contract","reassoc","arcp"}` without `nnan`), or filter/raise on non-finite PIT values in `anderson_darling_statistic` before sorting; add a regression test with a NaN PIT asserting NaN (or a raise).

## NUM-W2-2 (Low) - `src/mlframe/feature_engineering/hurst.py:34, 206-209` (same `_ZERO_EPS` used at `:274`, `:537`, `:564`)

Statement: the R/S Hurst estimate is scale-invariant but rejects a window with an absolute threshold `sd <= 1e-12 or range <= 1e-12`, so the result depends on the units of the input.

Scenario: a series in small units (e.g. per-tick log returns of a liquid instrument with std 5e-13, or a column pre-scaled by 1e-9) gives `sd = 5e-13 <= 1e-12` and the feature returns NaN for every window, while the identical series multiplied by 1e6 returns a valid H. Conversely the guard is far too weak for a large-offset near-constant series, though centring by `mu` makes that case benign.

Disposition: RESOLVED-needed (cheap): make the threshold relative, e.g. `sd <= 1e-12 * max(1.0, abs(mu))` or compare against `np.finfo(float).eps * max|x|`; or DOC the absolute floor in the feature docstring if the owner prefers.

No other real numerics defects found in the searched classes.
