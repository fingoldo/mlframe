# metrics/ (classification, regression, calibration metrics) -- mlframe audit

## Scope

All 39 files under `src/mlframe/metrics/**` matching the cluster's declared scope (the `_benchmarks/` subfolder -- 43 files / 4368 LOC of standalone dev benchmark scripts -- is excluded; see Coverage notes) were read in full:

- `src/mlframe/metrics/__init__.py`
- `src/mlframe/metrics/_auc_per_group.py`
- `src/mlframe/metrics/_core_auc_brier.py`
- `src/mlframe/metrics/_core_cb_logits.py`
- `src/mlframe/metrics/_core_numba_warmup.py`
- `src/mlframe/metrics/_core_precision_mape.py`
- `src/mlframe/metrics/_drift.py`
- `src/mlframe/metrics/_fairness_metrics.py`
- `src/mlframe/metrics/_gpu_metrics.py`
- `src/mlframe/metrics/_ice_metric.py`
- `src/mlframe/metrics/_log_loss_and_separation.py`
- `src/mlframe/metrics/_multilabel_extras.py`
- `src/mlframe/metrics/_multilabel_metrics.py`
- `src/mlframe/metrics/_numba_params.py`
- `src/mlframe/metrics/_ranking_extras.py`
- `src/mlframe/metrics/calibration/__init__.py`
- `src/mlframe/metrics/calibration/_calibration_metrics.py`
- `src/mlframe/metrics/calibration/_calibration_plot.py`
- `src/mlframe/metrics/classification/__init__.py`
- `src/mlframe/metrics/classification/_classification_calibration.py`
- `src/mlframe/metrics/classification/_classification_extras.py`
- `src/mlframe/metrics/classification/_classification_extras_blocks.py`
- `src/mlframe/metrics/classification/_classification_report.py`
- `src/mlframe/metrics/classification/_gains_lift.py`
- `src/mlframe/metrics/classification/_ordinal_cutpoints.py`
- `src/mlframe/metrics/classification/_threshold_optimization.py`
- `src/mlframe/metrics/classification/_weighted_kappa.py`
- `src/mlframe/metrics/core.py`
- `src/mlframe/metrics/iteration_metrics.py`
- `src/mlframe/metrics/quantile.py`
- `src/mlframe/metrics/rank_correlation.py`
- `src/mlframe/metrics/ranking.py`
- `src/mlframe/metrics/regression/__init__.py`
- `src/mlframe/metrics/regression/_regression_benchmark.py`
- `src/mlframe/metrics/regression/_regression_corr.py`
- `src/mlframe/metrics/regression/_regression_deviance.py`
- `src/mlframe/metrics/regression/_regression_extras.py`
- `src/mlframe/metrics/regression/_regression_metrics.py`
- `src/mlframe/metrics/scoring.py`

Every file was read completely (none were too large to review in full; the largest, `_core_auc_brier.py` at 987 LOC, was read end-to-end). Real totals: **39 files, 14086 LOC** reviewed (via `wc -l`, excluding `_benchmarks/`).

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P0 | correctness | regression/_regression_metrics.py:822 | `fast_regression_metrics_block` returns R2 = **-inf** (not sklearn's 0.0) for a constant `y_true` with any nonzero residual. |
| F2 | P0 | correctness | regression/_regression_extras.py:824 | Same `-inf` defect duplicated in `fast_regression_metrics_block_extended`; also poisons the returned `NSE` (`nse = r2`). |
| F3 | P1 | correctness | regression/_regression_metrics.py:144-145 | `_fast_r2_score_seq`/`_fast_r2_score_par`/weighted variants return a flat `0.0` for constant `y_true` even on a **perfect** fit, where real sklearn returns `1.0`; three R2-family kernels in this package (`fast_r2_score`=0.0-always, `fast_regression_metrics_block*`=0.0-or--inf, `_nash_sutcliffe_kernel`=NaN) disagree with each other and with sklearn on the exact same degenerate input. |
| F4 | P1 | correctness | classification/_classification_extras_blocks.py:334-336 | `fast_multiclass_confusion_metrics_block`'s `macro_precision`/`macro_recall`/`macro_f1` average over **all** `n_classes` (including labels absent from both `y_true` and `y_pred`), deflating the macro metrics -- the exact bug class `fast_classification_report`'s `macro_over_present` flag was added to fix, left unfixed in this sibling "fused block" function. |
| F5 | P1 | robustness | _core_auc_brier.py:151-187 | `fast_roc_auc_unstable` has no `_check_equal_length` call (unlike `fast_roc_auc` two functions below it), so a mismatched `y_true`/`y_score` length reaches the `@njit` kernel with bounds-checking off and reads out-of-bounds memory instead of raising. |
| F6 | P1 | robustness | _core_auc_brier.py:436-453 | `fast_roc_auc`'s `sample_weight` branch never validates `len(sample_weight) == len(y_true)` before calling `fast_numba_auc_weighted`, which indexes `w[desc_score_indices]` with numba bounds-checking off. |
| F7 | P1 | robustness | classification/_weighted_kappa.py:24-30,61-65 | `weighted_kappa`/`quadratic_weighted_kappa` never validate that `y_true`/`y_pred` labels fit inside `[0, n_classes)` when the caller supplies an explicit (too-small) `n_classes`; `_confusion_matrix`'s `obs[y_true[k], y_pred[k]] += 1.0` then writes out of bounds under `@njit` (bounds-checking off). |
| F8 | P1 | correctness | classification/_classification_report.py:577-585 | `_batch_per_class_ice_kernel`'s inline min/max scan seeds `min_val=1.0, max_val=0.0` instead of the first data value (unlike the "gold" `_fast_calibration_binning_serial`, which explicitly seeds from `y_pred[0]` "so predictions outside [0,1] bin correctly"); when a class's predicted-probability column is entirely `<0` or entirely `>1`, one sentinel never gets touched and the computed bin span (and hence ICE) is silently wrong -- breaking the function's own documented "bit-exact equivalent of `fast_ice_only`" contract. |
| F9 | P2 | edge-case | _multilabel_metrics.py:293-300 | `_coerce_multilabel_array` does `np.asarray(arr).astype(np.uint8)` with no NaN/out-of-{0,1} validation; a NaN or out-of-range float in a multilabel indicator matrix is silently cast to an in-range 0/1 code (undefined-but-typically-0 on NaN) instead of raising, corrupting hamming/subset/jaccard/F1 results. |
| F10 | P2 | edge-case | scoring.py:75-79 | `rmsle_loss` clips `y_pred` to `[0, None]` before `log1p` but never validates/clips `y_true`; a negative `y_true` silently produces `NaN` (or a domain error) with no warning, unlike the numba `fast_rmsle` counterpart in `regression/_regression_extras.py`, which explicitly counts and warns on negative rows. |
| F11 | P2 | robustness | regression/_regression_corr.py:154-166 | `fast_concordance_index` reads `y_true.shape[0]` directly (no `np.asarray()` coercion, no `_check_equal_length`) while every sibling function in the same module (`fast_pearson_corr`, `fast_kendall_tau`, ...) coerces first; a plain Python list -- a legal input to every other function here -- raises an unhelpful `AttributeError` instead of a validated error or a graceful result. |

**F1/F2** (`fast_regression_metrics_block` / `_extended`): with a constant `y_true` batch (`ss_tot == 0`) and ANY nonzero residual, `r2 = 0.0 if sum_sqr == 0.0 else float("-inf")`. Real `sklearn.metrics.r2_score` semantics (confirmed against sklearn's source: `output_scores` initialised to `1.0`, only overwritten to `0.0` when the residual is nonzero and the denominator is zero) give **1.0** for a perfect fit of a constant target and **0.0** for an imperfect one -- never `-inf`. This is realistic: any small validation fold, extreme-quantile target, or per-iteration metric capture on a tiny batch can produce a constant `y_true` slice; the caller then sees `R2 = -inf` silently poisoning any downstream mean/best-tracking/early-stopping comparison. The existing regression test (`tests/metrics/regression/test_fused_regression_metrics_block.py::test_constant_y_true_returns_r2_zero_when_perfect`) actively pins the wrong value (asserts `R2 == 0.0` for the *perfect* case, when sklearn's real answer is `1.0`), and there is no test at all for the imperfect/`-inf` branch. `fast_regression_metrics_block` is wired directly into the training suite's regression report (`training/reporting/_reporting_regression`), so this is user-facing. Suggested fix: replace the `ss_tot<=0` branch in both functions (and align `_fast_r2_score_seq`/`_par`/weighted variants, see F3) with sklearn's actual convention (`1.0` when `sum_sqr==0` else `0.0`), and fix the misleading test/comment.

**F3**: three different "R2-equivalent" kernels in this same package disagree on the ss_tot==0 degenerate case (`0.0` always / `0.0`-or-`-inf` / `NaN`), none of which matches real sklearn. Suggested fix: converge all of them on the sklearn-accurate `1.0`-perfect/`0.0`-imperfect rule documented above, in one pass, with a shared regression test.

**F4**: e.g. `n_classes=10` declared but only 6 labels ever appear in a given eval batch (common on rare-class / early-training-iteration multiclass data) -> the 4 phantom classes contribute `precision=recall=f1=0.0` each into `.mean()`, deflating `macro_precision`/`macro_recall`/`macro_f1` by `(K_present/K_declared)`. Suggested fix: port the same `present_macro = (row_sums>0)|(col_sums>0)` mask + `macro_count` divisor pattern `fast_classification_report` already uses.

**F5**: a caller that (accidentally) passes differently-sized `y_true`/`y_score` to `fast_roc_auc_unstable` gets a numba out-of-bounds read instead of the `ValueError` `fast_roc_auc` raises for the identical mistake -- inconsistent contract between two closely related public functions, and a genuine crash/undefined-behavior risk. Suggested fix: add the same `_check_equal_length(y_true, y_score)` call `fast_roc_auc` already has.

**F6**: a caller passing a `sample_weight` array whose length doesn't match `y_true`/`y_score` (e.g. a stale weight vector from a previous fold) gets silently-garbage or crashing behavior instead of a clear error. Suggested fix: `_check_equal_length(y_true, sample_weight)` before dispatching to `fast_numba_auc_weighted`.

**F7**: `weighted_kappa(y_true, y_pred, n_classes=3)` where the actual labels reach `5` writes `obs[5, ...]` into a `(3,3)` array under `@njit` (bounds-checking off) -- undefined behavior (crash or silent corruption), not a clean error. Only the `n_classes=None` (auto-derived) path is currently safe. Suggested fix: validate `yt.max() < n and yp.max() < n` (and `min() >= 0`, already checked) whenever `n_classes` is explicitly supplied, before calling `_confusion_matrix`.

**F8**: realistic trigger is any upstream bug that feeds raw scores/logits instead of properly-normalized probabilities into `compute_probabilistic_multiclass_error`'s batched fastpath for one class column -- the batched kernel silently produces a different (wrong) ICE than the "legacy" per-class loop it claims bit-exact equivalence to, instead of both agreeing or both erroring. Suggested fix: seed `min_val`/`max_val` from `y_p[0]` (mirroring `_fast_calibration_binning_serial`) instead of the `1.0`/`0.0` constants.

## Proposals

| ID | Category | File:Line | Summary |
|----|----------|-----------|---------|
| PR1 | test-coverage | regression/_regression_metrics.py, regression/_regression_extras.py | Add one shared regression test parametrized across `fast_r2_score`, `fast_regression_metrics_block`, `fast_regression_metrics_block_extended`, and `fast_nash_sutcliffe` pinning the sklearn-accurate constant-`y_true` convention (1.0 perfect / 0.0 imperfect), replacing the currently-wrong `test_constant_y_true_returns_r2_zero_when_perfect`. |
| PR2 | test-coverage | classification/_weighted_kappa.py | Add a regression test for `weighted_kappa(..., n_classes=<too small>)` asserting a clean `ValueError` (once F7 is fixed) instead of relying on numba's unguarded OOB write. |
| PR3 | consistency | _gpu_metrics.py:66-93,445-461 | `is_gpu_metrics_available`/`_resolve_backend` have no `MLFRAME_DISABLE_GPU` opt-out check, unlike every GPU-branching module under `feature_selection/filters/`; wiring the same env-var convention in would give one consistent GPU-disable UX across the codebase (today the only override here is the per-call `force_backend='cpu'` kwarg or the OS-level `CUDA_VISIBLE_DEVICES=""`). |
| PR4 | architecture | _core_auc_brier.py (987 LOC) | This file is within ~1-13 lines of the repo's own ~900-1000 LOC split convention; the bootstrap-resampler section (`make_bootstrap_auc_resampler` + its 3 supporting njit kernels, ~180 LOC) is a natural sibling-module candidate before the file grows further. |
| PR5 | code-quality | quantile.py:375,500-501 | `quantile_summary`'s `Dict[str, Any]` return annotation references `Any`, which is only imported at the very bottom of the file (after every use); it works today only because `from __future__ import annotations` postpones evaluation -- move the `from typing import Any` up to the top import block with the other `typing` imports for readability and to avoid tripping any future strict-annotation tooling. |
| PR6 | docs/consistency | _multilabel_metrics.py:317-329 | `hamming_loss`'s parallel-dispatch gate is a hardcoded `N*K > 1_000_000` literal while `subset_accuracy`/`jaccard_score_multilabel` share the row-count `_PARALLEL_MULTILABEL_THRESHOLD` constant from `_numba_params.py`; both are separately benchmarked and documented, but a one-line comment cross-referencing the two conventions (or hoisting the `1_000_000` into a named constant next to `_PARALLEL_MULTILABEL_THRESHOLD`) would save the next maintainer a re-derivation when retuning thresholds. |

## Coverage notes

- `src/mlframe/metrics/_benchmarks/` (43 files, 4368 LOC of standalone `bench_*.py` / `profile_*.py` scripts) was intentionally **not** audited for findings: the task's own file/LOC estimate ("~39 files, ~14.1k LOC") matches exactly the non-`_benchmarks` file set (verified via `wc -l`), and these are dev-only measurement harnesses (not imported by any production code path) rather than the metrics library surface itself. Skimmed several for obvious red flags while cross-referencing bench claims cited in docstrings elsewhere; found nothing that changed a Findings-section verdict.
- Did not run the test suite (read-only audit constraint) -- test-coverage statements above are based on `Grep`/`find` over `tests/metrics/**` plus reading the specific test files cited (`test_fused_regression_metrics_block.py`, `test_weighted_kappa.py`), not a live pytest run.
- Did not verify F1-F3's sklearn-semantics claim by importing/running real `sklearn.metrics.r2_score` in this environment (read-only, no code execution beyond `wc -l`/`grep`); the claim is based on sklearn's well-known, documented `force_finite`/degenerate-R2 behavior (`output_scores` defaults to `1.0`, overwritten to `0.0` only when the numerator is nonzero and the denominator is zero) and is independently corroborated by the fact that mlframe's own three R2-family kernels already disagree with each other on this exact input.
