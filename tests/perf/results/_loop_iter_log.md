# /loop iter log -- fuzz combo profile-and-optimize cycles

Self-paced loop session started 2026-05-17 to find hot mlframe code paths and
apply optimizations. Each iteration: pick an existing baseline profile from
`tests/perf/results/profile_iter*_baseline.txt`, analyse the top hotspots,
propose + apply an optimization OR reject with rationale (per
`feedback_perf_measure_first`: skip if no actionable speedup, document so the
finding isn't re-flagged).

Max 5 iterations per user instruction. Stop early if 2 consecutive iterations
reject (suggests the suite is already well-optimized at the profiled scale).

## Iter 1 -- 2026-05-17

Profiles examined: `profile_iter195_baseline.txt` (1M rows, HGB+LGB, 187s
train + 17s predict, multiclass_classification), `profile_iter193_baseline.txt`
(MRMR + XGB, 349s train, 4 model rounds).

**Hot tottime breakdown (iter195 train phase):**

| Function | tottime | Domain |
|---|---|---|
| HGB `grower.split_next` | 45.97 s | sklearn C++ kernel |
| LGB `basic.update` | 19.85 s | LightGBM C++ kernel |
| HGB `grower._initialize_root` | 18.37 s | sklearn C++ kernel |
| HGB `gradient_boosting.fit` | 10.26 s | sklearn coordinator |
| `train_mlframe_models_suite` | **0.009 s** | mlframe orchestration |
| `_train_one_target` | **0.005 s** | mlframe orchestration |
| `_call_train_evaluate_with_configs` | **0.001 s** | mlframe delegation |

**Hot tottime breakdown (iter193 train phase):**

| Function | tottime | Domain |
|---|---|---|
| XGBoost `core.update` | 270.59 s | XGBoost C++ kernel |
| `train_mlframe_models_suite` | **0.005 s** | mlframe orchestration |
| MRMR `fit` (via _passthrough_cols_fit_transform wrapper) | ~24 s cumtime | mlframe + numba-heavy |

**Verdict: REJECT optimization on these combos.**

Reasoning:
- Total mlframe-code tottime across both profiles: < 0.1 s out of 187-349 s
  wall (< 0.05% of train time).
- Bottlenecks (XGBoost / HGB / LGB `update` / `split_next`) are third-party
  C++ kernels we cannot replace without swapping engines entirely.
- MRMR.fit is the only mlframe-owned hot path (24 s, ~7% of iter193), but the
  function already runs on numba-compiled inner loops per
  `mlframe.feature_selection.filters.mrmr` (numba kernels under `_kernels_*`).
  Further optimization there would need a dedicated benchmark on the MRMR
  inner-loop variants -- not a fuzz-combo-driven discovery.

This confirms the prior audit wave's verdict: the orchestration layer is
already near-zero overhead at the 1M-row scale.

## Iter 2 -- 2026-05-17 -- RESOLVED, 63x speedup

Cell profiled: `c0105_69237005-cb_hgb_lgb_linear-pl_utf8-n600` (small-n, 4-model).

Total wall 148.7s. cProfile reveals new hotspot category: bootstrap CI
computation in `dummy_baselines._bootstrap_ci_for_strongest`. Hot tottime
breakdown (mlframe-code only, excluding C++ kernels):

| Function | cumtime | calls | mean ms / call |
|---|---|---|---|
| `_bootstrap_ci_for_strongest` | 21.1 s | 1 | 21 094 ms |
| `_resample_metric` | 21.1 s | 2 | 10 547 ms |
| `fn` (line 2235, log_loss_macro closure) | 21.0 s | 2002 | 10.5 ms |

The 2002 `fn` calls dispatch to sklearn's `log_loss` once per bootstrap
resample (1000 resamples * 2 metrics = 2000 + 2 point estimates). Each
sklearn call pays input-validation + dtype-cast overhead.

**Fix**: new `_vectorized_bootstrap_logloss_samples` helper in
`dummy_baselines.py` that generates all bootstrap indices in one shot and
computes log-loss via numpy broadcasting. Handles 1D binary and 2D
multilabel-macro shapes; returns None for shapes the caller should bounce
to the sklearn fallback.

Measured (median of 3 trials, n=600 / 1000 resamples):

| Path | ms |
|---|---:|
| sklearn-loop (reference) | 1 519.6 |
| vectorised (new) | 24.0 |
| **speedup** | **63.3x** |

Regression suite at `tests/training/test_audit_2026_05_17_loop_2_bootstrap_logloss.py`
asserts:
1. Percentile equivalence to sklearn loop to 2 dp (binary 1D).
2. Multilabel 2D returns finite log-loss in [0.2, 2.0] range.
3. Bad shapes return None (caller falls back).
4. Perf gate >= 5x (actual 63x at production n).

Wired into `_resample_metric` BEFORE the sklearn fallback for any path with
"log_loss" in `primary_metric` -- covers binary (when numba guard misses
because y is float-encoded), multiclass, and multilabel-macro. Numba binary
path is preserved as the fastest happy path for int-encoded binary y.

## Iter 3 -- TODO

Hypothesis: with the log-loss bootstrap fixed, the next non-trivial mlframe
hotspot at small-n is likely either the per-target loop setup overhead in
`_phase_train_one_target` (CACHE-* / SCHEMA-HASH-* sites from the original
audit table) or pandas/polars conversion in `_phase_helpers`. Profile a
different small-n fuzz cell (e.g. n=1000 mlp+xgb or n=600 mrmr) and look
specifically for non-zero mlframe-code tottime.
