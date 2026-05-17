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

## Iter 3 -- 2026-05-17 -- RESOLVED, 53.6x speedup

Cell profiled: `c0036_9f570e62-linear_mlp_xgb-pl_utf8-n1000` (small-n + MLP).

Total wall 192.9s. MLP (PyTorch + Lightning) dominates cumtime as expected --
`neural/base.py:fit` 64.8s, `flat.py:validation_step` 22.7s -- but most of
the time inside is `torch._C._nn.linear` (15.3s tottime), cross_entropy_loss
(4.2s), and Lightning's own infrastructure. No mlframe-specific MLP hotspot
worth touching at this iteration's budget.

The next mlframe-owned hotspot the cProfile surfaced was a SIBLING of the
iter-2 fix: `dummy_baselines._paired_bootstrap_vs_runner_up` (line 1960) ran
2000 sklearn `log_loss` calls (1000 paired resamples for strongest vs
runner-up + 1000 for the deltas) -- 6.5s tottime (~3% of suite wall on the
small-n cell, but a constant tax on every classification run with a
runner-up to compare).

**Fix**: reuse iter-2's `_vectorized_bootstrap_logloss_samples` twice with
the SAME seed so the index matrices match and per-resample deltas align.
Inserted BEFORE the legacy sklearn-loop, gated on "log_loss" in
`primary_metric` and NOT "macro" (legacy "log_loss_macro" path returns None
on cost-vs-value grounds; preserved by the gate).

Measured (median of 3 trials, n=600 / 1000 resamples):

| Path | ms |
|---|---:|
| sklearn-paired-loop (reference) | 3237.1 |
| vectorised paired (new) | 60.4 |
| **speedup** | **53.6x** |

Regression suite at `tests/training/test_audit_2026_05_17_loop_3_paired_bootstrap.py`
asserts:
1. Percentile equivalence to sklearn loop (q2.5/q50/q97.5, 0.02 tol).
2. `p_strongest_beats` rate within 0.01 of the sklearn-loop reference on
   a deliberately-tilted setup (p1 ~= y vs p2 random).
3. Perf gate >= 5x (actual 53.6x at production n).
4. Macro-metric path still returns None (legacy contract preserved).

Combined cumulative iter-2 + iter-3 win on dummy-baselines bootstrap CI
path: ~28s -> ~0.5s (~56x at n=600/1000), about 14% wall on small-n cells
where dummy baselines fire. No correctness regressions in 8/8 unit tests
across both iters.

## Iter 4 -- 2026-05-17 -- REJECTED

Cell profiled: `c0070_21030005-cb_hgb_lgb_linear_mlp-pandas-n5000` (5-model,
n=5000, pandas).

Total wall 142.6s. Three angles examined:

1. **Largest cumtime mlframe hotspot**: `_eval_helpers.run_confidence_analysis`
   22.5s (5 calls x 4.5s avg). Callees breakdown:
   - `catboost.core.fit` 8.4s (10 calls — 2 per call due to GPU-then-CPU
     fallback path; CB C++ kernel, not optimisable from Python).
   - SHAP `TreeExplainer + beeswarm plot` ~14s (third-party).
   - mlframe-owned glue: < 50 ms across all 5 calls.
   The 200-iteration default for the confidence regressor was considered for
   reduction to 50 (the file's own comment says 50 is "serviceable"), but the
   estimated 1.4x speedup on the function (6.5s saved out of 22.5s) sits at
   the user's REJECT threshold (perf gain < 1.2x is a hard reject; 1.4x with
   a documented quality reduction is borderline). Deferred to a separate
   diagnostic-quality discussion, not a /loop optimisation.

2. **Second cumtime mlframe hotspot**: `_reporting.report_probabilistic_model_perf`
   4.9s (10 calls x 0.49s). Callees:
   - `metrics.fast_calibration_report` 2.8s — already heavily vectorised
     ("fast_" prefix; calls fast_brier_score_loss, fast_calibration_binning,
     fast_aucs_per_group_optimized internally + matplotlib plotting).
   - `compute_batch_aucs` 0.65s — already GPU-dispatched.
   - `_cb_pool._predict_with_fallback` 0.65s — CB predict + cache lookup.
   No further mlframe-side computation to remove; the elapsed time is split
   across matplotlib drawing + GPU AUC calls already at their performance
   ceiling.

3. **TOTTIME (own CPU) on mlframe code**: every entry < 0.1 s except the
   already-resolved `_resample_metric` (iter 2) and `_paired_bootstrap_vs_runner_up`
   (iter 3). Top non-resolved entry: `atomic_write_bytes` 0.007 s tottime
   (file I/O for model save). Even 100% removal would buy ~7 ms.

**Verdict: REJECT.** No mlframe-owned hot computation surfaces above the 1.2x
gate on this profile. All measured elapsed time at small-n is in torch / CB /
sklearn / SHAP / matplotlib C++ or third-party kernels. Iter 2 + iter 3
fixed the only two mlframe hotspots visible in cProfile across iters 1-4
(both bootstrap CI sites, ~56x combined speedup).

## Iter 5 -- 2026-05-17 -- RESOLVED, 16s cold-start saved for non-MRMR suites

Angle: import-time overhead. cProfile (iter 3 c0036) showed
`mlframe/training/core/_setup_helpers.py:1(<module>)` 25.6s cumtime spent in
the module body's eager imports.

Inspecting the imports:
- `sklearn.impute / pipeline / preprocessing` -- expected cost (~5s).
- `category_encoders` -- pulls statsmodels (~5s).
- `from mlframe.feature_selection.filters import MRMR` -- 16s in isolation
  (3-trial median in a fresh process with mlframe.training pre-loaded so the
  filters subgraph is the only new work).
- `mlframe.configs` -- pydantic transit (~3s).

The MRMR eager import is the actionable one: ~16s on every first call to
`train_mlframe_models_suite`, even when the caller passes `use_mrmr_fs=False`
(which is the FeatureSelectionConfig default -- so MOST users pay this tax).

**Fix**: deferred the import from module top to inside the
`if use_mrmr_fs:` branch in `_build_pre_pipelines`. Mirrors the pre-existing
BorutaShap pattern in the same function (a few lines below) that gates the
shap+matplotlib+seaborn import behind `use_boruta_shap`. Module-level
reference is preserved as a `TYPE_CHECKING`-guarded stub for static checkers.

Measured (3-trial median, fresh subprocess with mlframe.training pre-loaded):

| Path | seconds |
|---|---:|
| Before: `from mlframe.feature_selection.filters import MRMR` | 15.9 |
| After (non-MRMR caller): no MRMR import fires | 0 |
| **Cold-start saving (non-MRMR caller)** | **~16s** |

Default `FeatureSelectionConfig.use_mrmr_fs = False`, so opt-out is the
common path. Opt-in callers (`use_mrmr_fs=True`) pay the 16s once when the
import fires lazily inside `_build_pre_pipelines`; subsequent calls share
Python's module cache, so the cost amortises to zero.

Regression suite at
`tests/training/test_audit_2026_05_17_loop_5_lazy_mrmr_import.py` asserts:
1. `_setup_helpers` no longer re-exports `MRMR` at module top
   (TYPE_CHECKING-guarded references are invisible at runtime).
2. After fresh import of `_setup_helpers`, `mlframe.feature_selection.filters`
   stays out of `sys.modules`.
3. `_build_pre_pipelines(use_mrmr_fs=False, ...)` does NOT trigger the import
   (opt-out path cheap).
4. `_build_pre_pipelines(use_mrmr_fs=True, ...)` DOES trigger the import
   (opt-in path correct).
4/4 pass in 16s.

## Final wrap-up

5/5 iterations complete:
- Iter 1: REJECT -- 1M-row orchestration < 0.05 % of wall, no mlframe hotspot.
- Iter 2: 63x on `_resample_metric` log_loss bootstrap CI (commit 4c2574e).
- Iter 3: 53.6x on `_paired_bootstrap_vs_runner_up` log_loss bootstrap
  (commit 57736a1).
- Iter 4: REJECT -- all measurable hotspots third-party-dominated (CB / SHAP /
  matplotlib) on small-n cells; iters 2+3 already covered the only two
  mlframe-owned hotspots cProfile surfaced (commit 2c88fe4).
- Iter 5: 16s cold-start saved on non-MRMR suites by deferring the MRMR
  import from `_setup_helpers` module top to its single call site.

Combined wins across the loop session:
- Bootstrap CI surface (iters 2+3): ~28s -> ~0.5s on small-n cells = ~56x.
- Cold-start (iter 5): -16s on every first call where `use_mrmr_fs=False`
  (the default).
