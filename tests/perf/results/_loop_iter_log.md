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

## Iter 6 -- 2026-05-17 -- REJECTED (streak 1/100)

Cell profiled: `c0020_695e4e82-linear_mlp-pl_enum-n5000` (linear+mlp on
polars Enum dtype, n=5000). 204s wall.

mlframe-owned TOTTIME breakdown (only entries > 0.05 s OWN CPU):

| Function | tottime | calls |
|---|---:|---:|
| `io._writer` (zstd compression for model save) | 0.565 s | 2 |
| `neural/data._extract` (MLP DataLoader per-batch) | 0.313 s | 78 |
| `_train_one_target` | 0.001 s | 1 |
| `train_and_evaluate_model` | 0.001 s | 8 |

Cumulative time inside `_train_one_target` is 183s (~90% of wall), but the
mlframe-side share of that is negligible: `train_and_evaluate_model` adds
0.001s own; the rest is sklearn (linear) + torch + Lightning (MLP) C++ /
C-extension work.

`io._writer` 0.565s is pickle.dumps + zstd.compress -- both C-ext-bound;
mlframe just dispatches. `neural/data._extract` 0.313s across 78 batches =
4ms/batch -- batch-extract bookkeeping with no obvious vectorisation gain
without architectural changes.

**Verdict: REJECT.** No mlframe-owned hot computation surfaces above the
1.2x gate on this profile. Streak counter: 1/100. Loop continues.

## Termination policy clarification (post-iter-5)

Mid-session correction by user: the loop is NOT capped at a fixed iteration
count. The real termination signal is a **consecutive-rejection streak**:
stop only after 100 reject-in-a-row (each `RESOLVED` resets the counter).
Documented in CLAUDE.md.

Streak after iter 5: **0/100** (iter 5 was RESOLVED). Loop resumes with iter 6+.

## Partial wrap-up (iters 1-5)

5 iterations so far:
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

## Iter 7 -- 2026-05-17 -- REJECTED (streak 2/100)

Cell profiled: `test_predict_round_trip_parity` -- predict-path cProfile on
the round-trip parity test (LGB + HGB + CB suite, n=600). 18.4s wall.

mlframe-owned TOTTIME breakdown (only entries > 0.05 s OWN CPU):

| Function | tottime | calls |
|---|---:|---:|
| `_run_predict_phases` orchestrator | 0.014 s | 8 |
| `_combine_probs` | 0.011 s | 24 |
| `_resolve_quantile_alphas_metadata_lookup` | 0.003 s | 24 |

Cumulative time inside predict entries is 17.8s (~96% of wall), but the
mlframe-side share is < 0.1 s OWN CPU. Bottlenecks are CatBoost
`_predict_with_fallback` (8.4 s cumtime, CB pool C-ext), LGB Booster
`predict` (4.1 s cumtime), and HGB `predict_proba` (2.3 s cumtime).

**Verdict: REJECT.** Predict path is third-party-dominated; nothing in
mlframe orchestration measurable above noise.

## Iter 8 -- 2026-05-17 -- REJECTED (streak 3/100)

Candidate: `metrics/core.py:2606(_batch_per_class_ice_kernel)` -- 114.30 s
tottime / 939 calls in `profile_iter147_baseline.txt`, largest mlframe-
owned TOTTIME across all 196 archived baselines.

Already decorated `@numba.njit(fastmath=False, cache=True, nogil=True,
parallel=True)`. Probed whether enabling `fastmath=True` would speed up
the reductions at the parallelism threshold.

Bench (N=1_000_000 K=3, 7 trials each):
- current (fastmath=False): 234.6 ms median
- fastmath=True clone:      251.4 ms median
- speedup: 0.93x (REGRESSION)
- numerical drift: 3.8e-6 max abs diff

**Verdict: REJECT.** Kernel is already memory-bandwidth bound at this N;
fastmath does not help and trades 4 ppm of numerical fidelity for nothing.
Documented so it isn't re-flagged in future loop iterations.

## Iter 9 -- 2026-05-17 -- REJECTED (streak 4/100)

Candidate: `metrics/core.py:374(_cb_logits_to_probs_multiclass_par)` --
4.85 s tottime in `profile_iter147_baseline.txt`. Parallel softmax with
max-subtraction trick.

Bench (N=1_000_000 K=3, 7 trials each):
- current (fastmath=False): 9.54 ms median
- fastmath=True clone:      11.55 ms median
- speedup: 0.83x (REGRESSION)
- numerical drift: 5.55e-17 (machine epsilon, indistinguishable)

**Verdict: REJECT.** Same conclusion as ICE kernel: softmax is already
memory-bandwidth bound; fastmath's reassociation overhead exceeds its
compute win on this kernel shape. Numerically identical, so fastmath
would be safe -- but slower is slower.

## Iter 10 -- 2026-05-17 -- REJECTED (streak 5/100)

Candidate: `training/helpers.py:400(integral_calibration_error)` --
2.68 s tottime, 929 calls, 134.15 s cumtime in `profile_iter147_baseline.txt`.

Closure-wrapped `integral_calibration_error` that forwards to
`compute_probabilistic_multiclass_error` with captured `method`,
`mae_weight`, `*_weight` config. Own work / inner work ratio = 2.681 /
134.152 = **2.0 %**.

**Verdict: REJECT.** Even if the closure dispatch were free, the ceiling
speedup is 1.02x -- below the 1.2x gate. The 144 ms / call cumtime is
dominated by the numba kernel (iter 8) and downstream brier / roc_auc
sklearn calls, not by Python orchestration.

## Iter 11 -- 2026-05-17 -- REJECTED (streak 6/100)

Candidate: `training/utils.py:375(get_pandas_view_of_polars_df)` -- up to
1.05 s tottime in `profile_iter103_baseline.txt` (3 calls, 5.41 s cumtime).
Per-call OWN time ~350 ms.

Read the function: per-call work is (1) nested-dtype scan with WARN, (2)
`to_arrow()` C-ext, (3) dictionary-column int32 cast loop, (4) bool-column
detection, (5) `to_pandas()` C-ext. Existing inline comment
(`utils.py:513-515`) already records a 2026-04-14 bench: "short-circuit on
'no dictionary columns' delivered only 1.16x on pure-numeric workloads
(below 1.2x threshold)".

Available Python-level reductions:
- Combine scans (2) + (3) + (4) into one pass: marginal, < 5 % of OWN time.
- Skip nested-dtype WARN scan after first call same-schema seen: already
  cached in `_NESTED_DTYPE_WARN_SEEN`.

Effective ceiling on the wrapper: ~1.05 s OWN out of 5.41 s cumtime = 19 %.
Compressing wrapper OWN to zero would yield 1.24x on the wrapper alone,
but it's called 3x and the bulk of training-loop wall is elsewhere; net
suite-level impact < 5 %.

**Verdict: REJECT.** Below 1.2x net-suite gate; the prior 2026-04-14 bench
already explored this surface.

## Status after iters 6-11

| Iter | Verdict | Streak | Candidate |
|---:|---|---:|---|
| 6 | REJECT | 1/100 | linear+mlp pl_enum n=5000 fuzz combo |
| 7 | REJECT | 2/100 | predict-path round-trip parity |
| 8 | REJECT | 3/100 | `_batch_per_class_ice_kernel` (numba) |
| 9 | REJECT | 4/100 | `_cb_logits_to_probs_multiclass_par` (numba) |
| 10 | REJECT | 5/100 | `integral_calibration_error` (closure) |
| 11 | REJECT | 6/100 | `get_pandas_view_of_polars_df` |

After 6 consecutive REJECTs, the surface mlframe owns at >1.2x gate
appears exhausted on the archived baselines. The pattern across all six:

1. **numba parallel kernels** (iters 8, 9) are memory-bandwidth bound at
   the parallelism threshold; fastmath is a regression on both. Likely
   also true for the remaining numba kernels (mrmr inner loops etc.) but
   each individual probe is cheap if needed.
2. **Closure wrappers** (iter 10) have low own/cum ratios (< 5 %),
   capping any wrapper-level win below the 1.2x gate.
3. **C-extension bridges** (iter 11, pandas-view) are dominated by the
   underlying C-ext call itself; Python-level scans contribute < 1.2x
   ceiling.
4. **Third-party heavy lifters** (iter 6 LightGBM/Lightning, iter 7
   CB/LGB/HGB predict) are off-limits.

Loop continues per the 100-consecutive-reject policy, but each subsequent
iteration is expected to follow one of these four patterns until an
unprofiled code path surfaces.

## Iter 12 -- 2026-05-17 -- RESOLVED (streak resets 0/100)

Picked random fuzz cell:
`c0097_867cf5d3-cb_hgb_lgb_mlp_xgb-pl_utf8-n1000` (5-model suite, polars
utf8, n=1000, binary classification with MRMR + ensembles).

cProfile run via pytest `-k "c0097_867cf5d3"` hit `pytest.mark.timeout(300)`
**before any model fit started**, hanging inside `_cb_pool._cb_gpu_usable`
on the very first GPU probe. Traceback bottom:

```
helpers.py:293  _cb_task = "GPU" if _cb_gpu_probe() else "CPU"
_cb_pool.py:532   if not _cached_gpu_info():
_cb_pool.py:494     with _GPU_PROBE_LOCK:      <-- HANG
+++ Timeout +++
```

**Root cause: reentrant deadlock.** `_cb_gpu_usable()` acquires
`_GPU_PROBE_LOCK` (line 529), then while still holding it calls
`_cached_gpu_info()` which tries to acquire the same lock (line 494).
With `threading.Lock()` (non-reentrant) the second acquire blocks
forever. The bug surfaced only on the very first GPU probe in a process
(when `_GPU_INFO_PROBED` is still False so `_cached_gpu_info` cannot
short-circuit on its pre-lock fast path).

Fix: `threading.Lock()` -> `threading.RLock()` at module load. Single
line + 4-line WHY comment.

Concurrent landing: a sibling fix-agent committed the same RLock change
in `dcd9270` 45s before this iter's commit; my `git commit -o` ended
up only contributing the regression test (since the source diff against
the new HEAD was already zero). Test still locks in the requirement.

Regression test (3 sub-tests, all green in 7.4s):

| Test | What it asserts |
|---|---|
| `test_gpu_probe_lock_is_reentrant` | Module-level lock is an `RLock` instance |
| `test_reentrant_acquire_does_not_deadlock` | Same-thread double-acquire succeeds within 1s timeout — would fail on plain `Lock` |
| `test_cb_gpu_usable_then_cached_gpu_info_no_hang` | Worker thread completes `_cb_gpu_usable` within 30s with `_GPU_INFO_PROBED` reset — direct repro of the deadlocking call chain |

cProfile baseline-vs-after measurement skipped: the bug was a hang
(infinity vs `<10ms`), not a perf tradeoff. The 5-min timeout is the
"baseline", the RLock fix turns it into the millisecond-scale GPU probe
the function was always meant to do.

Streak counter: **0/100** (RESOLVED resets the consecutive-reject count).
Commit: `ca67aa9`. Loop continues.

## Iter 13 -- 2026-05-17 -- REJECTED (streak 1/100)

Re-ran the same `c0097_867cf5d3-cb_hgb_lgb_mlp_xgb-pl_utf8-n1000` fuzz
cell now that iter 12 unblocked the GPU-probe deadlock. cProfile dump
landed in 295.83s test wall (cProfile-internal 328s).

**Wall breakdown** (mlframe vs third-party):

| Component | Time | % of wall | Owner |
|---|---:|---:|---|
| MLP / PyTorch Lightning fit | 173 s | 58 % | torch (third-party) |
| `screen_predictors` numba JIT cold-start | 9 s | 3 % | numba JIT (one-shot) |
| MRMR + pre-pipeline orchestration | 40 s | 14 % | mlframe (distributed) |
| CB / LGB / HGB native fits | 30 s | 10 % | C-ext (third-party) |
| Misc orchestration + model I/O | 43 s | 15 % | mixed |

**mlframe-owned TOTTIME (only entries > 0.05 s OWN CPU):**

| Function | tottime | calls | per-call |
|---|---:|---:|---:|
| `permutation.py:35(shuffle_arr)` | 0.478 s | 21 | 23 ms |
| `io.py:174(_writer)` (pickle/zstd save) | 0.191 s | 11 | 17 ms |

Everything else is < 0.05 s tottime. The largest mlframe-OWN compute is
0.478 s out of 295 s = 0.18 % of wall. Compressing it to zero buys
0.0018x suite speedup -- two orders of magnitude below the 1.2x gate.

**The biggest single cost (173 s MLP / Lightning fit on n=1000) is
genuinely torch overhead**, not mlframe orchestration: 2 fits x ~86 s
each, with ~15 epochs at ~30-75 it/s. Lightning DataLoader / Trainer
setup amortizes poorly at this N, but the setup itself is library
code we cannot rewrite without forking Lightning.

**Verdict: REJECT.** No actionable mlframe hotspot on c0097 above the
1.2x gate. The deadlock fix from iter 12 was the only real bug surface
this fuzz cell exposed. Streak counter: 1/100.

## Iter 14 -- 2026-05-17 -- REJECTED (streak 2/100)

Picked `c0079_8aeeb5d5-cb_hgb_lgb_xgb-pl_nullable-n5000` -- 4-booster
suite (no MLP, to avoid the iter 13 torch dominance), pl_nullable
dtype, n=5000, target_type=`multilabel_classification`, 1 embedding col.

**cProfile run aborted at 79s** with
`joblib.externals.loky.process_executor.BrokenProcessPool: A task has
failed to un-serialize. Please ensure that the arguments of the function
are all picklable.` during `MultiOutputClassifier.fit` for the LGB
member with `weight=uniform`.

**Native run (no cProfile) passes in 28s.**

Root cause: `python -m cProfile` instruments the parent interpreter
state, which loky must pickle and ship to child workers when
`MultiOutputClassifier` parallelises per-target fits. The cProfile
profiler object is not picklable; the loky child receives garbage and
unpickle dies before invoking the wrapped fit. Multilabel combos that
spawn loky child pools cannot be cProfile-wrapped at the parent layer.

**This is a profiler limitation, not an mlframe bug.** The native test
passes; no source code change is warranted.

Mitigation for future iterations: skip combos with
`target_type == "multilabel_classification"` when running under
cProfile wrap, OR force sequential joblib via
`JOBLIB_MULTIPROCESSING=0` env. Going with the former for simplicity.

**Verdict: REJECT.** Streak counter: 2/100.
