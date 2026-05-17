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

## Iter 2 -- TODO

Hypothesis to test: at SMALL data scales (n=600-1000 fuzz cells), orchestration
overhead might be relatively larger because the model-fit time shrinks linearly
with n but bookkeeping is constant. Pick a fuzz cell with n=600 or n=1000 from
`test_fuzz_suite.py` and re-profile, looking specifically for non-zero
mlframe-code tottime entries.
