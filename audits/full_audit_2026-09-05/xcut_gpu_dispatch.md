# Cross-cutting audit: GPU / numeric-backend dispatch

Scope: `src/mlframe` GPU + backend-dispatch paths (cupy / numba.cuda / njit), concentrated on
`feature_selection/` and `training/`, plus the shared `_ktc_dispatch*` modules and `metrics/_gpu_metrics.py`.
Bug class hunted: a GPU or backend-dispatch path that produces a DIFFERENT answer, or a decision made from a
measurement that does not mean what it appears to.

Method: read-only. ~55 dispatch/kernel modules read. Two small numpy-only reproductions run
(`PYTHONPATH=src python -c ...`); no GPU benchmarking, no test suite (machine under foreign load).

---

## CONFIRMED

### XGD-01 [P0] gpu-relevance-null-drops-base_seed
**File:** feature_selection/filters/evaluation.py:501
**Summary:** In `evaluate_candidate`, the GPU relevance branch calls `mi_direct_gpu(...)` **without**
`base_seed=_baseline_seed`, while BOTH CPU branches in the same if/else (lines 534 and 560) pass it. The
seed is what the permutation null is drawn from, and the null is subtracted from the score that drives
selection (`direct_gain = max(0.0, direct_gain - null_mean)` at line 545).
**Failure scenario:** Any MRMR fit on a CUDA host with `use_gpu=True`. `mi_direct_gpu` does
`_shuf_rng = cp.random.default_rng(base_seed)` (filters/gpu.py:625); with `base_seed=None` cupy seeds from
OS entropy. Consequences, all on the GPU host only:
  1. `null_mean` and `p_value` are **non-reproducible run to run** with the caller's `random_seed` fixed, so
     `direct_gain` and therefore the selected feature set are non-deterministic.
  2. The comment immediately above the call (lines 493-497) exists solely to make the null MOVE with
     `random_seed` -- "a caller varying random_seed to probe selection stability got a null that never
     changed" -- and the GPU call site defeats exactly that fix.
  3. Two candidates in the same fit are supposed to get *different* derived seeds
     (`hash((random_seed, cand_idx))`); on GPU they get unrelated entropy instead.
**Evidence:** `grep -n base_seed src/mlframe/feature_selection/filters/evaluation.py` returns only lines 534
and 560 -- both CPU. The sibling confirm path proves the intended convention: _confirm_predictor.py:472-482
passes `base_seed=_marginal_base_seed` to `mi_direct_gpu` AND to its CPU fallback. filters/gpu.py's own
docstring documents `base_seed` as "so the shuffle stream is reproducible across calls/hosts".
**Suggested fix:** add `base_seed=_baseline_seed,` to the `mi_direct_gpu(...)` call at evaluation.py:501.

### XGD-02 [P0] mi_direct-gpu-fastpath-drops-base_seed
**File:** feature_selection/filters/permutation.py:731
**Summary:** `mi_direct(..., base_seed=S)` forwards `base_seed` to every njit permutation kernel
(lines 800, 837, 868, 943) but its GPU fastpath re-dispatch to `mi_direct_gpu` omits the argument entirely.
The public API therefore returns a seeded, reproducible `(mi, confidence)` on a CPU-only host and an
unseeded, run-varying one on a CUDA host -- for the identical call.
**Failure scenario:** `mi_direct(..., npermutations>=32, base_seed=42)` twice in one process on a CUDA host.
`confidence = 1 - nfailed/(_i+1)` and the early-stop `original_mi = 0.0` zeroing both depend on the shuffle
stream, so back-to-back calls can return different `confidence` and, at the decision boundary, a zeroed vs
non-zeroed MI. The same call on a CPU-only host is bit-stable.
**Evidence:** read permutation.py:568 (`base_seed: int = 0` in the signature), 728-744 (the GPU call, no
`base_seed` kwarg), and filters/gpu.py:625.
**Suggested fix:** add `base_seed=base_seed,` to the `mi_direct_gpu(...)` call at permutation.py:731.

### XGD-03 [P0] discretize-bin-edges-depend-on-free-vram
**File:** feature_selection/filters/discretization/__init__.py:939
**Summary:** `discretize_2d_array` has three backends that do NOT agree on bin edges. The CPU njit path and
`discretize_2d_array_cuda` both compute EXACT full-column quantile edges. `discretize_2d_array_cuda_row_chunked`
computes them from a **random subsample** (`quantile_subsample_rows`, default `UNIFIED_FE_SUBSAMPLE_N`). The
row-chunked path is entered purely because a *transient* free-VRAM probe failed -- not because of anything
about the data.
**Failure scenario:** Same frame, same `n_bins`, same code, two runs. Run A: the cushion probe passes ->
exact edges. Run B: another process (or an earlier stage's cupy pool) holds VRAM -> `_vram_ok` False ->
row-chunked -> edges from a `rng(0)` subsample of `sub_n < n_rows` rows. Any column whose quantile boundary
falls between two subsample-adjacent values gets different codes for a non-empty set of rows; those codes are
the direct input to the MI / MRMR pipeline, so the SELECTED FEATURE SET changes as a function of another
process's VRAM usage. The dispatch logs "completed via row-chunked CUDA (GPU speed preserved, VRAM-safe)" at
INFO -- it never says the edges became approximate. Note the fallback is NOT a last resort: it is tried
*before* the exact CPU prange path, which is reached only if the chunked GPU call also raises.
**Evidence:** discretization/__init__.py:900-950 (probe -> warning -> row-chunked -> only then CPU);
_discretization_cuda.py:282-289 (`sub_idx = np.sort(np.random.default_rng(0).choice(n_rows, size=sub_n,
replace=False))`); _discretization_cuda.py:188-195 docstring: "APPROXIMATE by construction".
**Suggested fix:** on `not _vram_ok`, go to the exact CPU prange path; keep row-chunked reachable only via an
explicit opt-in argument. Failing that, thread an `exact_edges=True` flag from the FS caller and raise the log
to WARNING naming the approximation.

### XGD-04 [P1] ktc-tuner-times-gpu-resident-but-production-pays-h2d
**File:** calibration/_ktc_dispatch.py:101 and inference/_ktc_dispatch.py:52
**Summary:** Both tuners hoist the host-to-device upload OUT of the timed region and return a device array
(after `Stream.null.synchronize()`, so the timing is correctly *synchronised* -- the defect is what it
measures, not when it stops). Production callers are host-array callers and pay `cp.asarray(...)` on entry
and `cp.asnumpy(...)` on exit. The persisted crossover therefore systematically over-selects `cupy`.
**Failure scenario:** `odds_ratio_combine` at n=100k, k=5, float64. Tuner timing = kernel only (the module
docstring records 5.23 ms cupy vs 8.38 ms njit_parallel -> "cupy" persisted). Production `_odds_combine_cupy`
(calibration/ensembling.py:55-64) additionally moves ~4 MB H2D + ~0.8 MB D2H per call, which the measurement
excludes; the persisted "cupy wins" verdict is drawn from a workload the production path never runs. The CPU
side of the same comparison is biased the OTHER way in the inference tuner, which times
`_apply_njit(preds.copy(), rules_arr)` -- an extra full host copy per iteration that production's in-place
`_apply_njit(out, ...)` does not pay (inference/logical_constraints.py:192-200).
**Evidence:** calibration/_ktc_dispatch.py:101 `p_gpu = cp.asarray(p)` sits above `_gpu_call`, which returns
`r` (a device array); inference/_ktc_dispatch.py:52-53 hoists `preds_gpu`/`rules_gpu` and `_gpu_call` starts
from `preds_gpu.copy()` (D2D, not H2D). The sibling
votenrank/_confidence_gated_blend_ktc_dispatch.py:64-71 gets this RIGHT -- `cp.asarray` inside `_gpu_call`,
ending in `cp.asnumpy(out)` -- and its module docstring states the reason explicitly: "cupy resident 0.8 ms /
cupy e2e 8.5 ms (host input: H2D transfer dominates -- slower than njit_parallel)". Same repo, same shared
helper, opposite measurement.
**Suggested fix:** move `cp.asarray(...)` inside `_gpu_call` and end it with `cp.asnumpy(...)` (matching the
votenrank tuner), and drop the per-iteration `.copy()` from the inference tuner's njit lambdas so both sides
measure the production call shape. Bump the salt to invalidate entries measured the old way.

### XGD-05 [P1] batch-rmse-gpu-returns-float64-cpu-returns-float32
**File:** metrics/_gpu_metrics.py:432
**Summary:** `compute_batch_rmse`'s GPU branch force-upcasts to float64 (`gpu_multiple_rmse_scores`:
`cp.asarray(actual, dtype=cp.float64)`) and returns a float64 array; the CPU reference computes
`np.sqrt(np.mean((yt - yp)**2.0, axis=0))` in the CALLER's dtype and returns float32 for float32 input. The
two backends disagree in both accumulation precision and result dtype.
**Failure scenario:** N=2,000,000, float32 `y_true`/`y_pred`, M=1. Measured (numpy reproduction of the two
accumulation regimes):
  - CPU (float32 accumulation): `0.49994543`, dtype **float32**
  - GPU (float64 accumulation): `0.49994545`, dtype **float64**
  - absolute difference `1.33e-08`
Verified live that the CPU branch really returns float32:
`compute_batch_rmse(f32, f32, force_backend='cpu')` -> `float32 [0.10000001]`.
A reported RMSE and its dtype therefore depend on which backend fired. Rubric note: this is literally "a
reported metric differs by backend" (P0 by the letter) but the magnitude is ~1e-8 relative, so it is filed P1
-- the dtype flip is the part most likely to bite a downstream consumer.
**Evidence:** _gpu_metrics.py:216-217 (`cp.asarray(..., dtype=cp.float64)`), :436-439 (CPU reference, no
cast), plus the two runs above.
**Suggested fix:** cast in the CPU reference to match the GPU contract --
`np.sqrt(np.mean((yt.astype(np.float64) - yp.astype(np.float64)) ** 2.0, axis=0))` -- so both backends return
float64 computed in float64.

### XGD-06 [P2] strict-gpu-defeats-disable_gpu-via-memoised-probe
**File:** feature_selection/filters/_fe_gpu_strict.py:172 (with info_theory/_cmi_cuda.py:823)
**Summary:** `gpu_globally_disabled()` is read LIVE everywhere else, but `_fe_gpu_strict._cuda_usable()`
folds `MLFRAME_DISABLE_GPU` / `CUDA_VISIBLE_DEVICES=""` into a **process-lifetime memo**
(`_CUDA_USABLE_CACHE`). The module itself argues the env flag must be "read LIVE on every call ... a process
that toggled MLFRAME_FE_GPU_STRICT mid-run would freeze on the stale first value -> order-dependent dispatch"
-- and then caches the *other* two flags with exactly that hazard. Compounding it, `_should_use_cuda`
(_cmi_cuda.py:823-828) consults STRICT and `return True` **before** it ever reaches
`_cmi_cuda_ktc.cmi_use_cuda`, which is the function that checks `gpu_globally_disabled()` first and whose
comment states "The global GPU opt-out outranks both the tuning cache and STRICT mode".
**Failure scenario:** A process (test session, notebook, long-lived worker) that touches any strict-gated FE
dispatch before setting `MLFRAME_DISABLE_GPU=1` keeps `_CUDA_USABLE_CACHE=True`; every later CMI dispatch
still returns True from the STRICT branch and runs on the GPU despite the explicit opt-out. The reverse also
holds: setting the flag first and clearing it later leaves the GPU permanently off. _gpu_policy.py's own
docstring calls sites like this "a silent divergence -- no exception, just different backends".
**Evidence:** _fe_gpu_strict.py:165-190 (memo including the env short-circuits, docstring calling them
"start-of-process device gates"), :262 / :265 (both STRICT branches gated on `_cuda_usable()`),
_cmi_cuda.py:823-836 (STRICT returns True above the KTC call), _cmi_cuda_ktc.py:50-56 (opt-out first).
**Suggested fix:** in `_cuda_usable()` memoise only the pyutilz/numba device probe and evaluate
`gpu_globally_disabled()` live on each call; independently, hoist a `gpu_globally_disabled()` check to the top
of `_should_use_cuda`.

### XGD-07 [P2] forced-cupy-silently-downgraded-to-njit
**File:** feature_selection/filters/batch_pair_mi_gpu.py:473
**Summary:** `dispatch_batch_pair_mi(force_backend="cupy")` is honoured only when `_vram_ok`; when the VRAM
estimate says no, control drops straight through to `return batch_pair_mi_njit_prange(...), "njit"` with no
log line at all. This is the "`force_backend` escape hatch silently ignored below a size threshold" shape that
has already been found once in this codebase.
**Failure scenario:** A caller benchmarking or pinning `force_backend="cupy"` on a large frame gets the njit
kernel, and only the returned `backend_name` (which many call sites discard) says so. Contrast the adjacent
forced-CUDA branch (:457-472), which logs a WARNING on every downgrade, and the forced-cupy branch's own
`except` at :475, which does log.
**Evidence:** read batch_pair_mi_gpu.py:455-479. `force_backend == "cupy"` with `_vram_ok` False matches no
branch condition and falls to the shared njit return at :479.
**Suggested fix:** split the guard -- `elif force_backend == "cupy" and _CUPY_AVAIL:` then
`if not _vram_ok: logger.warning(...)` before the njit return, mirroring the CUDA branch's message.

---

## LEADS (read, plausible, not proven divergent here)

### XGD-08 [P3] cupy-reduction-order-claimed-bit-identical-to-sequential-loop
**File:** feature_selection/filters/_batch_pair_mi_cuda_kernels.py:387
**Summary:** `_mi_from_joint_counts_cupy`'s docstring asserts it is "Bit-identical to
`_mi_from_joint_counts`: same `sum jf*log(jf/(px*py))` reduction over the same iteration order". It is not
the same order: line 402 builds a full `terms` array and reduces it with a cupy tree reduction, while the CPU
twin (:434) accumulates `total += ...` sequentially. Both are float64, so the gap is last-ULP, but the claim
is stronger than the code supports and any downstream exact-tie comparison inherits it.
**Failure scenario:** Not constructed (needs a GPU run; machine under load). Would show as a ~1e-16 relative
difference in per-pair MI, capable of flipping an exact-equality tie-break in a downstream argmax.
**Evidence:** read :385-410 vs :418-436.
**Suggested fix:** weaken the docstring to "agrees to float64 rounding (different reduction order)", or make
the CPU twin reduce pairwise if bit-parity is actually depended upon.

### XGD-09 [P3] rmse-atomic-add-partials-are-run-to-run-nondeterministic
**File:** metrics/_gpu_metrics.py:190
**Summary:** The numba.cuda RMSE fast path accumulates via `cuda.atomic.add(partial, (blockIdx.x, j), d*d)`.
Within one block-row the ordering is nondeterministic, so the SAME GPU call on the SAME data returns a
slightly different RMSE run to run. Documented in the docstring as "~1e-15 jitter", but it means a reported
metric is not reproducible on GPU while it is on CPU.
**Failure scenario:** Repeated `compute_batch_rmse` on identical input, GPU backend; last-ULP variation.
Interacts with XGD-05: the GPU number is both a different value and a non-repeatable one.
**Evidence:** read :190-198 and :245-258.
**Suggested fix:** none needed if the jitter is accepted; if reproducibility matters, accumulate into shared
memory per block and do one atomic per block.

### XGD-10 [P3] hardcoded-npermutations-32-fanout-changes-early-stop-granularity
**File:** feature_selection/filters/gpu.py:513
**Summary:** `mi_direct_gpu` fans out to `mi_direct_gpu_batched` at a hardcoded `npermutations >= 32` with a
hardcoded `batch_size=64`, and the two branches differ in RESULT, not just speed: the batched branch checks
`nfailed >= max_failed` at batch granularity, so up to 63 extra permutations run before the short-circuit,
changing `confidence = 1 - nfailed/(_i+1)` and whether `original_mi` is zeroed. Honestly documented, and the
code even excludes `return_null_mean` from the fan-out for this reason -- but it is still a hardcoded
crossover where a KTC lookup is used elsewhere in the same file (:601-609 looks up `joint_hist_single_perm`
block_size from the cache).
**Failure scenario:** A caller at exactly `npermutations=32` with a tight `max_failed` sees a different
`confidence` than the same call at `npermutations=31`.
**Evidence:** read gpu.py:490-532 and :600-610.
**Suggested fix:** route the 32/64 pair through the same `get_kernel_tuning_cache()` lookup already used for
`block_size`, keeping 32/64 as the documented fallback.

### XGD-11 [P3] engineered-recipe-gpu-replay-f32-vs-cpu-fallback-f64
**File:** feature_selection/filters/engineered_recipes/_recipe_unary_binary_gpu.py:264
**Summary:** Under `MLFRAME_FE_VRAM_F32` the GPU recipe replay materialises engineered columns in float32
(`dt = cp.float32 if _vram_f32() else cp.float64`), while the numpy fallback taken on ANY cupy failure
(documented at :250-254: "Raises on a cupy runtime failure - the caller's try/except logs debug + falls back")
produces the operand's own dtype. So a transient GPU fault mid-`transform()` changes the VALUES of the
engineered columns produced from that point on, within one call.
**Failure scenario:** A recipe with `log`/`div` legs whose `smart_log` shift is data-dependent: replayed in
f32 on device, f64 on the fallback. Not measured (needs the GPU plus the env flag set).
**Evidence:** read :64-72 and :250-268.
**Suggested fix:** have the numpy fallback cast its output to the dtype the GPU path would have used when
`_vram_f32()` is on, so a mid-call fallback does not change the column's precision.

### XGD-12 [P3] mi_direct_gpu-permutes-a-caller-owned-device-buffer-in-place
**File:** feature_selection/filters/gpu.py:643
**Summary:** `classes_y_safe[:] = classes_y_safe[cp.argsort(_shuf_rng.random(_shuf_n))]` shuffles cumulatively
and never restores the buffer. When the caller supplied `classes_y_safe` (the pre-warmed-device-buffer path --
evaluation.py:507, _confirm_predictor.py:479), the caller's cached device array is left permanently permuted
relative to `classes_x`. It is harmless for the permutation null itself (a composition of random permutations
is still a random permutation) and `original_mi` is computed from the HOST `classes_y`, so I could not
construct a divergence -- but any future consumer that reads that shared buffer as the TRUE label vector would
silently get scrambled labels.
**Failure scenario:** Not reproducible against current consumers; latent.
**Evidence:** read gpu.py:576-582 (buffer adoption) and :643 (in-place cumulative shuffle).
**Suggested fix:** shuffle into a scratch buffer, or restore the caller-supplied buffer before returning.

---

## Summary

| ID | Sev | File:line | Disagreement |
|----|-----|-----------|--------------|
| XGD-01 | P0 | feature_selection/filters/evaluation.py:501 | GPU relevance null unseeded while both CPU branches pass `base_seed`; selection non-deterministic on GPU hosts |
| XGD-02 | P0 | feature_selection/filters/permutation.py:731 | `mi_direct`'s GPU fastpath drops the caller's `base_seed`; seeded on CPU, unseeded on GPU |
| XGD-03 | P0 | feature_selection/filters/discretization/__init__.py:939 | Quantile bin edges exact vs random-subsample depending on transient free VRAM; changes discretized codes and thus selection |
| XGD-04 | P1 | calibration/_ktc_dispatch.py:101, inference/_ktc_dispatch.py:52 | Persisted crossover measured on GPU-resident data while production pays H2D+D2H; CPU side additionally burdened with a `.copy()` |
| XGD-05 | P1 | metrics/_gpu_metrics.py:432 | RMSE: GPU float64 / CPU float32 accumulation AND dtype; measured 1.33e-08 apart, float64 vs float32 return |
| XGD-06 | P2 | feature_selection/filters/_fe_gpu_strict.py:172 | `MLFRAME_DISABLE_GPU` memoised for process life while read live elsewhere; STRICT returns True above the opt-out check |
| XGD-07 | P2 | feature_selection/filters/batch_pair_mi_gpu.py:473 | `force_backend="cupy"` silently downgraded to njit when the VRAM estimate fails, with no log |
| XGD-08 | P3 | feature_selection/filters/_batch_pair_mi_cuda_kernels.py:387 | "Bit-identical" claim over a tree reduction vs a sequential loop |
| XGD-09 | P3 | metrics/_gpu_metrics.py:190 | float64 `cuda.atomic.add` makes GPU RMSE non-repeatable run to run |
| XGD-10 | P3 | feature_selection/filters/gpu.py:513 | Hardcoded `npermutations>=32` / `batch_size=64` fan-out changes early-stop granularity, hence `confidence` |
| XGD-11 | P3 | feature_selection/filters/engineered_recipes/_recipe_unary_binary_gpu.py:264 | Mid-call GPU-to-numpy fallback changes engineered-column precision under `MLFRAME_FE_VRAM_F32` |
| XGD-12 | P3 | feature_selection/filters/gpu.py:643 | Caller-supplied device label buffer left permanently permuted |

Checked and found CLEAN (recorded so they are not re-audited):
`_batch_mi_noise_gate_tuning.py` (KTC-backed, memory-aware grid, bit-identical variants);
`_cmi_cuda_ktc.py` (opt-out ordered above STRICT and the cache);
`batch_pair_usability_corr_gpu.py` (float64 accumulator on both sides; the DISABLE_GPU-over-`force_backend`
override is deliberate and documented); `votenrank/_confidence_gated_blend_ktc_dispatch.py` (tuner times the
real end-to-end call, ending in `cp.asnumpy`); `metrics/_gpu_metrics._resolve_backend` (KTC-driven; the
100k / M=5 constants are the documented fallback, not the live threshold);
`_fe_resident_operands.resident_operand` (content-hash + device-id key, no stale aliasing);
`_discretization_cuda._discretize_quantile_rawkernel` (explicit float64 cast before the `const double*`
kernel; NaN lands in the same top bin as `searchsorted`); `gpu_multiple_pr_auc_scores` (reversed-stable
argsort is AP-invariant because precision is sampled only at tie-run boundaries).
