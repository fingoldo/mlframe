# MRMR fit-path performance critique (READ-ONLY)

Repo: `.../scratchpad/master_wt/src/mlframe/feature_selection/filters`

## Context / honesty note

This path is already heavily perf-tuned (the code carries dozens of `bench-attempt-rejected` notes, fused RawKernels, hoisted permutation-invariants, pruned-kernel twins like `joint_freqs_2var`/`joint_entropy_2var`, KTC-backed dispatch). Most *remaining* waste is either (a) copies on the hot CPU screen path, or (b) inside GPU/STRICT paths that are **gated/opt-in and not on the default wall**. I flag both, but call out which are default-path (real wall impact today) vs gated (latent). No padding: the genuinely default-hot findings are #1, #2, #3.

---

## Ranked findings

### 1. `data_copy = factors_data.copy()` — full-matrix copy on every screen call [DEFAULT PATH]
- **File:** `_screen_predictors.py:530`
- **Waste:** Copies the ENTIRE (possibly subsampled) factors matrix once per `screen_predictors` call. Because the outer FE loop (`_fit_impl_core.py:6499` `while True`) re-invokes `screen_predictors` every FE iteration + a confirm-rescreen, this copy is paid `num_fs_steps + 1` times per fit. It is kept as the pristine reference for in-place column shuffling (`extra_x_shuffling`) and DCD swap re-snapshots (`_screen_dcd_swap.py:111` snapshots it AGAIN).
- **Impact:** On a 100GB-class frame this is the single most expensive line here (CLAUDE.md explicitly forbids whole-frame copies on hot paths). Even at moderate n it is an O(n·p) alloc+memcpy per screen.
- **Severity:** P1 (measure-first; verify whether all consumers only READ it — a mutate-and-restore of just the shuffled column would avoid the full copy). Note the subsample slice at `:522` (`factors_data[_sidx]`) already produces a fresh array, so on the subsampled path `data_copy` is a copy-of-a-copy.
- **Fix:** If the only mutation is per-column shuffle during permutation testing, hold the pristine column on demand (`col = data[:, j].copy(); shuffle; ... ; data[:, j] = col`) instead of copying the whole matrix. Gate the full copy on `extra_x_shuffling or dcd_enable`.

### 2. `cpu_fe_batch_mi` forces a full float64 contiguous copy of the candidate matrix per FE step [DEFAULT PATH]
- **File:** `_fe_cpu_batch.py:64` (`X = np.ascontiguousarray(X_cands, dtype=np.float64)`), plus `:75` full `np.isfinite(X).all(axis=0)` scan and `:80` a SECOND copy `np.ascontiguousarray(X[:, dense])` when any column is non-dense.
- **Waste:** The FE candidate matrix (n × K) is upcast+copied to contiguous float64 every batch even when it is already f64/contiguous; a partial-NaN column then triggers a further dense-subset copy. This is the CPU FE-scoring backend (default on unprofiled hosts per `_fe_batch_dispatch.py:60`).
- **Impact:** One (n×K) copy + one full finite-scan per FE step. Moderate at typical K; large when K is wide.
- **Severity:** P2.
- **Fix:** `np.ascontiguousarray(..., dtype=np.float64)` is a no-op when already f64+C-contig, but the explicit dtype forces a copy on any non-f64 input — check `X_cands.dtype`/`.flags` first and skip. Cache `finite_all` if the same matrix is re-scored.

### 3. Redundant target re-discretization across ~15 FE-family blocks [DEFAULT-ish PATH]
- **File:** `_fit_impl_core.py` — `_y_for_hybrid` (476/479), `_y_for_extra` (597), `_y_for_triplet` (902), `_y_for_quad` (1053), `_y_for_aa` (1170), `_y_for_adapt` (1282), `_y_for_route` (1377), `_y_for_diff` (1471), `_y_for_cb` (1575), `_y_for_boot` (1680), `_y_for_tg` (1786), `_y_for_ksg` (1903), `_y_for_mig` (2890), `_y_for_cmi` (2978), `_y_for_ga` (3884), `_y_for_cga` (3962), `_y_for_eng_mi` (5336-5343).
- **Waste:** Each FE-family gate independently recomputes the SAME invariant discretization of the target — `pd.qcut(y, q=10, labels=False, duplicates="drop").astype(int64)` (O(n log n) sort each) or an int64 cast — from the same source `_y_np`. When several FE families are enabled, the identical binned-y is recomputed N times per FE step per outer loop iteration.
- **Impact:** Each `qcut` is an O(n log n) sort + pandas overhead; ~15 potential repeats. Real cost scales with how many families the config enables (many default-off, so typical exposure is a handful).
- **Severity:** P2.
- **Fix:** Compute `_y_discrete = qcut(y,10)` / int-cast ONCE per fit (cache keyed on `_y_np` identity + shape) before the family blocks and reuse. Cheap-now win; no numeric change.

### 4. STRICT `mi_from_codes` RawKernel recomputes the y-marginal inside the x-loop [GATED]
- **File:** `_fe_batched_mi.py:104-116` (kernel `ry` recompute at 113)
- **Waste:** For each x-bin `xx`, the inner loop recomputes the y-column marginal `ry = Σ_xx2 sh[xx2*Ky+yy]` afresh for every `yy` — O(Kx²·Ky) instead of precomputing the Ky column sums once (O(Kx·Ky)).
- **Impact:** Small in absolute terms (Kx,Ky ≈ nbins ≈ 10 → ~10x redundant on a tiny table) and the kernel is behind `MLFRAME_FE_GPU_STRICT`, but it is a clean single-thread reduction the block already serializes on `tid==0`.
- **Severity:** P2 (gated).
- **Fix:** Precompute `py[yy]` column sums once into a small shared/register array before the `xx` loop; reuse. Bit-identical by construction.

### 5. `batched_quantile_bin_gpu` per-column Python launch loop [GATED] (agent-found)
- **File:** `_fe_batched_mi.py:613-621`
- **Waste:** After a good batched `cp.percentile`, drops into `for k in range(K)` issuing per-column `cp.unique` + masked `cp.searchsorted` + strided column scatter — 2-4 `cuLaunchKernel`/column, the launch-count pattern the rest of the file was rewritten to kill.
- **Severity:** P1 in the kernel, P2 real (module is "imported by nothing in production yet"; STRICT-gated).
- **Fix:** Vectorize the (nbins+1)-value dedup across columns + single batched in-kernel `searchsorted` over the (nbins-1, K) interior edges (the `mi_from_values` kernel already exists in-file).

### 6. `batch_pair_mi_cupy` per-pair blocking `.get()` sync [GATED/rare] (agent-found)
- **File:** `batch_pair_mi_gpu.py:376-402` (per-pair `float(mi.get())`)
- **Waste:** `for p in range(n_pairs)` with ~6 launches/pair AND a per-pair D2H `.get()` that drains the queue every iteration. Dispatcher avoids this path (`CUPY_MIN_PAIRS=200`; cupy "never beat numba.cuda"), so exposure is low.
- **Severity:** P2.
- **Fix:** Stage MIs on-device, one `.get()` after the loop (as `gpu.py:mi_direct_gpu_batched` does), or route to `compute_joint_hist_multi_pair_cuda`.

### 7. `mi_direct_gpu` legacy per-permutation argsort + `totals.get()` sync [GATED/rare] (agent-found)
- **File:** `gpu.py:974-996` (argsort-shuffle 975, `totals.get()[0]` 988)
- **Waste:** Per-permutation full `cp.argsort(random(n))` + `joint_counts.fill(0)` + per-iter cross-device sync. Only runs when `npermutations < 32` or `return_null_mean=True` (the ≥32 path already fans out to the batched twin).
- **Severity:** P2.
- **Fix:** Batched kernel that accumulates the null sum on-device; one `.get()` for the null-mean caller.

### 8. `batch_pair_mi_cuda` host-side Python guard scan over all pairs (agent-found)
- **File:** `batch_pair_mi_gpu.py:275-292`
- **Waste:** Python `for a,b in zip(...)` computing `nbins[a]*nbins[b]` just to find `max_joint`/`min_nb`.
- **Severity:** P2 (host, small n_pairs).
- **Fix:** Vectorize: `nb = nbins[pair_a]*nbins[pair_b]; max_joint = int(nb.max())` etc.

### 9. `binned_mi_from_codes_gpu` double contiguity/int64 upcast on host path (agent-found)
- **File:** `_fe_batched_mi.py:563-566`
- **Waste:** `cp.asarray(np.ascontiguousarray(code_cols).astype(int64))` then `cp.ascontiguousarray(C.astype(cp.int64, copy=False))` — host int64 materialization + redundant contiguity pass per candidate batch.
- **Severity:** P2.
- **Fix:** Single contiguity enforcement; prefer resident int32 codes (kernel indexes `codes[i*K+c]`; int32 halves H2D bytes).

### 10. `_fe_batched_mi.py:518` split-N trigger is a raw hardcode with no KTC lookup (agent-found)
- **Waste:** `if K < 48 and n >= 262144` is GTX-1050-Ti-tuned; unlike sibling dispatches it does NOT consult `kernel_tuning_cache`. Wrong crossover on other GPUs.
- **Severity:** P2.
- **Fix:** Route the split-vs-single decision through KTC (axes n × K), keep constants as fallback.

### 11. `fused_propensity` recomputes what `second_moment_propensity` already derived (minor)
- **File:** `_fe_interaction_prerank.py:227-240`
- **Waste:** Calls `second_moment_propensity` (which builds `V`, `V2`, standardized stats, one-hot classes), then re-derives `V`, `y_arr`, `yf`, `classes`, and re-loops the marginal `|corr(x,1[y=c])|` channel — duplicating the nan_to_num + class setup.
- **Severity:** P3 (only on the `fused` criterion, small/moderate p by gate).
- **Fix:** Have `second_moment_propensity` optionally return its intermediates, or factor a shared prep so `fused` reuses `V/V2/yf/classes`.

---

## Optimization ideas worth benchmarking (measure-first)

1. **Eliminate the whole-frame `data_copy`** (finding #1) via per-column mutate-and-restore for shuffling; gate the full copy on `extra_x_shuffling or dcd_enable`. Expected: removes one O(n·p) copy per screen call — the biggest default-path lever at scale. A/B on a wide frame with `extra_x_shuffling=False`.
2. **Hoist a single `_y_discrete` cache** (finding #3) shared by all FE families. Bench a config with many families enabled (hybrid+extra+triplet+quad+ga) at n=100k; count `qcut` calls before/after via cProfile.
3. **Skip no-op float64 copies in `cpu_fe_batch_mi`** (finding #2): branch on `dtype/flags`. Bench with an already-f64 contiguous (n=100k, K=500) candidate matrix — should show a measurable drop in the FE-step tottime.
4. **Precompute y-marginals in `mi_from_codes`** (finding #4) and vectorize `batched_quantile_bin_gpu` (#5) TOGETHER, then re-run the STRICT F2 300k full-fit wall A/B (the file's own methodology) to confirm the launch-count reduction moves the wall before promoting STRICT toward default.
5. **KTC-ify the split-N crossover** (#10) and re-sweep on a modern GPU — the hardcoded 262144 crossover likely leaves occupancy on Ampere/Ada.

## Not waste (verified)
- `_batch_kernels.py`: permutation-invariants hoisted out of shuffle loops (`batch_pair_mi_perm_batched`), fused densify+relevance passes — no in-loop recompute.
- `info_theory/_class_encoding.py`: `merge_vars` has documented rejected bincount/prange attempts; the pruned twins `joint_freqs_2var`/`joint_entropy_2var` (23x / 1.24x) already handle the discard-part callers per the "audit hot kernels" rule.
- `_mrmr_fe_step_helpers.py`: triple/pair sweeps use vectorized `np.fromiter` + `batch_triple_mi_prange`, not Python loops.
- No joblib-threading-over-GPU sleep-contention found in the four GPU/batch files (kernel-init lock is a correctness guard).
