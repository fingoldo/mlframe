# Every-family MRMR fit, GPU, 100000 x 100 (GTX 1050 Ti, Pascal sm_61, 6 SMs, 4 GB)

Measured 2026-11-01 with `nsys 2022.4.2` (`--trace=cuda --sample=none --cpuctxsw=none --delay=150 --duration=25`), the warm main fit, GPU flags of `family_profile --mode gpu`
(`MLFRAME_FE_GPU_STRICT=1 MLFRAME_CMI_GPU=1 MLFRAME_FE_VRAM_F32=1 MLFRAME_FE_GPU_DISCRETIZE=1 MLFRAME_FE_GPU_BINNING=1`). The whole-fit trace is not usable: 10 million events for a 30000 x 40 fit, a 1.8 GB SQLite export that
filled the drive. Use a timed window.

## Whole-GPU picture of a 25 s window

| quantity | value |
|---|---|
| GPU kernel time | about 3.2 s (about 13% of the window) |
| CUDA API time | about 5.5 s: `cudaMemcpyAsync` 2.7 s (6375 calls), `cuLaunchKernel` 1.6 s (67811 calls), `cudaStreamSynchronize` 0.6 s, `cudaMemsetAsync` 0.25 s |
| launches | about 80000 in 25 s (67811 `cuLaunchKernel` + 12966 `cudaLaunchKernel`), median kernel 2-30 us for the cub / cupy_take / memset families |
| memory operations | HtoD 87 ms (1721 copies), memset 19 ms (13630), DtoH 16 ms (1889), DtoD 6 ms (2765) |

The rest of the window is host work (numpy / numba CPU kernels, pandas, python glue). The device is idle most of the time, so the wall is bound by host compute, launch overhead and synchronisation, not by kernel speed.

## Kernels (share of GPU kernel time)

| kernel | share | instances | grid x block | note |
|---|---|---|---|---|
| `mi_hist_split_f32` | 44% | 40 | 1016x2 x 256 | 35 ms avg; the plug-in MI histogram of the resident candidate batches |
| `fold_cell_centred` / `fold_cell_sums` | 6% / 3% | 282 each | 24 x 256 | binned numeric aggregation moments, 24 blocks < 4 SMs x waves: under-filled |
| `batched_joint_hist2_i32` | 5-6% each | 2 each | 108000+ x 256 | pair joint histograms, 75-100 ms each |
| `radix_select_f32_v3` | 3% | 2 | 1016 x 768 | quantile edges |
| cub partition / scan kernels | 2% + 1% + 0.5% | 4988 / 1378 / 932 | 1-14 blocks | thousands of tiny launches |
| `binagg_oof_gather` | 0.4% | 1128 | 782 x 128 | one launch per column |
| `cupy_take` | 0.2% | 5498 | 1 x 11 | 5498 launches of a 11-thread kernel |

## Fixes made from this profile

* The cupy pool was capped at 60% of the card while 0.75 GB belonged to the display and other applications; at the cap the device showed under 1 GB free and the 1 GB dispatch cushion declined every later
  GPU dispatch to the CPU (9+ declines in a fit). The pool cap is now bounded by the memory free at install, and a caller-supplied probe reclaims the pool's cached-free blocks before declining (0 declines).
* `score_candidates_by_cmi` binned every candidate on the device at once (1.5 GB allocation) and fell back to the per-candidate CPU loop: now bins and scores in blocks sized from free VRAM.
* The device honest-null pipeline declined every spec with a NaN column: it now applies the host kernel's NaN code.

## Open items (next kernels / fusions)

1. Launch count: about 3200 launches per second; merge the per-column launches (`binagg_oof_gather`, `cupy_take`, cub partitions) into batched kernels.
2. `fold_cell_*` grids of 24 blocks on a 6-SM device.
3. CPU-only stages with no device twin: surrogate LightGBM, honest-null host redo, DCD joint entropies, grouped-quantile / group-distance / composite-group-agg statistics, bootstrap_rank_mi.
4. Host round trips: `cudaMemcpyAsync` is 2.7 s of API time in the window (pageable copies and syncs).

## Host-side sampling, 100000 x 100, every family (no profiler, 0.2 s sampler)

Fit walls 221-246 s in one harness (noise range 230-400 s). Stage seconds by innermost stage frame, then the self time inside the largest stages:

| stage | s | what the main thread is doing |
|---|---|---|
| `_fit_body_run_fit_impl` | 14-17 | spread: MDLP edges on the pool 1.8, `_pair_collisions` 1.4, `build_honest_spec` 2.0 (now pooled), rest at 0.4 or less |
| `honest_null_columns_device` | 13.6 | 10.2 waiting for the k2 kernel (random label-bit gather), host prep 2.4 |
| `score_prospective_pairs` | 10.6-11.6 | spread: usability-corr warp kernel 1.2, conditional gate 2.2, pair perm-null admission 2.0, CMI joint entropies 0.8 |
| `apply_surrogate_gbm_seeder` | 10.0 | 8.2 in two CPU LightGBM fit sets (fastest exact form; concurrent fits measured slower) |
| `generate_univariate_basis_features` | 6.8-7.4 | flat, every frame 0.7 s or less |
| `_stage_binned_numeric_agg` | 7.0-7.6 | flat, every frame 0.8 s or less |

Fixed from this sampling: CMI chunk width remembered across calls (`score_candidates_by_cmi` 13.6 -> 4.8 s), grouped-quantile binning on the pool, one content hash per composite-agg column, balanced
triple-MI kernel loop, root of the chosen LOF neighbours only, shared conditional-gate code matrix (int16), honest-spec edges on the pool.

Checked and left alone (no dominant frame or no safe saving): `_gate_base_columns` probes (pooled), routing cells (pooled), bootstrap tables (pooled), `bootstrap_rank_mi` (radix walk), pair-admission column
copies (a cache would pin the data matrix on the estimator), honest-null host prep overlap (about 0.6 s).

Later fixes from the same sampling: OOF-warp few-values screen without the full sort, grouped-quantile global spreads from the sorted column, composite-agg non-finite replacement on the group values before the
gather, adaptive-column relevance ranks by inverse permutation. Same-harness walls after them: 219.7-223.6 s (before them 227-246 s; the harness noise is about 10%, so treat the difference as indicative).
A heavy-tail memo for the routing cells was tried and reverted: 18 -> 3 detector calls per column, column time unchanged (the 7.57 s seen in a call counter was thread-summed, not wall time).

Fixes after the previous update, each from a self-time frame of the sampler: dCor columns with a one-pass kernel (no 1000 x 1000 matrix per column), batched KSG noise drawn once, HSIC added to the batched scorers of
the ensemble / auto scorers, basis-routing Pearson without `np.corrcoef`, CMI candidate matrix filled by contiguous rows (`score_candidates_by_cmi` back to its 4.6 s), group percentile rank with one binary search for
untied values. Same-harness walls (stage_sample, 100000 x 100, every family): 235.8 s at the start of the sampling, 210-217 s with the fixes (harness noise about 10%).

What is left is kernel-bound or CPU-only: honest-null k2 wait (about 10 s), surrogate LightGBM (8 s), device pair MI and conditional permutation null waits, MDLP edges on the pool; no stage has a frame above 1 s that has a cheaper
equivalent. Further reduction needs a faster GPU than the 6-SM 1050 Ti, a different surrogate model, or a quality trade.

## Session of 2026-11-01: where the 100000 x 100 fit stands (every family except the LightGBM surrogate seeder)

Wall in the same harness: 235.8 s (start of the sampling) -> 205-217 s (host fixes) -> 154-169 s on a quiet machine after the device ports below (machine noise about 10%, and the box was quieter in the last runs, so
compare walls only inside one process). nvprof over a whole fit (165 s): the GPU kernels add up to about 25 s, i.e. the card is idle about 85% of the time; `root_split_stats_k2` (honest null) is 10.5 s of those
(41%), then `batch_pair_usability_corr` 1.2 s, `radix_select` 1.2 + 1.05 s, `mi_hist_split` 0.96 s, `batched_joint_hist2` 0.95 s, `pair_mi_shared_fused` 0.71 s, 9394 host-to-device copies 0.72 s.

Device ports made: conditional-routing cells, adaptive-degree cells, univariate-generator routing in one batch, the four non-polynomial bases (fourier, rbf, sigmoid, pade; these were unreachable before), LOF (11x),
the fused cell-validity pass and the column-tiled |corr|. Host fixes: Newton gate kernel, dCor / KSG / HSIC scorers, group percentile ranks from the sort permutation, group broadcast scrub, composite z / ratio in one
pass, CMI chunk memo, candidate matrix filled by rows, honest-spec edges on a pool. Tried and rejected, with numbers in their notes: the shared-memory honest-null kernel (1.4-1.8x slower), the device bootstrap
rank-MI walk (0.56 s against 0.24 s host), threads over the independent stage cascades (151-155 s against 154 s: the stage time is GIL-bound host work).

Looked at and left alone (no frame above 0.8 s, or a kernel already at its measured limit): `_stage_binned_numeric_agg` (7.4 s, flat, mostly device waits), parity scan (`_parity_mi_grid_njit`, 1.8 s, a parallel
kernel whose cupy twin was measured and rejected), `screen_predictors` (3.2 s inside the serial Fleuret confirmation, the core MRMR njit kernels), wavelet and group-distance (flat).
