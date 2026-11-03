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

## Idle share of the cores and the card (target: both near zero)

Tool: `python -m mlframe.feature_selection._benchmarks.profiling.utilization 100000 100` (psutil per core and nvidia-smi, twice a second, one every-family fit; the surrogate seeder excluded).
Baseline of 2026-11-01 (fit 192.9 s under the sampler, 8 logical cores, GTX 1050 Ti):

| resource | busy on average | idle |
|---|---|---|
| CPU, all cores | 40.8% | about 59%; under a quarter of the cores busy in 12% of the samples |
| GPU | 21.1% | about 79%; the card under 10% in 62% of the samples |
| both idle together | | 6% of the samples |

Window means (CPU % / GPU %) run 36-50 / 6-21 for the whole fit except one window (140-160 s: 42 / 65). Reading: the cores are used less than half and the card a fifth, and the two are idle at DIFFERENT times,
so overlapping the host work with the device work is where the idle time can be recovered; threads cannot do it while the stage time is GIL-bound Python (stage cascades in threads: 151-155 s against 154 s), so
the order is (1) turn the GIL-bound host loops into nogil / parallel kernels so that all cores can work, (2) then feed the card from several threads (the opt-in stage concurrency, the device scheduler and the per-stage
VRAM budgets are in place for that).

## Utilization after the column pools (2026-11-01)

Same meter, 100000 x 100, every family, one process: fit wall 230.0 s (noisy machine). CPU busy 41.6% of all cores (baseline 40.8%), samples under 25% busy 8%; GPU busy 17.4% (baseline 21.1%), card under 10% in 67% of the samples;
both idle together 3% of the samples (baseline 6%).

Reading: the pools added since the baseline (wavelet, extra basis, group distance, oof warp, univariate basis, composite group agg) moved the mean CPU load by under one point, because the big single-core stages are
unchanged: honest null columns (17 s, 1.5 cores, device wait), the binned numeric agg (8.8 s, 1.0), score_candidates_by_cmi (6.6 s, 1.0). The gate stage (`hybrid_conditional_gate_fe_with_recipes`) is a per-column
device loop (median, where, batched MI each sync), so a host pool would only add contention. Next lever is overlapping those device waits with host work of the next stage (the concurrent stage mode, now correct,
gained 7%), not more host pools.

## Utilization after the honest-null v5, memo key and pair pools (2026-11-01)

Same meter, 100000 x 100, every family, one process: fit wall 176.1 s (230.0 s in the previous measurement of this note; the machine is noisy, so read it as a direction). CPU busy 45.3% of all cores (41.6% before),
samples under 25% busy 4% (8% before); GPU busy 20.3% (17.4% before), card under 10% in 61% of the samples (67% before); both idle together 3%.

Changes since the previous section: honest-null root split v5 (one float pass, 287 ms against 861 ms for v2 on 60 x 200 x 100000; the stage fell from 17.4 s to 11.2 s in the stage map), the univariate basis memo key hashed
with xxh3 (a 600-column call 1.39 s to 0.35 s), the composite group-agg tasks and the binned-agg linear-usability pairs on the edge-jobs pool, the univariate basis column pool.

## Stage map and hashing findings (2026-11-01, later run)

Stage map under the sampler at 100000 x 100, every family: wall 153.9 s (194.7 s in the previous map). Largest stages by main-thread seconds: `_fit_body_run_fit_impl` 12.2 (spread over the discretization 2.0 s on the edge pool, the engineered-duplicate scan about 2 s,
the honest-spec build 0.8 s), `honest_null_columns_device` 7.2 (was 17.4), `_stage_binned_numeric_agg` 6.0 (device-bound), `score_prospective_pairs` 5.8 (the fused observed-MI kernel), `score_candidates_by_cmi` 5.4 (waits on the device).

Whole-array content hashes were a hidden cost spread over many stages: blake2b runs at about 0.4 GB/s (0.19 s per 80 MB), xxh3 at 13 GB/s (0.006 s). Moved to the shared `_content_hasher.new_content_hasher` (xxh3, blake2b fallback): the univariate basis memo key,
the content memo, the accuracy-gate baseline key (rebuilt for every engineered sibling), the duplicate-transfer fingerprint, the fit-constant memmap key and the fit signature / fit cache key (three full-frame hashes per fit).
Also: `_row_center_finite` of the dedup (2.77 s to 0.16 s for 1800 x 100000) as one parallel njit pass.

Not parallelisable by a host pool (checked): `_stage_greedy_cmi` (sequential greedy rounds), `hybrid_orth_mi_fe` (one device batch), `score_candidates_by_cmi` (device wait), `fleuret_conditional_recheck` (early stop on shared caches).

## State after the device kernels of the 2026-11-01 series

Stage map under the sampler (100000 x 100, every family, one process): wall 162.2 s, no OutOfMemoryError, no family raised. Largest main-thread stages: `_fit_body_run_fit_impl` 13.6 s (2.7 cores), `honest_null_columns_device` 8.8 (the v5 root-split kernel is the bulk:
about 20x the 60 x 200 shape of the micro-benchmark), `_stage_binned_numeric_agg` 7.6 (device waits), `score_prospective_pairs` 6.6, `score_features_by_bootstrap_mi` 6.2 (4.2 cores), `score_features_by_auto_scorer_uplift` 5.6 (5.6 cores).

New device kernels of the series, each with a host twin, an equivalence test and a profile note under `audits/2026-10-22/profiles/`: `_parity_gpu` (15x), `_triple_mi_gpu` (6.5x), `_quantile_codes_gpu` (radix selection, 2.4x; the sort variant is rejected, kept), `_honest_null_k4` (v5 default, v4/v4f kept),
`_bootstrap_rank_mi_gpu` (default from 30000 rows after a pool-pressure fix). Host kernels given `nogil` so the column pools scale: Wasserstein, polynomial basis evaluators, digit extract, modular residue; the B-spline basis is now a compiled Cox-de Boor kernel (extra-basis scan 16.1 s to 10.9 s).

Method lessons recorded here because they cost time: (1) a micro-benchmark win is not a fit win until the full fit is grepped for `OutOfMemoryError` and `FE raised` (the device bootstrap walk passed its benchmark and broke the pool in the fit); (2) identical code gives 145 to 230 s on this machine, so
two-run A/B pairs are only direction, and the stage seconds of the sampler or the call time are the instruments for a 1 to 3 s change; (3) the pool wait of a main thread in a profile means the work is in the workers: scale the pool (1, 2, 4, 8 jobs) to see whether a GIL-held section is hiding there.
Still not parallel or device-ported: `fleuret_conditional_recheck` and `_stage_greedy_cmi` (sequential with early stop), the within-stratum permutation null of the triplet scorer (needs its own device RNG scheme), `score_candidates_by_cmi` (waits on the device).

## nsys window, 50k x 100 GPU fit (2026-11-03, after the binning / radix / Fourier changes)
`--trace=cuda --sample=none --cpuctxsw=none --delay=60 --duration=25` (after a 20000 x 30 warm-up fit), 1.6 MB report.
GPU kernels 2.08 s in 25 s (about 8% busy); CUDA API 2.67 s: cudaMemcpyAsync 1.40 s (2669), cudaStreamSynchronize 0.51 s (2109),
cuLaunchKernel 0.42 s (16444) + cudaLaunchKernel 0.07 s (4154), cudaMemsetAsync 0.14 s (4039).
Top kernels: batched_joint_hist2_i32 0.445 s (15), cub DeviceMergeSort 0.281 s (405) + 0.165 s (38), ent_nnz_1d 0.184 s (840),
null_mi_perms_bits 0.121 s (3), radix_select_interp_f64_v3 0.111 s (46), fold_cell_centred 0.097 s (352), radix_select_f32_v4 0.055 s (3).
Reading: in this window the fit is host-bound. The main-thread waits on device queues seen by stage_sample (pair analytic gate 2.0 s,
materialise uploads) are real, but most of the wall is host code; the remaining lever is moving host-only stages to the device, as on 2026-11-01.
Stage sampler (same day, leaves per stage): every stage's time is spread over 0.2-1.2 s leaves; no single leaf above 1.2 s outside pool waits.

## 100k x 100 all families, GPU, 2026-11-03 (after the binning / radix / Fourier / HSIC / dCor changes)
Two clean runs (no cProfile), same code: stage_sample 91.5 s; plain fit with mlframe DEBUG memory/fallback filter 101.1 s (2026-11-01: 237-272 s).
Selected (run 2): x4, oofwarp(x4), add(mul(cbrt(x0),cbrt(x1)),add(x2,abs(g))), add(log(x2),x0*x1__He1_He1), opat__g_x4_x2, add(sqrt(g),sign(x4)).
Run 1 logged two cupy thrust-allocator OutOfMemoryError tracebacks (365 MB against the 1.97 GB pool cap, "Exception ignored in cupy_malloc") right after
the fe_lof_device lookup; the fit completed. Run 2 had none, so the caller (a cupy sort / argsort that fell back) is not identified yet.
Main-thread leaves at 100k: edge-pool wait 4.4 s (extra-basis 2.0 of it), honest-null `_continuous_accept_block` D2H 1.6 s, grouped-quantile pool wait 1.4 s,
per-feature edges wait 1.4 s, bootstrap pool 1.2 s, pooled binning 1.2 s, pair analytic gate 1.2 s.

## GPU vs CPU selection quality, 100k x 100 all families (2026-11-03)
Same frame (family_scale._make(100000, 100, 0)), held-out frame _make(50000, 100, 1); each mode in its own process (D:/Temp/sel_quality.py), LightGBM
(400 trees, 31 leaves) and logistic regression on the transformed selected columns:
- GPU (111.3 s incl. transforms): x4, oofwarp(x4), add(mul(cbrt(x0),cbrt(x1)),add(x2,abs(g))), add(log(x2),x0*x1__He1_He1), opat__g_x4_x2, add(sqrt(g),sign(x4)):
  test AUC 0.9020 (LightGBM), 0.9028 (logistic).
- CPU (294.9 s): g, x4, x2, add(x0*x1__He1_He1,sin(sir__...__dir0)), x4__He3, x4__fourier3: test AUC 0.9018 (LightGBM), 0.8958 (logistic).
- All 107 raw numeric columns, LightGBM: 0.8918.
The selections differ (known since 2026-10-31) but the GPU one is not worse: equal under LightGBM, +0.007 under logistic regression.
