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
