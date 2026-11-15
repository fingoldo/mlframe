# `_gain_stats_codes_gpu` profile record (2026-11-15)

Device coder of the gain statistics: `quantile_codes_right` codes a block of columns against numpy's own quantile edges in one launch per bin count. Consumers: `target_has_structure` (`_target_mix_gate`) and `pair_interaction_screen` (`_pair_residual_screen`). CPU twin: `_fe_gain_stats.quantile_codes` (njit edge count, `np.quantile` edges). Host: GTX 1050 Ti (sm_61), Windows, nothing else running during the timed lines.

## What it replaces

Per candidate column, `np.quantile` of the finite values and a `searchsorted` of the column against the edges, twice per column in the pair screen (10 and up to 60 bins) and once in the gate. At 50000 sampled rows and 100 columns one `target_has_structure` call was 2.5 s on the host; 300 `quantile_codes` calls were 1.37 s of it, plus 0.72 s of `np.unique` for the few-distinct check (now an njit scan with an early exit, CF11-13).

## cProfile of one `target_has_structure` call (100000 x 100 float32 frame, 50000 scanned rows)

| step | before (host, njit edge count) | after (device) |
|---|---|---|
| whole call | 1.35 s | 0.72 s (0.65 s without the profiler) |
| `quantile_codes_columns` (gate and screen, 2 calls) | about 0.9 s | 0.244 s |
| of which `device_sort` of the blocks | | 0.122 s |
| of which host to device and codes back (`get`) | | 0.037 s |

Before any change the call was 2.5 s. The rest of the call (0.22 s in the function itself: column extraction and the main-effect statistics, 0.33 s in the pair screen outside the coder) is host numpy and is not touched here.

## Kernel launches per call

One block of columns per call when it fits the pool (`pool_column_block`, 25 percent of the free pool): per bin count one `quantile_codes_right` launch (grid x = min(rows / threads, 8 x SMs), grid y = columns), plus the sort of the block (shared by all bin counts: 11 merge-sort kernel launches for the 100 columns) and a few elementwise kernels for the edges. The gate asks for one bin count, the screen for two.

## nvprof (`D:/Temp/exp/gsc_prof.py`, 100 columns x 50000 float32, bins 10 and 60, 4 calls)

| item | value |
|---|---|
| GPU time share | merge sort 78 percent (88 + 8 launches), `quantile_codes_right` 6.7 percent (8 launches, 2.26 ms average), copies 11 percent |
| `gld_efficiency` | 99.2 percent |
| `gst_efficiency` | 65.5 percent (one byte per code, the stores of a warp are 32 bytes) |
| `achieved_occupancy` | 0.937 |
| `sm_efficiency` | 98.5 percent |
| device memory read / write | 14.2 GB/s / 3.3 GB/s of the card's 112 GB/s: the kernel streams the column once and reads 9 or 59 edges from shared memory, so it is bound by the binary search (6 compares a value), not by the bus |

## Whole-GPU throughput and the sort

Block of 100 columns x 50000 rows, both bin counts: 92 ms on the device against 661 ms on the host (7.2x), codes identical. The sort is the cost. The radix order-statistics route of `_quantile_codes_gpu._order_statistics_f32` was measured against it (min of 4, synchronized): 18 ranks (10 bins) radix 18.6 ms against sort 52.5 ms; 118 ranks (60 bins, or 10 + 60) radix 79.5 / 80.8 ms against sort 47.7 / 61.2 ms. The consumers ask for 60 bins, where the sort wins, so the sort stays; a 10-bin-only caller would gain from the radix route (not wired: no such caller).

## Equivalence and tuner

Codes equal the host `quantile_codes` for float32 and float64, 2, 10 and 60 bins, continuous, heavily tied, constant, binary-like, heavy-tailed and non-finite columns; edges are bit-identical to `np.quantile` (difference taken in the data dtype, numpy's `t >= 0.5` branch). Dispatch by the kernel-tuning entry `gain_stats_codes_backend` (axis rows; the host-versus-device sweep compares the two code matrices exactly), fallback mark 20000 rows. Tests: `tests/feature_selection/gpu/test_gain_stats_codes_gpu.py`.
