# Profile record: per-category order statistics of the K-fold target encoder (`_target_encoding_order_stats`, device twin `_order_stats_gpu`)

Host: GTX 1050 Ti (sm_61, 4 GiB), 8 logical cores, elevated process (hardware counters). Measured 2026-11-14, medians of 3 after a warm call. Workload: a skewed float target over 200 categories; all seven statistics (median, 10 percent trimmed mean, q10, q90, IQR, min, max).

## What was added
The host groups rows by category with `np.lexsort((y, category))` and one numba kernel (`_segment_order_stats`) sweeps the contiguous segments. `per_category_order_stats` now asks `order_stats_dispatch` first: `cupy.lexsort` makes the same grouping on the device and `segment_order_stats` (one 128-thread block per category) emits the same seven arrays: thread 0 takes the quantiles and extremes with numpy's `linear` interpolation expressions, the block sums the trimmed segment by a warp-shuffle reduction; empty categories and stats not asked for are NaN as on the host. The categories are one column block of the encoder's input (one device sort covers them all). Host or device is the kernel-tuning entry `order_stats_backend` (row count), with a 50,000-row floor before a sweep; a device fault falls back to the host with a throttled warning. The rare-cell fallback to the global value stays on the host.

## cProfile (device call, 500,000 rows)
The host side is three uploads (1.3 ms); the call is its device sort and kernel.

## nvprof (hardware counters)
`segment_order_stats` (launches per call: 1 for all categories; plus the 16 cub merge-sort passes and a gather of `cupy.lexsort`): the sort passes are 41 percent of the traced GPU time (10.1 ms over 16 launches), the uploads 5.4 percent, the statistic kernel itself well below 1 ms. Kernel counters: achieved occupancy 88.5 percent, double-precision efficiency 12.5, DRAM read throughput 56.3 GB/s (50 percent of the card's 112 GB/s peak bandwidth), global load efficiency 87.6 percent. Utilization: the call is bound by the device sort, not by the sweep.

## Wall clock (host `np.lexsort` + numba sweep against the device call with uploads)
| rows | host | device |
|---|---|---|
| 5,000 | 0.9 ms | 1.0 ms |
| 20,000 | 2.8 ms | 1.1 ms (2.5x) |
| 100,000 | 21.3 ms | 2.8 ms (7.6x) |
| 500,000 | 179.9 ms | 25.5 ms (7.1x) |

All seven arrays agree with the host to 7e-16 relative with NaNs in the same places; the encoder's output dict is the same with the device forced on (empty category, single-row category, ties, rare-cell fallback), and the test counts the device call (`tests/feature_selection/gpu/test_order_stats_device.py`).
