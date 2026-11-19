# `_oof_warp_fe` family (`_oof_warp_fe`, `_oof_warp_service`, `_oof_warp_gpu`): scan of the out-of-fold warp candidates

Written 2026-11-17. The family fits, per numeric column, the cross-fitted warp `E[rank(y) | x]` on the scan rows and compares its held-out MI with the raw column's. Until now the scan
ran on the host only (one column per pool thread, numba kernels of `_oof_warp_service` plus numpy quantiles and the pointwise MI of `_fe_gain_stats`); `_oof_warp_gpu` scores a block of
columns on the device and the host service then fits only the accepted candidates, so an emitted column and its recipe are unchanged to the bit.

Box: GTX 1050 Ti (4 GB), 8 logical cores, one process, card at 65-68 C and 1708-1721 MHz before each measurement, nothing else running.

## Host (cProfile and wall)

`warp_candidates` over the 102 numeric columns of the 100000 x 100 benchmark frame, host pool (`device=False`), warm: 0.62-0.63 s wall on eight threads. cProfile of the calling
thread shows only the pool joins (`_thread.lock.acquire`, 107 results); the all-thread profile of a whole fit attributes 1.81 s of self time to the per-column body `_one_column`
(107 calls), 0.87 s to `_fit_core`, 0.36 s to `_bin_codes` and about 2 s of `np.quantile` partitions shared with the other scan families (cProfile-inflated thread-seconds).

Device scan, same frame, warm: 0.285-0.30 s wall. cProfile (7785 calls): 0.148 s in 109 `cupy.sort` calls (the first sort of the block is one sort per column, the policy of
`_device_sort_safe` for vectors of 80000 values and more), 0.053 s in 7 `cupy.bincount` calls, 0.030 s in `fe_stack_rows` (the host copy of the columns into contiguous rows), 0.006 s
in the two result copies.

Sweep rows (`_sweep_inputs`, 40 columns, min of 5): 5000 rows host 99 ms / device 30 ms; 30000 rows 132 / 57 ms; 100000 rows 286 / 150 ms.

## Device (nvprof)

`nvprof --print-gpu-summary`, six scans of the 102-column block (one warm-up and five timed), per scan:

- kernel launches: 1917, kernel time 211 ms. The launches are dominated by the per-column sorts of the first pass (cub merge and radix sort stages, about 1700 launches); the scan's
  own kernels are few: `oof_warp_interp` 1 launch / 21.2 ms, `oof_warp_codes` 3 launches / 5.1 ms, `cupy_bincount_kernel` 5 launches / 15.9 ms,
  `cupy_bincount_with_weight_kernel` 2 launches / 7.4 ms; the two half-block sorts (edges of the held-out codes) are one batched sort each.
- host-to-device copies: 145 calls in six scans (the columns come from the resident operand cache after the first scan; the target, the fold ids and the class codes are small).
- device-to-host copies: two per scan (the usable flags and the `(targets, columns, 5)` statistics, 4.2 kB for 102 columns).

`nvprof --metrics achieved_occupancy,gld_efficiency,gst_efficiency` (an eight-column block of 100000 rows: the full block replays its 1917 launches per metric and did not finish
in 14 minutes), averages per kernel:

| kernel | achieved occupancy | global load efficiency | global store efficiency |
|---|---|---|---|
| `oof_warp_codes` | 0.92 | 53.8 % | 100 % |
| `oof_warp_interp` | 0.70 | 30.2 % | 86.8 % |
| `cupy_bincount_kernel` | 0.86 | 100 % | 0 % (atomic adds) |
| `cupy_bincount_with_weight_kernel` | 0.85 | 100 % | 0 % (atomic adds) |
| `cupy_take` (edge gathers, tiny) | 0.33 | 50.3 % | 100 % |

The load efficiency of the two own kernels is that of a per-row table read: `oof_warp_codes` searches its column's 19 edges (data-dependent positions inside one 152-byte row),
`oof_warp_interp` reads up to 20 bins of three tables of the row's (column, fold). The row operands themselves are read in order.

Whole-GPU throughput: 10.2 million (column, row) cells scored per scan; 211 ms of kernel time inside a 0.29 s call is a utilization of about 73 % of the card for the duration of the
call, the rest being the host copy of the columns (30 ms) and launch overhead of the per-column sorts.

## Equivalence and dispatch

- Edges: `_row_quantiles` equals `np.quantile` (linear) to the bit, including positions that are whole numbers (regression test `test_row_quantiles_are_numpys_to_the_bit`; a
  position off by one rounding had moved one row of a 29406-value column to the next bin and its warp table with it: the sweep's equivalence check caught it).
- Statistics: device against host at 5000 / 12345 / 29406 / 30000 / 40001 / 77777 / 100000 / 250000 rows: at most 6e-13 relative (double sums by atomic adds).
- Tests: `tests/feature_selection/gpu/test_oof_warp_device_scan.py` (missing and infinite values, a nominal-like and a constant column, heavy tails, a row sample, one-vs-rest targets,
  the host fit of a device-scored candidate, the family's output through both routes).
- Kernel-tuning entry `fe_oof_warp_scan_device` (5000 / 30000 / 100000 rows, equivalence 1e-9, three repeats), swept on this box: device at every size.
  `MLFRAME_OOF_WARP_DEVICE=0|1` forces a side.
- Column blocks: `run_column_blocks`, block width from the room of the calling thread's pool (96 bytes per cell at the peak), halved on an out-of-memory block.

## In the fit (100000 x 100, all families, STRICT)

Alternating `MLFRAME_OOF_WARP_DEVICE=0` and `1` in one session: wall 44.1 / 44.2 s (host) and 44.2 / 44.7 s (device), same seven selected features. The lane that holds the family
(`mid_b`) went from 11.45 / 11.60 s to 10.82 / 10.86 s, but the two other lanes of the section (11.3-11.7 s) set its wall, so the fit's wall does not move until the lanes are rebalanced.

## Open

- The first sort as one batched launch (a key of column id and value, as the group-distance twin does) would remove about 1700 launches per scan.
- `oof_warp_interp` walks the 20 bins of the row's table in double precision; a table without empty bins (the common case) could take the bin from the fit code already computed.
