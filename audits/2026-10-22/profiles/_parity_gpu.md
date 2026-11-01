# _parity_gpu (floor-parity period grid scan, many columns per launch)

Kernel `parity_mi_grid` in `src/mlframe/feature_selection/filters/_parity_gpu.py`; used by `_parity_fe._scan_parity_columns` for the all-finite columns of `hybrid_parity_fe_with_recipes` (target with at most 4 classes). The host twin `_parity_mi_grid_njit` stays (columns with a NaN, more classes, `MLFRAME_PARITY_DEVICE=0`).

Why a new device attempt after the rejected one (2026-10-14: per-column cupy bincount per period, 0.030 s against 0.026 s at 1e5 rows): that version launched per period and per column. Here one block owns a column and 8 consecutive periods and strides the column once,
keeping 8 x 4 plain counters per thread (rows of parity 1 per class; the parity-0 counts are the class totals minus them), so a value and its class are loaded once for 8 periods and no shared histogram is hit per row.

Measured (GTX 1050 Ti, one process, host = the 8-core prange njit kernel):

| shape | device | host |
|---|---|---|
| 100000 rows x 100 columns, 2 classes | 132 ms | 2011 ms |
| 30000 rows x 40 columns, 4 classes | 29 ms | 231 ms |
| 5000 rows x 7 columns, 3 classes | 3 ms | 7 ms |

Values equal the host's to 1e-12 (observed 6e-19) and the chosen period is the same. Dispatch: `PARITY_DEVICE_MIN_ROWS` = 5000 rows (device already ahead there), `MLFRAME_PARITY_DEVICE=0|1` forces a path.

GPU profile (nvprof, 100 columns x 100000 rows, 4 launches of 32 columns, 21.7 ms each): achieved occupancy 0.25 (register-bound: 8 reciprocals and 32 counters per thread), multiprocessor activity 97.8%, global load efficiency 99.8%,
device memory utilization low, warp execution efficiency 92.4%. The kernel is bound by the double-precision `floor(x * inv)` (1/32 rate on this card), not by memory; a float pre-filter was not tried because the value must be exact on large timestamps.
Tests: `tests/feature_selection/fe/test_parity_gpu.py`.
