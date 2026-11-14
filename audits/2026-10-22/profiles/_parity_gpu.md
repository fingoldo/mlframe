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

## Family record for the contract: tuner, counters, utilization (2026-11-14, parent `_parity_fe`)
`use_device_parity` no longer reads only a fixed row threshold: the numba prange scan (`_parity_mi_grid_njit` per column) and `parity_mi_grids_gpu` are swept per host over the row count in the kernel-tuning entry `parity_scan_backend` (16 timestamp-like double columns, two classes, a period-7 parity pattern), with `PARITY_DEVICE_MIN_ROWS` as the pre-sweep floor and `MLFRAME_PARITY_DEVICE=0|1` still forcing a path. The kernel text uses `{ACC}` for its all-wide types (floor of a timestamp needs the wide type, as the dtype note in the kernel says); columns go up in column blocks of 32.

Measured on the sweep inputs (16 columns, numba against the whole device call with uploads), elevated process: 2,000 rows 3.4 / 3.7 ms, 5,000 rows 6.6 / 4.1 ms, 20,000 rows 19.1 / 6.7 ms, 100,000 rows 96.8 / 19.9 ms (4.9x). Agreement 3.5e-18 absolute. cProfile of the device call at 100,000 rows: the host side is the column stack (2 ms) and the uploads (1 ms); the rest is the kernel wait.

Hardware counters (nvprof `--metrics`, 100,000 rows x 16 columns): `parity_mi_grid` launches per call: 1 per column block (one here), 13.6 ms of the 15.6 ms traced GPU time (87 percent; the uploads 12.8 percent); achieved occupancy 21.6 percent (100 plus registers for the 8 periods x 4 class counters per thread), double-precision efficiency 18.6, DRAM read throughput 14.2 GB/s (13 percent of the card's 112 GB/s peak bandwidth), global load efficiency 99.8 percent. Utilization: neither memory nor arithmetic is saturated; the low occupancy is the price of keeping 32 counters in registers (the design choice of the kernel); lowering the periods per block was not tried.
