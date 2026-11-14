# Profile record: joint-synergy combo screen (`_fe_synergy_screen`, device twin `_synergy_combo_gpu`)

Host: GTX 1050 Ti (sm_61, 4 GiB), 8 logical cores, elevated process (hardware counters). Measured 2026-11-14, medians of 5 after a warm call. Workload: eight columns of 4 levels over 20,000 rows, all pairs and triples (cycled to the combo count), a binary target that is the parity of two columns.

## What was added
`detect_synergy_combos` used to score each pair or triple with one pass of `_combo_mm_mi_cols_njit` in a Python loop (up to `max_combos` = 4000). It now collects the combos that pass the cell and rows-per-cell filters (same order, same `max_combos` cut), scores orders 2 and 3 together through `combo_mm_mi_dispatch` (device `synergy_combo` kernel or the numba kernels by the kernel-tuning entry `synergy_combo_backend`, rows x combos), keeps orders above 3 on the host, and applies the joint-MI and synergy-ratio filters in the original order, so the result list is identical.
`synergy_combo`: one 256-thread block per combo, rows fill a shared `(joint cell, class)` integer table (integer atomics), class totals from the table, plug-in MI over the occupied cells with a warp-shuffle block reduction and the Miller-Madow debit formed from the occupied counts as on the host. Combos over 8192 table entries (or more than 16 classes) return NaN and are scored by the host. The code matrix goes up once, as bytes when every cardinality is at most 127 (a column block of the codes, halved on OOM by `run_column_blocks`).

## cProfile (device call, 640 combos)
A first version spent 4.6 of its 7 ms in 640 `np.prod` calls on the host side (the per-combo cell count); replaced by Python integer products. After the change the call is 2.5 ms: the upload, the launch and one `ndarray.get`.

## nvprof (hardware counters)
`synergy_combo` (launches per call: 1 for all combos): achieved occupancy 93.9 percent, double-precision efficiency 17.4, DRAM read throughput 7.5 MB/s, global load efficiency 99.1 percent. The eight code columns (160 KB as bytes) live in L2 and L1, so DRAM utilization is near zero out of the card's 112 GB/s peak bandwidth: the kernel is bound by the shared-memory atomics and the logarithms.

## Wall clock (numba column kernel per combo in a Python loop against the device call)
| combos | numba | device |
|---|---|---|
| 10 | 0.73 ms | 0.98 ms |
| 40 | 3.16 ms | 0.91 ms |
| 160 | 11.93 ms | 1.22 ms |
| 640 | 47.19 ms | 2.52 ms (18.7x) |

Equality to 2.6e-15 relative on all pairs and triples; `detect_synergy_combos` returns the same combos in the same order with the device off and on (`tests/feature_selection/gpu/test_synergy_combo_device.py`); the pre-sweep floor is 0.5 million row-combos.
