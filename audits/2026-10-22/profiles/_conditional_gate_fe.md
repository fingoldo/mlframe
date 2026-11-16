
## 2026-11-03: resident operands keyed on the caller's array

Stage sample (100000 x 100): 6 of 17 samples in `_fe_resident_operands.resident_operand` (the content-hash line), from `gate_grid_mi_resident` and
`best_existing_op_mi_resident`. They pass float32 host columns with `dtype=float64`: the conversion made a new array per call, so the identity memo of
`_content_hash_memoized` never hit and every call copied and hashed 100000 values. The key is now the content of the caller's array plus its dtype and the
target dtype (the conversion is a pure function of those); the copy and upload happen only on a miss. After: 0 samples in `resident_operand`, about 0.6 s of
the stage. Side effect: `test_pair_search_residency_codes_resident_into_mi_gate` (an xfail for a host-code re-upload) now passes.

## 2026-11-16: the family under the contract (row-argmax, gate rank, tau grids); whole-pass profile

Host: GTX 1050 Ti (sm_61, 112 GB/s), 8 logical cores. Workload: `hybrid_conditional_gate_fe_with_recipes` plus `hybrid_row_argmax_fe_with_recipes` on 100000 x 100 float32 columns whose target is a real regime switch (`where(f5 > 0, f1 > 0, f2 > 0)`), so the tau-grid sweep runs in full: 12308 grid columns in 26 flushes. Found: `gate_select__f1__f2__f5` and `gate_mask__f1__f5` (tau -0.00026) on every path.

### Whole pass, by path

| path | wall per pass |
|---|---|
| host scan (`MLFRAME_FE_GPU_STRICT=0`) | 26.6 s |
| device-born, rank-binned (the previous AUTO) | ~ the host's (a 204-column flush: 0.41 s device against 0.35 s host at 100000 rows) |
| device-born, percentile-edge binned (STRICT, and AUTO from 10000 rows now) | 2.5 to 3.8 s |

At 50000 rows and 60 columns, three seeds: host 15.6 / 12.6 / 12.8 s, AUTO now 4.0 / 1.2 / 1.2 s; the gates chosen (gate, operands, tau to eight digits) are the same on 5000, 10000, 20000, 50000 and 100000 rows and differ at 2000 rows, where the rank binner stays.

### Host profile (cProfile, one AUTO pass)

2.68 s for 298084 function calls: 1.22 s in the blocking cupy `.get()` of the MI vectors (93 calls: the device time surfacing), `bincount` 0.14 s (6 calls), the array uploads 0.13 s, `concatenate` 0.12 s, `column_stack` 0.10 s, the device median 0.07 s. On the host path 20.0 s of the 26.8 s are 26 calls of `_plugin_mi_classif_batch_njit` and 3.2 s `_gate_flush_pending`.

### Launches per call and the GPU profile (nvprof, 4 AUTO passes, 100000 rows)

About 4700 kernel launches per pass (18880 driver launches for 4 passes plus 693 runtime launches). Share of the GPU time: `gate_tau_select_float` 16.4 % (2880 launches, 453 us each: the tau-grid block of a select spec), `mi_hist_split_f32` 13.1 % (208 launches, 5.0 ms), `build_op_cands_cm` 12.6 % (1920 launches, 521 us), `cupy_copy__float32_float32` 12.1 % (10468 launches, 92 us: small casts and gathers, the launch-bound tail), the prefix-scan kernels 5.4 %, the cub merge sort 2.2 %, `gate_tau_mask_float` 1.6 %.

Metrics (nvprof, hardware counters, 20000-row run): `gate_tau_select_float` achieved occupancy 0.85, multiprocessor activity 93 %, DRAM write throughput 23 GB/s (a column write of 112 GB/s peak: 21 percent), read 3.5 GB/s; `mi_hist_split_f32` occupancy 0.86, activity 94 %, read throughput 36 GB/s (32 percent of the peak); `build_op_cands_cm` occupancy 0.88, activity 94 %, write 14 GB/s. The pass is launch-bound on its small elementwise tail (10468 copies), which is the open item.

### Backend choice and memory

Outside STRICT (which forces the device): the device-born grid bins by percentile edge from `_GATE_AUTO_EDGE_MIN_ROWS` (10000) rows (`MLFRAME_FE_GATE_AUTO_EDGE=0` keeps the rank binner); below that the tuner entry `fe_gate_grid_backend` (`_resident_candidate_mi_ktc.gate_grid_use_device`, a sweep over rows of the host scan against the rank-binned device grid, equal to 1e-9; host at every swept size here) decides, `MLFRAME_FE_GATE_GRID_DEVICE=0|1` forces a side. The row-argmax and the gate pruning have their own records (`_row_argmax_gpu.md`, `_gate_rank_divergence.md`). Memory: a flush takes specs until `_GATE_MI_COL_BUDGET` (512) grid columns are collected, a column block of that width.
