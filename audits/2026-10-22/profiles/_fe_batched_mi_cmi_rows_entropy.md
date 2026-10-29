# Device CMI batch (`batched_cmi_gpu`): GPU profile and the row-entropy kernel, 2026-10-29

Host: GTX 1050 Ti (sm_61, 4 GB). Shape that matters: `score_candidates_by_cmi`, 1500 candidates, 30-column support, n = 20000 (the one big call of the all-flags fit). Driver `D:/Temp/gpu_cmi_driver.py`, warm, three profiled calls inside the `cudaProfilerApi` window.

## Profile 1: host (cProfile)
The call was 4.5 to 4.7 s on the host thread pool (`score_candidates_by_cmi` 4.57 s cumulative in the whole-fit profile, all of it pool wait). The device path (`MLFRAME_CMI_GPU=1`) did the same call in 1.6 to 2.1 s with scores equal to 3.6e-15 and an identical ranking, at every shape measured (n = 20000 and 100000, p = 32 to 1000, support of 0, 2 and 30 columns; GPU 1.3 to 3.6 times faster).

## Profile 2: GPU (nsys, kernel summary; before the change)
| kernel | share of kernel time | avg per launch |
|---|---:|---:|
| `rows_ent_nnz_i32` (row entropy and occupied-cell count) | 65.6% | 326 ms |
| `batched_joint_hist2_i32` (joint histograms, atomics) | 27.0% | 134 ms |
| `radix_select_interp_f64_v3` (quantile binning) | 5.5% | 54 ms |
Launch shapes of the entropy kernel: (1500, 200000) and (1500, 400000) int32 count matrices, sparse (at most n = 20000 occupied cells per row).

## Profile 3: GPU counters (nvprof on this Pascal host, kernel-filtered run)
- `rows_ent_nnz_i32`: achieved occupancy 0.99, SM efficiency 98%, global load efficiency 100%, DRAM utilisation low (1 of 10), executed IPC 0.31, stalls: execution dependency 55 to 68%, memory dependency 3 to 20%. Not memory-bound and not occupancy-limited: it is latency-bound on the double-precision `log` (rate 1/32 on Pascal). Even with 10% of the cells occupied, every warp holds an occupied lane, so every warp runs the log for all its cells.
- `batched_joint_hist2_i32`: occupancy 0.69 to 0.82, SM efficiency 99%, global load efficiency 91.7%, IPC 0.08 to 0.28, memory-dependency stalls 14 to 41%, execution-dependency stalls 22 to 39%: random global atomics into the large counts buffer.

## Change (one thing): lookup-table entropy kernel
Counts are integers <= n, so `c * ln(c)` comes from a table built once per n, and with p = c / n the row entropy is (ln(n) * sum(c) - sum(c ln c)) / n, which also holds when a row does not sum to n. New kernels `rows_ent_nnz_i32_lut4` (int4 loads, M % 4 == 0) and `rows_ent_nnz_i32_lut1` (scalar), dispatched by `_rows_entropy_lut`; the log kernel stays as the fallback and for the split-K case; `MLFRAME_ENT_LUT=0` selects it.
- Kernel alone (microbenchmark `D:/Temp/ent_lut_bench.py`): 262.9 ms to 13.7 ms (19x) at (1500, 200000); 482.1 ms to 161.1 ms (3x) at (1500, 400000). An int4-load variant of the log kernel alone gave only 0.98 to 1.10x, so the vector load is not the lever; the log is.
- Values: max difference 8.9e-15 in the entropy, occupied-cell counts identical; whole call: LUT versus log kernel 0.0, versus host 3.6e-15, same ranking.
- Full call (`D:/Temp/lut_check.py`): host 4.72 s, device with the log kernel 1.62 s, device with the table 1.21 s. nsys after the change: kernel time per call 0.99 s to 0.52 s; the entropy kernel is now 32% (`rows_ent_nnz_i32_lut4`), `batched_joint_hist2_i32` 53%.
- Whole-fit wall of the big call inside the all-flags fit: 4.54 s (host) to 1.9 s (device, before the table).

## Dispatch change that goes with it
`score_candidates_by_cmi` now takes the batched device CMI by default when CUDA is usable, p >= 64 and n * p >= 1,000,000 (`_cmi_candidate_batch_on_device`), not only under STRICT. The device counts the same partitions as the host kernel (scores equal to 1.3e-13, identical ranking), it changes no binning and no selection; `MLFRAME_FE_GPU_STRICT=0` still forces the host path, and any device error falls back to the host loop with a warning.

## Next kernel (not done)
`batched_joint_hist2_i32` is 53% of the kernel time now. One thread per (row, column) cell does a random `atomicAdd` into the (K, Kx*Kb) int32 buffer; consecutive threads are consecutive columns, so every warp-level atomic touches 32 distant regions. Candidates: a row-fast thread mapping with the rows grouped by the joint support code, or a per-block shared-memory privatised histogram for the small cells; each needs the nvprof counters re-collected and a full-fit A/B before it ships.
