# Profile record: device-born conditional-dispersion candidates (`_extra_fe_families_dispersion_resident`, host twin `_extra_fe_families_dispersion`)

Host: GTX 1050 Ti (sm_61, 4 GiB, 112 GB/s), 8 logical cores, elevated process (hardware counters). Measured 2026-11-15, medians of 3 after a warm call. Workload: 6 numeric columns, 60 candidate columns (every ordered pair x `absz` and `z2`), 100000 rows, 10 bins, binary target; the device path builds the (60, n) matrix from the recipes and scores its plug-in MI.

## What the module is
The Tier-1 gate of the conditional-dispersion family scores `MI(candidate; y)` for every candidate. The host path scores the host-built matrix (`generate_conditional_dispersion_features`, whose per-bin z-score is the numba kernel `_zscore_from_bins_njit`, and `mi_classif_batch_chunked`); the device path rebuilds the candidates from the resident operand columns and the recipe constants (the per-bin mean and sigma, the x_j edges) with one `disp_column` launch per candidate (the gathers, the sigma floor, the divide and the emission fold fused, written into the candidate's row), then scores them with the resident plug-in MI. The kernel-tuning entry `dispersion_gate_backend` chooses host or device by row count.

## Column blocks (streaming)
The candidate matrix is `K x n` at the relaxed candidate dtype. `_mi_by_column_blocks` builds a column block, scores it and releases it: a block is a quarter of the room under this thread's pool limit divided by three copies of its columns, so the working set is a block however many candidates there are. A column's MI depends on that column alone (its own edges and histogram), so the blocks give the single-matrix values (a test with a room of three columns, 20 blocks, compares equal).

## cProfile (one device scoring, 100000 rows, 60 candidates)
0.024 s for 8739 function calls: 0.004 s in the 60 `disp_column` launches (Python side, 67 microseconds each), 0.002 s of `build_dispersion_matrix_gpu` itself, 0.002 s hashing the operands for the resident-operand cache (43 `xxh3` calls, one per pair operand), 0.001 s for the one blocking copy of the MI vector.

## nvprof (hardware counters)
Launches per scoring: 60 `disp_column` (one per candidate), 1 `radix_select_f32_v5` for the edges of the whole block, 1 `mi_hist_split_f32` and 1 `mi_from_counts_col_f32` for the histograms and the MI, plus 6 sets of small code kernels (one per distinct x_j: searchsorted, clip, where, isfinite). GPU time 3.9 ms: `disp_column` 34 percent (1.34 ms, 22 microseconds per launch), `radix_select_f32_v5` 33 percent (1.29 ms), `mi_hist_split_f32` 15 percent (0.57 ms).

| kernel | achieved occupancy | other counters |
|---|---|---|
| `disp_column` | 0.93 | DRAM read 47 GB/s, global load efficiency 90 %, single-precision FLOP efficiency 2.2 % (memory bound, as an elementwise fold is) |
| `radix_select_f32_v5` | 0.73 | DRAM read 34 GB/s, global load efficiency 99.9 % |
| `mi_from_counts_col_f32` | 0.15 | 55 microseconds, a reduction over the histograms, not a bandwidth kernel |

Whole-GPU utilization: 3.9 ms of GPU time inside a 15.8 ms wall (25 percent busy): the rest is the host side of 60 launches and of the operand hashing, so at this size the call is launch bound, not kernel bound; fusing the candidates of one x_j into one launch would remove most of the 60 and is open (it changes nothing in the values).

## Wall clock (host gate scoring against the device-born scoring, same top candidates)
| rows | host | device |
|---|---|---|
| 5,000 | 8.2 ms | 13.2 ms (host wins) |
| 30,000 | 31.7 ms | 14.0 ms (2.3x) |
| 100,000 | 102.5 ms | 15.8 ms (6.5x) |

Below the crossover the host gate runs (the resident gate returns `None`, as for any device fault); the device path equals the host's survivors in the parity test (STRICT, 6000 rows forced) and the candidate columns equal the cupy chain exactly (`test_dispersion_column_kernel`).
