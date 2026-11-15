# Profile record: device-born conditional-dispersion candidates (`_extra_fe_families_dispersion_resident`, host twin `_extra_fe_families_dispersion`)

Host: GTX 1050 Ti (sm_61, 4 GiB, 112 GB/s), 8 logical cores, elevated process (hardware counters). Measured 2026-11-15, medians of 3 after a warm call. Workload: 6 numeric columns, 60 candidate columns (every ordered pair x `absz` and `z2`), 100000 rows, 10 bins, binary target; the device path builds the (60, n) matrix from the recipes and scores its plug-in MI.

## What the module is
The Tier-1 gate of the conditional-dispersion family scores `MI(candidate; y)` for every candidate. The host path scores the host-built matrix (`generate_conditional_dispersion_features`, whose per-bin z-score is the numba kernel `_zscore_from_bins_njit`, and `mi_classif_batch_chunked`); the device path rebuilds the candidates from the resident operand columns and the recipe constants (the per-bin mean and sigma, the x_j edges) with ONE `disp_columns` launch for the whole block (every candidate's gathers, sigma floor, divide and emission fold in one kernel over the (K, n) output with the rows fastest; the first version launched once per candidate), then scores them with the resident plug-in MI. The kernel-tuning entry `dispersion_gate_backend` chooses host or device by row count.

## Column blocks (streaming)
The candidate matrix is `K x n` at the relaxed candidate dtype. `_mi_by_column_blocks` builds a column block, scores it and releases it: a block is a quarter of the room under this thread's pool limit divided by three copies of its columns, so the working set is a block however many candidates there are. A column's MI depends on that column alone (its own edges and histogram), so the blocks give the single-matrix values (a test with a room of three columns, 20 blocks, compares equal).

## cProfile (one device scoring, 100000 rows, 60 candidates)
The Python side is now the staging (one resident-operand lookup per distinct x_i and per distinct x_j, one small padded table upload, the stack of the operand rows) and one launch; the 60 per-candidate launches of the first version (0.004 s of Python) and their per-pair hashing are gone.

## nvprof (hardware counters)
Launches per scoring: 1 `disp_columns` (all 60 candidates), 1 `radix_select_f32_v5` for the edges of the whole block, 1 `mi_hist_split_f32` and 1 `mi_from_counts_col_f32` for the histograms and the MI, plus 1 `disp_bin_codes` launch for the bin codes of all distinct x_j (the first versions launched five small kernels per x_j). GPU time about 3.5 ms: `radix_select_f32_v5` 43 percent (1.88 ms), `disp_columns` 23 percent (1.0 ms for the 60 x 100000 block), `mi_hist_split_f32` 13 percent (0.57 ms).

| kernel | achieved occupancy | other counters |
|---|---|---|
| `disp_columns` | 0.95 | DRAM read 20 GB/s, global load efficiency 74.6 percent (the bin-table gather is the uncoalesced part), single-precision FLOP efficiency 3.2 percent (memory bound, as an elementwise fold is) |
| `radix_select_f32_v5` | 0.73 | DRAM read 34 GB/s, global load efficiency 99.9 percent |

Whole-GPU utilization: the device scoring is 10.0 ms of wall at 100000 rows, of which about 3.5 ms is GPU time (35 percent busy; the rest is the resident-operand lookups, the table upload and the blocking copy); the first version was 15.8 ms (25 percent busy). The candidate build alone, 396 columns at 100000 rows, fell from 210-227 ms for the cupy-op chain to 146 ms (one launch per column), 28 ms (one launch for the block) and 17 ms (one more launch for all the bin codes).

## Wall clock (host gate scoring against the device-born scoring, same top candidates)
| rows | host | device |
|---|---|---|
| 5,000 | 8.6 ms | 4.5 ms (1.9x) |
| 30,000 | 30.4 ms | 5.0 ms (6.1x) |
| 100,000 | 100.4 ms | 10.0 ms (10x) |

On this card the device now wins at every swept size; below a crossover on another card the host gate runs (the resident gate returns `None`, as for any device fault); the device path equals the host's survivors in the parity test (STRICT, 6000 rows forced) and the candidate columns equal the cupy chain exactly (`test_dispersion_column_kernel`).
