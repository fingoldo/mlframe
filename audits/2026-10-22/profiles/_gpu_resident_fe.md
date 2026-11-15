# _gpu_resident_fe: profile (2026-11-12)

Entry point profiled: `gpu_resident_pair_candidate_mi` (100000 rows, the pair-candidate generation and its MI), GTX 1050 Ti. Warm-up once, then three measured calls under nvprof (CUPTI on `PATH`, as in the other profiles). Wall time per call with the device synchronized: 0.0605 s.

| kernel | total ms | share of GPU time | launches |
|---|---|---|---|
| `fused_gen` (fused candidate generation, one launch per block) | 76.4 | 23.6% | 4 |
| `radix_select_f32_v5` (quantile edges) | 64.6 | 20.0% | 4 |
| CUB sorts (four argsort-like calls, 64-bit and 32-bit) | 34.2 + 24.6 + 12.9 + 12.5 = 84.2 | 26.0% | 4 each |
| `cupy_copy__float64_float32` (dtype copy) | 30.1 | 9.3% | 4 |
| `transpose_f32` | 24.4 | 7.5% | 4 |
| `mi_hist_split_f32` + `mi_from_counts_col_f32` | 16.2 + 15.9 = 32.1 | 9.9% | 4 each |

GPU total 323.4 ms over four calls (about 81 ms per call). The wall per call (60.5 ms) is below the summed kernel time because the kernels overlap on the stream. The sort is again the largest group; the float64-to-float32 copy and the transpose together are 17% of GPU time. The float32 candidate is already a documented rejection: the chain run entirely in float32 (bench of 2026-06-20) gave 1.09x at 100000 rows and flipped the winner at 300000 rows (a near-tie cluster, selection instability), so the copy is not removed. Correction, measured the same day: the `transpose_f32` launches were one per call from `_resident_mi_candidates`, because the entry point built its candidate block row-major and the scorer transposed it. The block is now built column-major (`column_major=True`, scorer `x_is_cm=True`); the generator's twin is bit-identical. A/B through the entry point at 100000 rows, synchronized, best of 3: old 54.8 ms, new 42.5 ms per call, MI vectors equal (maxdiff 0). The module's remaining GPU cost is the sort and the quantile select.

Dispatch: the chunk size comes from `_gpu_k_chunk` (VRAM fraction, KTC kernel `gpu_fe_k_chunk_vram_fraction`). Optimization status: profiled; the float32 candidate is a documented rejection and the column-major write is already in place; the remaining transposes belong to the batched-MI module.

## Host profile, launches, hardware counters and utilization (2026-11-14, elevated process, nvprof `--metrics`)
`gpu_resident_pair_candidate_mi`, 100,000 rows, 384 candidates, float32 operands, warm call, device synchronized: wall 30.3 ms (median of 5). cProfile of one call: 0.026 s of the call is the single device-to-host read of the (K,) MI vector (`ndarray.get`, 2 calls, it waits for the queued kernels and so carries GPU time); the host work is below 2 ms in total (`_unary_minimal` 16 calls 2 ms, `cupy.array` uploads 1 ms, `_unary_stack_cm` 3 ms cumulative). Kernel launches per call: 1 `fused_gen_cm` per candidate chunk, 1 `radix_select_f32_v5`, 1 `mi_hist_split_f32`, 1 `mi_from_counts_col_f32`, plus the unary-stack elementwise kernels (four `cupy_power` launches replaced by multiplications, see CUDA.md section 19, item 7) and the dtype copy; the chunk width is the kernel-tuning entry `gpu_k_chunk`.

| kernel | achieved occupancy | DP efficiency | DRAM read | global load efficiency |
|---|---|---|---|---|
| `fused_gen_cm` | 86.3 % | 14.7 % | 44.4 GB/s | 86.2 % |
| `radix_select_f32_v5` | 74.7 % | 0 | 53.0 GB/s | 99.7 % |
| `mi_hist_split_f32` | 98.0 % | 0 | 16.8 GB/s | 66.6 % |
| `mi_from_counts_col_f32` | 49.7 % | 2.3 % | 0.1 GB/s | 12.5 % (broadcast reads of a small table) |

Whole-GPU utilization and throughput: the four kernels are resident at 50 to 98 percent occupancy; the best DRAM read throughput is 53.0 GB/s (47 percent of the card's 112 GB/s bandwidth) in the radix select and 44.4 GB/s (40 percent) in the fused generation, so the call is neither memory- nor occupancy-starved, and its remaining cost is the sort-like radix select (33.6 percent of the GPU activity of the call in the trace). The CPU twin of the whole call is `_gpu_resident_shared.cpu_pair_candidate_mi` (numpy grid scored by the njit batch MI of the `hermite_fe` package).

## 2026-11-15: the orthogonal-basis block `_gpu_resident_basis` carved under this family

`_gpu_resident_basis` is the prewarp / orthogonal-basis block that `_gpu_resident_fe` carved out of itself (Tier E; the parent imports it as `_grb` and re-exports its names). Its batched evaluator `_gpu_evaluate_basis_matrix` builds the one-hot Clenshaw basis of every (column, degree) cell of a group in one `clenshaw_basis` launch after a batched robust preprocess. Measured on the GTX 1050 Ti (elevated process, hardware counters), 100000 rows, 8 float32 columns (4 chebyshev, 4 legendre), degrees 1 to 4 (a 32-column candidate block), robust axis on, warm call: wall 14.5 ms (median of 5).

GPU time 13 ms in 11 kernel kinds: `clenshaw_basis` 2 launches 3.9 ms (30 percent; the chebyshev group 0.68 ms, the legendre group 3.2 ms), `col_minmax_tiled` 2 launches 2.96 ms (23 percent), `heavy_tail_stats_tiled` 1 launch 2.85 ms (22 percent), `radix_select_interp_f64_v2` 0.77 ms, `radix_select_f32_v3` 0.37 ms, `basis_preprocess_nr` 0.34 ms.

| kernel | achieved occupancy | DP efficiency | DRAM read | global load efficiency |
|---|---|---|---|---|
| `clenshaw_basis` | 50.0 % | 31.1 % of the 1:32 peak | 1.3 GB/s | 75.0 % |
| `heavy_tail_stats_tiled` | 12.5 % | 0.7 % | 3.3 GB/s | 100 % |
| `robust_scale_combine` | 1.6 % | 0.04 % | 0.9 GB/s | 100 % |

`clenshaw_basis` is compute bound on the double pipe (the recurrence runs in double for either element type). `heavy_tail_stats_tiled` runs one block per column, so with 8 columns it fills 8 of the card's 6 x 8 resident-block slots: 12.5 percent occupancy; the robust decision is basis independent and is computed once per matrix (`heavy_host`), not per basis. The host baseline of the basis values is `polyeval_dispatch` of the `hermite_fe` package (forward recurrence in numba) and the batched kernels agree with it to 1e-12 (the parity tests of the cross-basis and prewarp families). The block works on the column group its caller hands it (`M` of shape (n, K)); the cross-basis builder asks for one leg column at a time (`_leg` in `_gpu_resident_cross_basis`, whose products are then scored in column blocks of at most a quarter of the pool room), so the candidate matrix of one call is (n, degrees x columns of the group) and nothing grows with the number of candidates of a family.

Follow-up the same day: `heavy_tail_stats_tiled` was a single block for 8 columns (12.5 percent occupancy, 2.85 ms, 22 percent of the call). Its rows are now split over `blockIdx.y` shares (about eight blocks per SM: 48 shares here) that write partial (outer count, bulk maximum, outer minimum) rows, and a finishing kernel `heavy_tail_finish` combines them; the three reductions are an exact integer sum, a max and a min, so the flags equal the single-block reference for every share count (a test runs 1, 3, 17 and the default grid fill against the reference). Counters after: `heavy_tail_stats_tiled` 0.54 ms, achieved occupancy 63 percent, global load efficiency 100 percent; `heavy_tail_finish` 12 microseconds. `_gpu_detect_heavy_tail_batched` as a whole (quantiles, scale, stats, verdict) 8.4 -> 4.1 ms interleaved, identical flags.
