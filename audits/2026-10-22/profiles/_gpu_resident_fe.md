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
