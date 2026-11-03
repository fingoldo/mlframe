
## 2026-11-03: f32-X fused MI kernel

Float32 edge search (edges rounded up to float32; same codes): 1000 columns x 100000 rows 0.0195 -> 0.0166 s. Rejected after measuring: a shared histogram
per warp (8 copies of the 20-cell binary table, merged before the MI) 0.0165 s, no gain: the atomics are not the limit. The kernel reads 400 MB in 16.5 ms
(~24 GB/s of ~112 GB/s), so it is latency / occupancy bound, not compare- or atomic-bound.
Kept: the column's interior edges in shared memory (up to 256; more falls back to global reads) for the single-launch and split kernels; the binary search read
them per value, strided by K in global memory. 1000 x 100000: float32 0.0205 -> 0.0114 s, float64 0.0193 -> 0.0165 s; 16 columns 0.0011 -> 0.0014 s (the
narrow split launch pays the shared fill per block). Same MI in every case.

## Conditional CMI: fused joint entropy by default for large joints (2026-11-03)
Where found: nsys window of the 100k x 100 GPU fit, batched_joint_hist2_i32 0.451 s over 13 launches; the launches of ~28M elements took 3.5 ms (small joints)
to 60-78 ms (joints of 1000+ cells: random global atomics over a (K, Kx*Kyz) count buffer far larger than L2, then a separate reduce).
Bench (n=30000, K=930, Kx=10, min of 5), count+reduce / fused ms: M=40 3.6 / 21.8, 400 36.6 / 22.9, 1000 59.9 / 21.7, 4000 73.3 / 11.5, 10000 78.7 / 15.8,
30000 81.9 / - (over shared memory). The June rejection measured only M <= 1000 joints of the marginal path.
Change: `MLFRAME_FE_GPU_FUSE_CMI_ENTROPY` defaults to `auto`: fused from `_FUSED_ENTROPY_MIN_M` = 300 cells (KTC `cmi_fused_entropy_backend` over (n, m), async
sweep, `_fe_cmi_fused_backend.py`); `1` / `0` force it. Same occupied cells; entropies differ by a few ULP (relative <= 4.4e-16: another summation order),
so the old "bit-identical" note was wrong. Fit check 50000 x 100 all families, one process, fit cache cleared, off/auto alternating: same 9 selected
features (twice each); warm walls 85.4 s (off) vs 77.4 / 77.1 s (auto).
Left: joints over the shared-memory limit (M > ~12000 cells) still take count+reduce at ~80 ms; a global-scratch per-block twin is the next step there.
Global-scratch twin (`batched_joint_entropy2_g`, same day): over the shared-memory limit, one block per SM walks the columns with its histogram in a global
slice; used while SMs x M x 4 bytes fit L2. n=30000, K=930, Kx=10, min of 5, count+reduce / global (blocks 6, 12, 24, 48, 96) ms: M=15000 77.9 / 24.0, 30.8,
61.2, 66.6, 68.5; M=30000 79.6 / 47.8, 59.2, 72.0, 82.2, 81.6; M=100000 86.1 / 125.4, 119.8, 105.7, 112.2, 111.0 (slices over L2: count+reduce kept).
Same occupied cells, entropies to round-off (test_cmi_fused_entropy_default). Fit check 50000 x 100, both fused paths off / on: same 9 features; warm 84.8 vs 76.3 s.

## cupy sorts in the GPU fit (2026-11-03)
50k x 100 all-families fit with cp.sort / argsort / unique / lexsort / percentile / quantile / median wrapped and synchronised (diagnostic; method calls
`ndarray.argsort` are not seen): about 1.9 s in total. Largest: hermite `_nonpoly_cells_gpu._quantile_columns` sort 0.41 s (24 calls, 30000 x 100 f32),
`_qbin_block_gpu.sync_free_qbin_codes_block` sort 0.34 s (25 calls, 14-16 x 50000 f32), `_fe_cmi_perm_null_gpu.conditional_perm_null_gpu` argsort 0.26 s,
`_gpu_resident_rank_bin` argsort 0.14 s (196 calls), conditional gate median 0.12 s, `_resident_mi_candidates` percentile 0.10 s.
`_quantile_columns` now takes `_radix_quantiles` (sort as fallback): equal values (test_nonpoly_quantiles_radix), 30000 x 100 f32, 3 fractions: 23.7 -> 1.8 ms.
