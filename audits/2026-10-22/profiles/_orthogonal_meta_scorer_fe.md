# `_orthogonal_meta_scorer_fe` (meta scorer: fingerprint the frame, pick the scorer) with its carved child `_meta_fingerprint_gpu`

2026-11-16. The meta scorer summarises a frame by a fingerprint (`x_unique_avg`, `mean_abs_pearson`, `mean_abs_skew`, `mean_kurtosis`, `inter_x_max_corr`, `dcor_proxy`), predicts the scorer from it and dispatches to that scorer (cmim, KSG, copula, ...), whose kernels are in their own modules. The fingerprint statistics of the skew / kurtosis / Pearson family are one set of column moments; `_meta_fingerprint_gpu` computes them in one pass.

## Kernels

Five sums per column about its mean: `mean`, `m2`, `m3`, `m4` (means of the 2nd to 4th power of `c = x - mean`) and `cxy = sum(c * yc)`; the host keeps the pandas `G1` / `G2` formulas and `r = cxy / sqrt(sum(c^2) * syy)` on the K-vectors. The vectorised host form made `(n, K)` temporaries `c`, `c^2`, `c^3`, `c^4` and a matrix product (7 passes).
- `_moments_njit`: `prange` over columns, two passes per column.
- `column_moments_blocks` (CUDA): ONE launch for the block, a block per column, warp-shuffle reductions, the column read twice coalesced.
- Kernel-tuning entry `meta_fingerprint_backend` (sweep over rows, 100 skewed columns, equivalence 1e-8); `MLFRAME_META_FINGERPRINT_DEVICE=0|1` forces a side. Columns go through in column blocks of 256 MB.

## cProfile and timings (100 columns, 100000 rows)

Skew + kurtosis + Pearson, warm: numpy vectorised 0.215 s, numba 0.033 s (6.5x), CUDA 0.066 s (the block's 80 MB upload included); the swept choice is the host. Equal to the pandas per-column loops to 1e-9 (`_VECTORISED = False` reference) and to the numpy form to 1e-14. cProfile of the family call (`hybrid_orth_mi_meta_fe`, 100000 rows): 2.53 s wall, `pd.unique` 0.40 s (the per-candidate marginal MI of the cmim scorer, 102 calls; now one more call of the fused cmim kernels, see `_orthogonal_cmim_fe.md`), `quantile_bin_block` 0.30 s (the binning the chosen scorer needs), `_mean_abs_skew_and_kurtosis` 0.15 s before this change.

## nvprof (GTX 1050 Ti, sm_61)

Kernel launches per call: 1 (`column_moments_blocks`) per column block plus the block upload; the kernel reads each element twice and does 4 fused multiply-adds per element, so on 100 x 100000 doubles it moves 160 MB in 0.01-0.02 s: device throughput is bounded by the PCIe upload of the block (80 MB, about 8 ms at 10 GB/s) rather than by the kernel, which is why the swept choice for a host-resident frame is the numba kernel. Occupancy of the 256-thread blocks is limited by the 100 blocks on 6 SMs; the whole-GPU utilization of the fingerprint is negligible against the scorer it picks (the family's kernels are in `_orthogonal_cmim_fe`, `_ksg_cd_gpu`, `_orthogonal_copula_gpu`).

## Tests

`tests/feature_selection/filters/test_meta_fingerprint_gpu.py` (both kernels against numpy; the three statistics against the pandas loops on either backend; a narrow block budget), the existing `test_meta_fingerprint_vectorised.py`, `test_meta_scorer_pearson_matrix.py`, `test_meta_scorer_autoselect.py` (42 tests with the sweep-input test).
