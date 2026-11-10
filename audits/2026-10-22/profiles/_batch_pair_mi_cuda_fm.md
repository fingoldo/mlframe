# `_batch_pair_mi_cuda_fm.py` — feature-major byte-code pair-MI kernel

Measured 2026-11-10, GTX 1050 Ti (Pascal, 6 SMs, 4 GB), 100000 rows, `D:/Temp/bench_pairmi.py`, `D:/Temp/bench_pairmi2.py`.

| kernel | pairs | time | per pair | effective read |
|---|---:|---:|---:|---:|
| `batch_pair_mi_cuda` (numba.cuda, row-major int32) | 179700 | 18.9 s | 105 us | 2.8 GB/s |
| `batch_pair_mi_cuda_shared_fused` (RawKernel, row-major int32) | 40000 | 6.8 s | 171 us | — |
| `batch_pair_mi_cuda_feature_major` (this kernel), through `dispatch_batch_pair_mi_chunked` | 179700 | 0.60 s | 3.4 us | 89 GB/s |

Why the old kernels were slow: a warp reads `factors_data[i * n_features + a]` for 32 consecutive rows i, 601 bytes apart at 600 columns, i.e. 32 sectors fetched for 32 useful bytes (~3 % of the card's ~112 GB/s). The new kernel transposes the codes once on the device to `(n_features, n_pad)` unsigned bytes, rows padded to a multiple of four so each column is word-aligned; a thread loads four rows of a column as one 32-bit word (coalesced, full sectors), counts into a dynamic-shared int32 histogram and finishes with the shared-fused kernel's warp-shuffle reduction. Counts are exact integers, so the MI equals the row-major kernels' (max difference 1.5e-18 in the benchmark; `tests/feature_selection/gpu/test_batch_pair_mi_cuda_fm.py` checks 1e-12 relative).

Applies when every code, class and cardinality fits a byte; else `ValueError` and the dispatcher keeps the row-major kernels (`_PairMiDispatch.try_feature_major`, tried first inside the existing "cuda" decision, so the kernel-tuning-cache choice between njit and CUDA is unchanged; backend tag stays `cuda`). VRAM: the byte matrix is a quarter of the int32 one; the existing `vram_ok` gate runs first.

Effect on the 100k x 600 fit: the pair-MI dispatcher was 36 s of the 118 s `fe_step` (cProfile).
