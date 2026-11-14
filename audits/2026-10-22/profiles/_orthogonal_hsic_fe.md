# Profile record: biased HSIC of many columns (`_orthogonal_hsic_fe`, device twin `_hsic_gpu`)

Host: GTX 1050 Ti (sm_61, 4 GiB), 8 logical cores, elevated process (hardware counters). Measured 2026-11-14, medians of 5 after a warm call. The first profile of the device twin (2026-11-03, where the cost was found) is in `_hsic_gpu.md`.

## Kernels and choice
Host: `_hsic_biased_columns_njit` (numba, `prange` over columns, O(m^2) pair terms per column). Device: `hsic_biased_cols` in `_hsic_gpu.py`, one block per column, the column's subsample in shared memory, each thread sums its rows `a` over `b > a` of `exp(-(x_a - x_b)^2 g) * Lc[a, b]`, block reduction; the values and the exponent are in the criterion type, the Gram matrix and the sums in the accumulator type (`{ACC}`). Columns go up in column blocks bounded by a quarter of the free VRAM. The backend is chosen by `hsic_backend` through the kernel-tuning cache (`hsic_biased_backend`, swept over (m, k); the sweep is also registered with the central registry so `retune_all` refreshes it), with a pre-sweep floor of one million pair terms.

## cProfile (device call, m = 1000, k = 400)
29 ms in total: the launcher 2 ms own time, `cupy.array` uploads 1 ms (3 calls), the dtype narrowing `astype` 1 ms; the rest is the kernel (the host bandwidths `hsic_column_sigmas` are computed before the call).

## nvprof (hardware counters)
`hsic_biased_cols` (launches per call: 1 per column block, one for the whole batch here): 25.2 ms of the 26.7 ms traced GPU time (94 percent; the uploads are 5.7 percent); achieved occupancy 98.8 percent, double-precision efficiency 20.5, single-precision efficiency 4.5, DRAM read throughput 32.0 GB/s (29 percent of the card's 112 GB/s peak bandwidth), global load efficiency 25.0 percent (each thread reads its own row of the Gram matrix, 8 * m bytes from its neighbour's).

## Wall clock (numba against the device call with bandwidths excluded, uploads included)
| m x k | numba | device |
|---|---|---|
| 500 x 100 | 31.8 ms | 1.5 ms (21x) |
| 1000 x 100 | 124.8 ms | 7.0 ms (18x) |
| 1000 x 400 | 503.0 ms | 28.9 ms (17x) |

The device scores agree with the host's to 1.6e-6 relative (the exponent is evaluated in single precision; the columns they rank at the top agree, which is what the sweep compares).

## Tried and rejected: coalesced reads of the Gram matrix (2026-11-14)
The 25 percent load efficiency is the uncoalesced-read signature. Variant: the centred Gram matrix is symmetric, so the pair (a, b), a < b, was read as row b, element a, with b the outer loop and the threads taking consecutive a (coalesced reads of one row per step). Interleaved A/B, 7 rounds, medians, whole call: m = 500, k = 100: 1.8 -> 6.9 ms; m = 1000, k = 400: 30.1 -> 39.0 ms; results equal to 1.5e-12. Slower at both sizes: in the original order each thread has a long, independent inner loop (ILP and L1/L2 reuse of its own row) while the new order has 500 to 1000 short, barrier-free passes in which most threads are idle for small b. The kernel is bound by the exponent and the double accumulation (DP efficiency 20.5 percent), not by the Gram reads. Reverted; the accumulator type placeholder `{ACC}` replaced the bare `double` words of the kernel text.
