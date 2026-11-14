# Profile record: pair interaction screen on the additive residual (`_pair_residual_screen`, device twin `_pair_residual_gpu`)

Host: GTX 1050 Ti (sm_61, 4 GiB), 8 logical cores, elevated process (hardware counters). Measured 2026-11-14, medians of 5 after a warm call. Workload: 40 columns of 8 bins over 20,000 rows (the screen's column cap), all 780 pairs, a residual carrying a parity interaction of two columns.

## What was added
`_score_pairs` (numba `prange` over pairs, two passes over the rows per pair) now has a device twin chosen by `score_pairs_dispatch` through the kernel-tuning entry `pair_residual_backend` (rows x pairs). `pair_residual_t`: one 256-thread block per pair, the even rows add their residual (as a 2^-k fixed-point integer, so the sums do not depend on the thread order) and their count to a shared cell table, the per-cell shrunk means are formed once per block, then the odd rows give the sum of `d = e^2 - (e - m)^2` and the sum of `(d - mean)^2` (two-pass in place of the host's Welford loop: equal to rounding). The host reorders the rows so the even ones come first, which removes the mask load and the divergent branch of half the threads from every pass. The bin codes go up as bytes for the pairs of a block (column block of the pair list, halved on OOM).

## Steps measured (whole device call, 780 pairs x 20,000 rows, interleaved with the numba pass in the same run)
| version | device | numba |
|---|---|---|
| first (per-row division by the cell count, masked passes) | 25.0 ms | 13.7 ms |
| per-cell mean computed once per block | 9.4 ms | 13.0 ms |
| rows reordered even-first on the host (fancy index of the int64 matrix cost 4 ms) | 12.8 ms | 13.9 ms |
| rows reordered on the device after a byte upload | 7.8 ms | 13.2 ms |

Final wall clock by size: 10 pairs 1.68 ms (numba 0.32), 40 pairs 3.81 (0.79), 160 pairs 4.56 (3.53), 780 pairs 7.81 ms (13.21, 1.7x). The device wins from about 300 pairs, the screen's own maximum is 780; the sweep decides per host.

## cProfile
The device call at 780 pairs: 8 ms cumulative in `block_work`, 2 ms of own time, uploads 1 ms (5 `cupy.array` calls); no host loop above 0.1 ms.

## nvprof (hardware counters, kernel alone)
`pair_residual_t` (launches per call: 1 for all pairs): kernel time 6.2 ms before the row reorder; after it achieved occupancy 96.6 percent, double-precision efficiency 31.2, DRAM read throughput 0.23 GB/s (the byte codes and the residual are re-read from L2 by every pair), global load efficiency 88.1 (it was 54.5 with the masked passes). The kernel is limited by the double arithmetic of the odd-row passes and the shared atomics of the even pass, not by memory bandwidth (112 GB/s peak, 0.2 percent used).

Equality: t statistics agree with the numba pass to 6e-14 relative on all pairs; `pair_interaction_screen` reports the same pairs with the device forced on, and the test counts the device call (`tests/feature_selection/gpu/test_pair_residual_device.py`).
