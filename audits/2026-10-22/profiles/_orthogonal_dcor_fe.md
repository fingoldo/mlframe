# Profile record: distance correlation of many columns (`_orthogonal_dcor_fe`, device twin `_dcor_gpu`)

Host: GTX 1050 Ti (sm_61, 4 GiB), 8 logical cores, elevated process (hardware counters). Measured 2026-11-14, medians of 5 after a warm call. The first profile of the device twin is `_dcor_gpu.md`.

## Kernels and choice
Host: `_dcor_columns_kernel` (numba, per column the double-centred distance moments `dcov2` and `dvar2_x` against the centred label distance matrix, O(m^2) per column). Device: `dcor_cols` in `_dcor_gpu.py`, one block per column with the column's subsample in shared memory, four per-thread sums (row sum, `sum d^2`, `sum d * B[i, q]`, `(row / m)^2`) reduced by the block; everything is in the accumulator type (`{ACC}`, double): `dvar2_x` is `sum d^2 - 2 m sum r^2 + m^2 g^2`, a cancellation a narrow sum would not survive (a narrow-value variant measured 45 ms against 46-49 ms, no gain, 2026-11-05). Columns go up in column blocks bounded by a quarter of the free VRAM. The backend is chosen by `dcor_backend` through the kernel-tuning cache (`dcor_columns_backend`, swept over (m, k); the sweep is also registered with the central registry so `retune_all` refreshes it), with a pre-sweep floor of one million pair terms.

## cProfile (device call, m = 1000, k = 400)
The call takes 56 ms cumulative in `dcor_columns_gpu`; its own time and the two `cupy.array` uploads are about 1 ms, nothing else on the host exceeds 0.1 ms: the call is its kernel wait.

## nvprof (hardware counters)
`dcor_cols` (launches per call: 1 per column block, one here): 52.7 ms of the 54.6 ms traced GPU time (96.4 percent; the uploads 3.6 percent). Achieved occupancy 61.7 percent (the shared scratch of m doubles plus 4 x 256 partials limits the resident blocks), double-precision efficiency 65.0 percent, DRAM read throughput 2.7 GB/s (2.4 percent of the card's 112 GB/s peak bandwidth), global load efficiency 25.0 percent (each thread reads its own row of the distance matrix). Utilization: the kernel is at 65 percent of the double-precision rate of a card that runs double at 1/32 of single, so it is arithmetic-bound near its ceiling; there is no memory or occupancy headroom of comparable size. Lowering the work needs a cheaper formulation of the double-centred sums, not a kernel layout change.

## Wall clock (numba against the device call, uploads included)
| m x k | numba | device |
|---|---|---|
| 500 x 100 | 30.2 ms | 4.6 ms (6.6x) |
| 1000 x 100 | 129.0 ms | 17.5 ms (7.4x) |
| 1000 x 400 | 516.0 ms | 56.3 ms (9.2x) |

Device and host agree to 4.2e-13 relative.
