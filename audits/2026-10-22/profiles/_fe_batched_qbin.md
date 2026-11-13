# `_fe_batched_qbin` (born-on-device equi-frequency binning, `batched_quantile_bin_gpu`)

Measured 2026-11-13, GTX 1050 Ti (4 GB, 112 GB/s), resident float32 matrices, free host RAM above the FE reserve.

## What the module does per call

Three device stages for a whole `(n, K)` block: the edge matrix by one radix-select kernel (`radix_select_f32_v5`, preceded by the column-major `transpose_f32` it reads), the key table by a few elementwise ops over the small `(nbins + 2, K)` edge matrix, and the fused coder `qbin_code` (one launch for any `K`: 32-bit order-preserving keys for float32, 64-bit for float64, edges rounded up to float32 so the codes equal the float64 route's). Launches per call: radix select 1, transpose 1, coder 1, plus about 25 tiny elementwise launches on the edge table (microseconds each).

## cProfile (main thread, 100000 x 400 float32, 5 calls)

27.1 ms per call; the host side is nothing: `_edge_keys` 1.4 ms and `_codes_of_block` 2 ms cumulative per call of Python and launch time, the remainder is the wait for the device (`Device.synchronize` is the only large entry). The call is device-bound.

## nvprof, GPU summary (4 calls, 100000 x 400)

| kernel | share | per call |
|---|---:|---:|
| `radix_select_f32_v5` | 33.0 % | 10.9 ms |
| `qbin_code` | 28.8 % | 9.5 ms |
| `transpose_f32` | 18.7 % | 6.2 ms |
| `[CUDA memcpy HtoD]` (the one-time upload of the test matrix) | 19.3 % | - |
| 24 elementwise table kernels | under 0.1 % | 2 us each |

## nvprof metrics of `qbin_code` (replayed, 4 launches)

Achieved occupancy 0.90, multiprocessor activity 97.8 %, global load efficiency 100 %, global store efficiency 98.9 %, device-memory read throughput 15.3 GB/s and write throughput 27.4 GB/s (the int64 code matrix is 320 MB, the float32 input 160 MB). Utilization of the memory system is about 42 of 112 GB/s: the coder is a short integer compare loop per element, latency-bound rather than bandwidth-bound. Block size: 64 to 1024 threads measured, 256 (the occupancy default) is best or within noise (100000 x 100: 2.22 to 2.39 ms; 100000 x 400: 15.2 ms at 256, 16.6 at 1024).

## Variant measurements behind the decisions

* Float32 coder against the float64 route (resident, nbins 10-16): 100000 x 100 8.2 ms against 12.9, 100000 x 600 47.7 against 87.5, 30000 x 600 23.3 against 52.6; the codes are identical (`test_qbin_float32_coder.py`).
* Radix-select edges against `cp.percentile` (sort) edges, same codes: 2000 x 50 4.4 against 4.8 ms, 10000 x 50 3.4 against 6.9, 30000 x 200 6.7 against 64.8, 100000 x 100 7.2 against 137.5; the sort route runs out of memory at 100000 x 600 on this card. The radix route is never slower, so there is no host-dependent choice between the two for the kernel-tuning cache to make; it stays the default and `MLFRAME_FE_GPU_RADIX_EDGES=0` is the force.
* Host twin `quantile_bin_host` (parallel `@njit`): 340 ms at 100000 x 100 against the device's 7 ms on resident data, codes identical on ties, binary, constant and signed-zero columns. Every caller of the device function already holds device data, so there is no host-input call site whose host-or-device choice a tuner could change.

## Column blocks

The working set of one column is the contiguous input copy, the column-major copy, the int64 codes and the key stand-in, `n * (2 * itemsize + 8)` bytes. `_column_block_width` halves the width until the VRAM cushion admits it, and a matrix that does not fit goes through `block_cols` columns at a time into one preallocated codes matrix (a column's codes depend on that column only, so the result equals the single pass: `test_qbin_column_blocks.py`).
