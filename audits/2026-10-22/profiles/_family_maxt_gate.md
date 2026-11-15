# Profile record: family max-T null (`_family_maxt_gate`; device kernels in `_family_maxt_cuda_bits`, host bit-set null in `_family_maxt_null_v2_njit`)

Host: GTX 1050 Ti (sm_61, 4 GiB), 8 logical cores, elevated process (hardware counters). Measured 2026-11-14, medians of 3 after a warm call. The kernel-level history (v2 to v3, the register-pressure registry entry, the attempts that failed) is in `_family_maxt_cuda_bits.md`.

## Pieces and choice
Host: `family_null_max_njit` (numba `prange` over columns) and the bit-set null `null_max_bitsets` of `_family_maxt_null_v2_njit`. Device: the `family_null_bits`, `family_null_bits_v2` and `family_null_bits_v3` kernels of `_family_maxt_cuda_bits` (one block per group of columns and permutations, bin bit sets of the columns AND the label bit sets by `popcll`), with the bit sets staged by `DeviceBitsets` and the permutations streamed in blocks (`_family_maxt_stream`, `_family_maxt_stack`; column blocks sized by the free VRAM). The backend is chosen by `_family_null_backend` through the kernel-tuning cache (`family_null_backend`, swept over (n_rows, n_cols); the sweep is also registered with the central registry so `retune_all` refreshes it), with a work-based pre-sweep fallback.

## Wall clock (99 permutations, 8 bins, 4 classes; the host bit-set null against the device call with the bit-set upload)
| n_rows x n_cols | host | device |
|---|---|---|
| 2,000 x 200 | 1.6 ms | 1.3 ms (1.2x) |
| 20,000 x 200 | 12.6 ms | 4.3 ms (2.9x) |
| 20,000 x 2,000 | 128.1 ms | 25.6 ms (5.0x) |
| 100,000 x 1,000 | 343.8 ms | 66.4 ms (5.2x) |

Maxima agree to 1.5e-15 absolute. The device wins from the smallest swept size and increasingly with rows and columns; the sweep decides per host.

## cProfile (device call, 20,000 x 2,000)
12 ms of the call is the kernel wait (the single `cupy max` read of the result, 15 ms in the profile, carries the queued kernel time); the host side is below 1 ms (two module loads, one SHA of the source key).

## nvprof (hardware counters)
Launches per call: 1 (`family_null_bits_v2`, which four classes keep; v3 serves one to three classes with up to 8 bins). 12.1 ms of the 12.35 ms traced GPU time (98 percent; the uploads 1.4 percent). `family_null_bits_v2`: achieved occupancy 54.5 percent, double-precision efficiency 1.0, DRAM read throughput 9.1 GB/s (8 percent of the card's 112 GB/s peak bandwidth), global load efficiency 97.5 percent. The v3 counters (IPC 2.20, 100 registers, memory-dependency stalls 51.6 percent, load efficiency 98.1) are in `_family_maxt_cuda_bits.md`. Utilization: the kernels are read-coalesced and not DRAM-bound; they are bound by the dependent popcount chain and the L2 re-reads of the bin bit sets (190 GB/s from L2 for v2).
