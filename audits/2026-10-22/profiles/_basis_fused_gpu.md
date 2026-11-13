# `_basis_fused_gpu` (one kernel for the orthogonal-polynomial design `B[i, k] = T_k(z[i])`)

Measured 2026-11-14, GTX 1050 Ti (sm_61, 112 GB/s), float32 resident column, 1600000 rows x 7 columns (degree 6), hermite basis. The column is resident when the design is built (device-born column of the ALS seed), so the figures are the build alone: no upload, the design stays on the card.

## This pass: coalesced stores through a shared tile

The kernel gave each thread one row and wrote its `nc` values at `B + i * nc`: global store efficiency 14 % (nvprof), write throughput 29 to 30 GB/s of 112. A block now owns a tile of consecutive rows, the threads run their recurrences into a shared tile and the block writes it out as one contiguous run of the row-major design (`threads * n_columns * itemsize` bytes of shared memory, the block halved while the tile exceeds 32 KB, so 40 double columns still fit).

| rows | kernel before | kernel after | cupy loop |
| --- | --- | --- | --- |
| 20000 | 0.063 ms | 0.057 ms | 0.97 ms |
| 100000 | 0.105 | 0.060 | 0.84 |
| 400000 | 0.383 | 0.144 | 2.72 |
| 1600000 | 1.528 | 0.577 | 11.1 |

Output bit-identical to the cupy loop (the existing degree x basis suite, plus ragged row counts 1, 31, 33, 257, 4099 with 1, 2 and 40 columns and a double design against the numpy recurrence).

nvprof metrics, 1.6M rows, 6 launches each: before achieved occupancy 0.85, multiprocessor activity 96 to 100 %, store efficiency 14.0 %, device-memory write throughput 28.9 to 30.4 GB/s, read 4.7 to 9.1 GB/s; after occupancy 0.97, multiprocessor activity 85 to 100 %, store efficiency 96.7 to 100 %, write throughput 74.8 to 80.5 GB/s (67 to 72 % of the card's bandwidth), read 10.9 to 16.2 GB/s.

## Launches per call

One `build_basis` launch per design (the cupy loop it replaces issues about 8 launches per degree: ~60 for a degree-6 pair). GPU time per launch 0.62 ms in the nvprof summary at 1.6M rows (profiler serialisation included).

## Host profile (cProfile, 20 calls)

0.004 s for 20 calls (0.2 ms per call, 941 function calls): the Python around the launch (block size from the occupancy rule, the argument tuple); the work is asynchronous and surfaces at the consumer's first blocking read.

## Backend choice

Entry `fe_basis_build_backend` (`_basis_fused_gpu._SPEC`): the single kernel against the cupy loop, both on the resident column, bit equality required. Swept for real: the kernel at 5000, 100000 and 1000000 rows (max difference 0.0); fallback "fused". `MLFRAME_FE_BASIS_FUSED=0` still forces the loop. The residency is the placement the caller has (a device-born column), stated in the entry; there is no host-in variant, the host design (`hermite_fe.build_basis_matrix`, njit) is a different code path chosen upstream.

## Streaming

The design is one block of at most `MAX_COLUMNS` (40) columns; the ALS solve reads all of it, so there is nothing to stream. A request the VRAM cushion refuses returns `None` before allocating.
