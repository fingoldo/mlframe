# `_pair_mi_perm_cuda` (pair joint MI against K permuted targets, the inner loop of the order-2 maxT floor)

Measured 2026-11-13, GTX 1050 Ti (4 GB, 112 GB/s, sm_61), 15000 rows x 2000 pairs x 25 permuted binary-or-ten-class targets (10 bins per column, 10 classes), free host RAM above the FE reserve.

## What changed in this pass and why

The kernel replayed the `(joint, y_perm)` contingency table per target: for each of the K targets one pass over the rows, each pass re-reading the two code bytes of every row and re-forming the joint code. It now takes `KB` targets per pass (`KB` = 8, 4, 2 or 1, the largest whose shared histograms and joint counts fit the 44 KB shared budget; a macro, one compiled kernel per `KB`): a row's joint code is formed once and feeds `KB` shared histograms. Measured in isolation, same inputs, MI values equal to 1e-17: 1 target 151 ms, 2 targets 142, 4 targets 137, 8 targets 132 ms. The shared-memory atomic updates are unchanged in number, which is why the gain is 13 percent and not `KB`-fold.

## cProfile (main thread, 5 calls)

160 ms per call before the change; `cupy.ndarray.get` (the wait for the device) is 95 percent of it (0.763 s of 0.802 s over 5 calls), the Python of the launcher 15 ms in total. Device-bound; the device call against the CPU kernel `batch_pair_mi_perm_batched` (an `@njit` kernel in `info_theory`): CPU 650 ms, device 168 ms before the change and about 140 ms after; the two agree to 1e-16 (`max abs diff 1.04e-16`).

## nvprof, GPU summary

| | before (1 target per pass) | after (8 targets per pass) |
|---|---:|---:|
| `pair_mi_perm` per launch (4 launches) | 196.3 ms (min 151.7, max 256.0) | 110.2 ms (min 98.4, max 125.5) |
| share of GPU time | 99.8 % | 99.6 % |
| host-to-device copies, 24 | 1.2 ms in total | 1.3 ms |
| device-to-host copies, 4 | 0.29 ms | 0.31 ms |

(The first launch of a run includes the clock ramp, hence the spread; the ratio of the minima, 151.7 to 98.4 ms, is 1.5x.)

## nvprof metrics of the kernel (replayed, 4 launches)

Achieved occupancy 0.56 (one block per pair at about 33 KB of shared memory leaves one or two blocks per SM), multiprocessor activity 97.9 %, global load efficiency 59.6 % (byte loads), device-memory read throughput 2.6 GB/s of 112, shared-memory load throughput 85 GB/s. The kernel is bound by shared-memory atomic updates and the occupancy their histograms allow, not by DRAM: launches per call are one for the whole pool, the table is uploaded once, and the output is one `(K, n_pairs)` matrix.

## Column blocks

When the whole pool (`n_features * n` bytes of codes, `K * n` of targets, `K * n_pairs * 8` of output) does not fit twice in free VRAM, the pairs go in chunks that touch a bounded set of byte columns (`pair_chunks_by_operand_block`, shared with the usability |corr| backend): only those columns are gathered and uploaded for a chunk and the matrix is assembled on the host. Equal to the unblocked matrix (`test_pair_mi_perm_column_blocks.py`).

## Backend choice

The kernel-tuning entry `pair_mi_perm_floor` (sweep over rows x pairs against the CPU kernel) already existed and chooses between them per host; the multi-target pass above changes the device times it sweeps, so its salt was raised.

## Four-byte loads in `pair_mi_perm`, from the counters (2026-11-14)
Inside the default 100000 x 100 MRMR fit (one 461 ms launch) the counters were: achieved occupancy 43.7 percent, SM efficiency 99.5, DRAM read 2.8 GB/s, global load efficiency 58.0, store efficiency 5.1, shared efficiency 0.18, double-precision efficiency 9.5. The row loops read the two code columns and each of the KB targets one byte per thread. Change: four rows per thread with 4-byte loads (code columns, targets and the pair histogram `fx` loop), the leftover `n & 3` rows scalar; the byte matrices are uploaded with rows padded to a multiple of 4 (`stride`). Counts are integers; the changed register count changes the block size and with it the order of the final floating-point sum (max difference 2e-18).
Interleaved A/B, whole `_launch_pairs` (2000 pairs, 25 targets, 96 columns x 100,003 rows, ky 4, 5 rounds, medians): 372.1 -> 177.3 ms (2.1x). Counters after: global load efficiency 82.4 percent, DRAM read 20.9 GB/s, achieved occupancy 49.9 percent, double-precision efficiency 14.9. Tests: `test_pair_mi_perm_cuda.py`, `test_pair_mi_perm_column_blocks.py`, the per-arch offline recipes (KB = 1, 4, 8; a first version failed them with a NVRTC unused-variable warning, fixed) and the contract test.
