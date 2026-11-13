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
