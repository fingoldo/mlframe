# _triple_mi_gpu (plug-in MI of every triple of joint quantile codes)

Kernel `triple_mi` in `src/mlframe/feature_selection/filters/_triple_mi_gpu.py`; dispatched from `_orthogonal_triplet_fe._triple_mi_scores` (called by `_enumerate_triplet_candidates`). Host twin: `_triple_mi_kernel` (parallel njit), kept for tables that do not fit shared memory,
more than 16 classes, `MLFRAME_TRIPLE_MI_DEVICE=0` and small work (`DEVICE_MIN_WORK` = 5 M row-triples).

Design: one block per triple, threads stride the rows once reading three int8 codes and the class, counting into a shared `base**3 x n_y` table with `atomicAdd`; the MI tail is spread over the block (class totals per thread, one term per cell, an ordered sum by thread 0).

Measured (GTX 1050 Ti, one process, 5 levels, 2 classes, 100000 rows; host = the prange kernel):

| columns | triples | device | host |
|---|---|---|---|
| 40 | 9880 | 64 ms | 419 ms |
| 60 | 34220 | 211 ms | 1318 ms |
| 12 (20000 rows, 3 classes) | 220 | 2 ms | 4 ms |

The first version finished the MI in thread 0 alone: 129 and 439 ms (3x over the host); spreading the tail halved it. Triples and order equal the host's, MI within 2e-17.

GPU profile of that first version (nvprof, 40 columns): achieved occupancy 0.57, multiprocessor activity 97.9%, global load efficiency 99.8%, device memory utilization low, shared memory efficiency 0.14% (random atomic hits on 250 bins),
warp execution efficiency 77% (the serial tail). Remaining bound: the shared atomics (about 15 G row-updates per second); a per-warp private table merged at the end would trade that for 32x the table, not tried.
Tests: `tests/feature_selection/fe/test_triple_mi_gpu.py`.

## Four-byte loads in `triple_mi`, from the counters (2026-11-14)
Inside the default 100000 x 100 MRMR fit (one 727 ms launch, 2.8 percent of the GPU activity) the counters were: achieved occupancy 51.4 percent, SM efficiency 96.3, DRAM read 8.7 GB/s, global load efficiency 72.5, shared efficiency 0.55, double-precision efficiency 7.4. The kernel issued four 1-byte loads per row (three code columns and the class byte). Change: each thread loads four consecutive rows of every array with one 4-byte load and counts them, the up-to-three leftover rows go through the scalar tail; the code matrix is uploaded with rows padded to a multiple of 4 bytes (`stride`) so every row starts on a 4-byte boundary and the pad is never counted. Outputs are bit-identical.
Interleaved A/B, kernel only (4000 triples of 100,000 rows, base 6, 2 classes, 7 rounds): 48.7 -> 20.1 ms (2.4x). Whole `triple_mi_gpu` call including uploads and the host triple enumeration, 34,220 triples, 100,001 rows (not a multiple of 4), 5 interleaved rounds, medians: 476 -> 242 ms (2.0x). Counters after: global load efficiency 85.9 percent, DRAM read 20.0 GB/s, achieved occupancy 49.0 percent. Test: `tests/feature_selection/gpu/test_triple_mi_row_remainders.py` (row counts of every remainder modulo 4 against an independent numpy MI) plus the existing triple MI tests.
