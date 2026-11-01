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
