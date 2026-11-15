# Profile record: triple MI scan of the orthogonal triplet FE (`_orthogonal_triplet_fe`, device twin `_triple_mi_gpu`)

Host: GTX 1050 Ti (sm_61, 4 GiB), 8 logical cores, elevated process (hardware counters). Measured 2026-11-14, medians of 3 after a warm call. The kernel history (the 4-byte loads, 2.0x) is in `_triple_mi_gpu.md`.

## Kernels and choice
Host: `_triple_mi_kernel` in `_orthogonal_triplet_fe.py` (numba, `prange` over the first index; the triples of joint codes in lexicographic order). Device: `triple_mi` in `_triple_mi_gpu.py`, one block per triple, a shared `(base^3 cell, class)` integer table filled four rows per thread with 4-byte loads from code rows padded to 4 bytes, the MI tail spread over the block. Triples go to the device in blocks (`TRIPLES_PER_LAUNCH`); a block that touches every column uses the full byte code matrix (uploaded once), a block that touches fewer columns uploads only those columns (a column block of the code matrix, remapped indices). The choice is `use_device_triples`: `MLFRAME_TRIPLE_MI_DEVICE=0|1` forces a path, else the per-host kernel-tuning entry `triple_mi_backend` (rows x triples), else the `DEVICE_MIN_WORK` floor.

## cProfile (device call, 59 columns = 32,509 triples, 20,000 rows)
The call is its kernel wait: the host side is the five uploads (1 ms) and the triple enumeration (`nonzero`, 1 ms).

## nvprof (hardware counters)
`triple_mi` (launches per call: 1 per block of triples, one here): 37.6 ms of the 37.9 ms traced GPU time (99 percent; the uploads 0.7 percent). Achieved occupancy 44.2 percent, double-precision efficiency 36.5, DRAM read throughput 0.56 GB/s (the byte code matrix, 1.2 MB, lives in L2; 0.5 percent of the card's 112 GB/s peak bandwidth), global load efficiency 99.3 percent. Utilization: the kernel is limited by the shared-memory atomics and the logarithms of the MI tail (DP efficiency 36.5 percent of the double rate), not by memory.

## Wall clock (numba scan against the device call, uploads included, base 6, 2 classes)
| columns (triples) | numba | device |
|---|---|---|
| 7 (35) | 1.4 ms | 0.7 ms (2.0x) |
| 15 (455) | 5.2 ms | 1.7 ms (3.1x) |
| 29 (3,654) | 30.8 ms | 5.5 ms (5.6x) |
| 59 (32,509) | 245.7 ms | 41.7 ms (5.9x) |

Agreement 2.9e-15 relative on every triple; blocks of 7 triples (touching few columns) equal the single launch exactly (`tests/feature_selection/gpu/test_triple_mi_row_remainders.py`). The device wins at every swept size on this host; the sweep and the floor still decide per host.

## 2026-11-15: `_orthogonal_triplet_recipes` carved under this family

`_orthogonal_triplet_recipes` holds the recipe construction and the with-recipes entry point of the triplet family; it was carved out of `_orthogonal_triplet_fe` for file size and the parent re-exports every name defined there. It has no kernel and no device code of its own, so it is judged with the parent and no counters were taken for it; the triplet's device work is recorded above (`_triple_mi_gpu`).
