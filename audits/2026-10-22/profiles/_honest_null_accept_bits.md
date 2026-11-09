# `_honest_null_accept_bits.root_split_accept` (acceptance kernel of the honest MDLP null)

2026-11-09. The only kernel on the serial main-thread path of the 100000 x 100 fit (the root-split acceptance of the honest null that sets the screening's gain floor and the candidate gates): 430 ms per launch in the fit, one launch per 100-column x 399-permutation block. Bench: `bench_rsa.py` in D:/Temp (`root_split_accept_gpu` on 100 normal columns of 100000 rows, 399 permutations, 40 % ones, a planted subset), nvprof metrics in an elevated shell.

## Diagnosis (version 1, thread-serial exact scan)

Achieved occupancy 0.75, SM activity 97.5 %, DRAM read 4 GB/s (not memory-bound), global load efficiency 66 %, warp execution efficiency 87 %, issue slot utilisation 44.7 %, **issue stalls 65 % on synchronisation**. Instrumented counts (atomic counters in a copy of the source, 39900 pairs): every one of the 3125 words per pair holds a candidate cut, the prune keeps 40 words per pair (1.3 %), and those words hold 589 candidate cuts per pair that are scanned exactly. The surviving words are contiguous (near where the random walk of the label sequence peaks), so they belong to 5 to 7 of the 512 threads of the block; each of those threads evaluated about 84 gains (six dependent table lookups each) while the other 505 waited at the next barrier. The block width sweep agreed (`MLFRAME_GPU_BLOCK_HONEST_ACCEPT` 128 / 256 / 384 / 512: 690 / 529 / 448 / 415 ms per call): the time was per-block serial latency, not throughput.

## Change (version 2, the default)

The prune loop no longer scans: a surviving word is appended to a shared queue (`q_w`, `q_l`: word index and the ones count before it; 512 entries, an overflow falls back to the version-1 scan in the owning thread). After a barrier the exact scans are shared out over the block, one warp per queued word and one lane per bit (`cand >> lane`): the gain expression is unchanged, and the winner rule is "larger gain, then smaller index" in the lane loop as in the block reduction, so the result is the thread-serial one. Version 1 stays compiled behind `ACCEPT_NOQUEUE` (`_MODULE_NOQUEUE`).

Equivalence: the 10 existing tests (accept booleans, gain, index and candidate counts equal to the exact k4 scan for the shared, global and precomputed variants) plus a new test that the default kernel equals version 1 and equals a build whose queue holds two words (every pair overflows into the thread-serial path); 23 tests with the k3 / k4 suites.

## Result (same bench, GTX 1050 Ti)

Kernel time per call (nvprof): 335 ms -> 205 ms average (149 to 238 ms across the four calls; the machine's other load moves it). nvprof metrics after: issue stalls on synchronisation 65 % -> 17.9 %, issue slot utilisation 44.7 % -> 71.4 %, warp execution efficiency 87 % -> 97.8 %, SM activity 99.7 %, achieved occupancy 0.75 (unchanged: `__launch_bounds__(512, 4)` and 22 KB of dynamic shared memory), global load efficiency 72 %. Remaining lever: the label packing (`ballot` of the row-space bit table through the column's sort order) and the bound evaluation of all 3125 words are now the cost; the packed sorted-space table of the `pre` variant gave the same time before this change, so it is worth re-measuring after it.
