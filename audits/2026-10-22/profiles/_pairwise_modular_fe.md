# Profile record: pairwise modular scan (`_pairwise_modular_fe`, with `_pairwise_modular_rank_gpu` and `_pairwise_modular_resident`)

Host: GTX 1050 Ti (sm_61, 4 GiB), 8 logical cores. Workload: integer columns on the 1000-row ranking subsample, 17 coarse moduli, pairs over the three operators. The kernel-level record
(parity, launches, triples) is `_pairwise_modular_rank_gpu.md`; this one covers the family and the candidate-block streaming added on 2026-11-14.

## cProfile
Whole-scan cProfile is in `_pairwise_modular_rank_gpu.md` (cProfile of `cheap_modular_scan`): the ranking is 9.65 s of 31.75 s at p=200 on the CPU path and the exact residue MI scan is the rest.

## nvprof / nsys
`residue_rank` runs one 128-thread block per candidate, launches per scan: 1 (all candidates in one grid); the streamed path adds 1 launch per candidate block.
nvprof counters fail on this machine today (profiling error 4211), so occupancy is the theoretical one from the launch bounds (128 threads, shared memory = counts of max_k x classes) and throughput is derived from timing below.

## Column blocks (streaming)
When the column matrix plus the residue table exceeds a quarter of the free device memory, `_score_candidate_blocks` halves the candidates and uploads for each half only the column blocks its candidates reference
(remapped), recursing until a block fits; a single candidate that does not fit returns None and the CPU kernel runs. Scores equal the unblocked call exactly (test `test_over_budget_candidates_stream_in_blocks_and_match`).

## Wall clock (median of 5, synchronized, warm)
| pairs | host njit | device, one call | device, forced candidate blocks |
|---|---|---|---|
| 190 (p=20) | 19.9 ms | 4.3 ms | 7.5 ms |
| 780 (p=40) | 78.3 ms | 8.5 ms | 14.6 ms |
| 3160 (p=80) | 321 ms | 25.4 ms | 30.1 ms |

Derived utilization: 3160 pairs x 3 operators x 17 moduli x 1000 rows = 161 M residue-class updates in 25.4 ms = 6.3 G updates/s on the device against 0.5 G/s on the host. Blocking costs one extra launch and upload per half, 1.2x at the largest size.
