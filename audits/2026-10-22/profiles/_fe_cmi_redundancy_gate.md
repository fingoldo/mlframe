# `_fe_cmi_redundancy_gate` family (`_fe_cmi_redundancy_gate`, `_fe_cmi_redundancy_null`): the greedy CMI-redundancy gate over the engineered candidates

Measured 2026-11-16, GTX 1050 Ti host (sm_61, 112 GB/s), 8 logical cores. One `apply_cmi_redundancy_gate` call: 30 engineered candidates (a third of them product features, the rest noisy sums of six base columns) over a binary target, 100000 rows, 10 bins, 25 permutations; 14 candidates kept on every path (the same 14 names).

## Whole pass, by path

| path | wall per pass |
|---|---|
| host (`MLFRAME_FE_GPU_STRICT=0`), before this work | 14.4 s |
| host, AUTO (the default: `_cmi_gpu_enabled` asks with p = the permutation count, below the STRICT floor) | 14.6 s |
| host after the null kernel work below | about 10.5 s on this machine at its quiet state |
| STRICT-GPU (`MLFRAME_FE_GPU_STRICT=1`) | 3.5 to 3.8 s |

## Host profile (cProfile, 160643 function calls)

14.43 s: `_null_njit` 11.0 s (231 calls, 48 ms each: the table draw of the conditional null, 76 percent), `_joint_entropy_two_dense_njit` 0.66 s, `_renumber_two_dense_njit` 0.46 s, `_combine_factorize_njit` 0.45 s, `dense_codes` 0.39 s. The null is a contingency-table draw per permutation over every stratum of the conditioning support, which is near-continuous as the gate admits features (tens of thousands of strata of two to four rows).

Work done on that kernel (null values bit for bit unchanged, test against the full-width margins): strata with at most one occupied x or y level are dropped before the draw (their table is fixed, their MI and Miller-Madow term are zero and they draw no random number: 40000 strata -> 19768 at 100000 rows), the zero margins are packed out of each stratum (a stratum of a few rows has two or three occupied levels, not ten), each stratum's contribution is added right after its table is drawn (no (strata, x, y) array), and the per-stratum scratch is reused (an allocation per stratum and permutation before). Isolated, 25 permutations at 100000 rows and 40000 strata: 147 ms -> 61 ms.

## Launches per call and the GPU profile (nvprof, one STRICT pass)

12223 driver launches and 9439 runtime launches for the pass of 30 candidates. Per null call (231 calls): `flat_strata_key_pack` 867 us (7.6 percent of the GPU time), the cuRAND key draw 666 us (5.8), `strata_xy_entropy` 267 us (4.7, 462 launches), the cupy copies (1111 launches, 4.4), the batched joint histogram `batched_joint_hist2_i32` 4.6 ms (4.2, 24 launches). The host-side cost of the pass is the batched cupy sort of the key matrix (1.77 s of 3.86 s cProfile, 463 calls) and the cuRAND draw; the sort is the device time surfacing. Kernel occupancy and bandwidth of the null path are in `_fe_cmi_perm_null_gpu.md` (`flat_strata_key_pack` achieved occupancy 0.89, 65 to 70 GB/s read of 112 GB/s) and of the batched CMI in `_fe_batched_mi*.md`.

## Backend choice and streaming

`_cmi_gpu_enabled(n, p, min_p)` decides STRICT / AUTO per call; the null backend is the `cmi_perm_null_backend` entry (`_fe_cmi_perm_null_gpu.perm_null_on_device`, host at every swept size on this card). The candidates are scored in column blocks sized by the free VRAM (`batched_cmi_gpu`), the null's permutation matrix is bounded by the `fe_gpu_has_vram_cushion` admission and the table draw keeps one stratum table per thread.
