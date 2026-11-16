# `_fe_raw_redundancy_drop` family (`_fe_raw_redundancy_drop`, `_fe_raw_redundancy_anchors`, `_fe_raw_redundancy_helpers`): the final raw-versus-engineered conditional-redundancy drop

Measured 2026-11-16, GTX 1050 Ti host (sm_61, 112 GB/s), 8 logical cores. One `drop_redundant_raw_operands` call on the seed-909 fixture of `test_raw_redundancy_sibling_fusion` scaled to 100000 rows: three raw operands (`a`, `b`, `c`) and two engineered survivors (`add(a,sin(c))`, `div(sqr(a),exp(b))`), a classification target. The same result on every path: all three raws dropped, both engineered kept.

## Whole pass, by path

| path | wall per pass |
|---|---|
| host (`MLFRAME_FE_GPU_STRICT=0`) | 0.04 s |
| AUTO with the fit state of a 100000 x 100 fit (STRICT engages from the fit size) | 0.04 s |
| STRICT-GPU (forced) | 0.07 s |

The pass touches the final selection only (a handful of raws and their children), so it is a decision layer, not a hotspot: the device path is slower here (0.07 s) because a launch costs more than the host's work on 5 columns.

## Host profile and launches

cProfile of one host pass: 0.065 s for 4648 function calls; the kernels it spends time in are the partition entropies of the observed CMI (`_entropy_from_classes_njit` 0.009 s over 44 calls, `_renumber_two_dense_njit` 0.007 s, `_joint_entropy_two_dense_njit` 0.007 s over 20 calls, `_combine_factorize_njit` 0.005 s) and two numpy calls of the fine re-binning of the engineered survivors (`searchsorted` 0.005 s, `partition` 0.005 s); the 14 `_conditional_perm_null` calls (0.0116 s) are all answered by the analytic floor `_analytic_cmi_null` (14 calls, 0.0104 s), so no permutation draw runs on this fixture (counted in the profile: no `_null_njit` / `StratifiedNull` call).

The cost sits in the calls it makes, not in its own code: the observed CMI (`_cmi_from_binned`, njit partition kernels) and the conditional permutation null (`_conditional_perm_null` of `_fe_cmi_redundancy_gate`: the contingency-table draw `StratifiedNull` on the host, `conditional_perm_null_gpu` on the device). Their kernels, launches per call, nvprof occupancy and throughput are in `_fe_cmi_redundancy_gate.md`, `_fe_cmi_perm_null_gpu.md` and `_fe_batched_mi*.md`; this family adds no kernel of its own. On the STRICT path the launches per pass are those of three null calls plus the batched CMI round (about 30 launches per null call).

## Backend choice and memory

Choice: `_cmi_gpu_enabled` (STRICT / AUTO by fit size) and the `cmi_perm_null_backend` entry for the null, as in the gate. Memory: the final selection's binned columns, a column block of the selected operands and their engineered children, bounded by the selection size (tens of columns) and independent of the candidate pool.
