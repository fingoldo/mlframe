# `_mi_greedy_device` (device scoring of the MI-greedy candidates)

Written 2026-11-09. Family `mi_greedy` (`_mi_greedy_fe`) had no device code: 1084 candidates per fit were materialised one by one as numpy columns (0.84 ms each, 0.91 s cumulative in the sequential cProfile), stacked in a DataFrame and scored, and only `top_k` of them survive.

Design: the (at most ten) distinct source columns are uploaded once, each candidate is a cupy expression of resident arrays (the host transforms written with cupy at the criterion dtype), the candidates of a block are stacked, scrubbed (`nan_to_num`) and tested for constancy (`std > 1e-12`) with one call each, and the block goes to the resident plug-in MI batch (`_plugin_mi_classif_batch_cuda_resident`, no transfer). Only the winners are materialised on the host by the unchanged `_materialise_candidate`, so the engineered columns, recipes and `transform` replay are the host's. Blocks of at most a quarter of the free device memory; a fault returns `None` and the host path runs.

Equivalence (`tests/feature_selection/fe/test_mi_greedy_device_matches_host.py`): the same kept candidates in the same order, MI within 2 % / 2e-4 (elementwise transcendentals can differ from numpy's in the last float32 bit; MI is computed on quantile codes, so a boundary row can move one bin), and `greedy_mi_fe_construct` returns the same winners with frames equal by `assert_frame_equal`.

First version: 155 / 170 / 161 ms at 5000 / 30000 / 100000 rows (host 58 / 148 / 410 ms). nvprof showed 1805 device-to-host copies and 900 `isnan` / `isinf` launches: a per-candidate `nan_to_num` and a per-candidate `float(col.std())` (a synchronisation each). Moving the scrub and the constancy test to one call per block: 150 candidates, synchronised, mean of 3 (`bench_mig.py` in D:/Temp): 5000 rows host 47 / device 21 ms; 30000 rows 144 / 24 ms; 100000 rows 403 / 33 ms.

Remaining device time at 100000 rows (nvprof, first version, shape unchanged): `radix_select_f32_v5` 4.0 ms per call (the quantile coder), `cupy_var_core` 3.5 ms, `concatenate` (the stack) 2.2 ms, `mi_hist_split_f32` 1.3 ms. Lever left: write the candidates straight into a preallocated `(k, n)` block (removes the stack) and fuse the nine unary / eight binary expressions into one kernel per transform.

Dispatch: kernel-tuning entry `fe_mi_greedy_candidates_device` (sweep over 5000 / 30000 / 100000 rows, equivalence rtol 1e-3 / atol 1e-4), no environment switch.
