# `_cat_pair_fe` profile (cat x cat crosses)

## 2026-11-06: factorisation by hash and by counting, not by sorting every row

Standalone, host path, 100000 rows, the five categorical / small-integer columns of the fit-shaped frame (`family_scale._make`), all unordered pairs, `generate_cat_pair_crosses`: wall 1.05 s, of which `np.unique` (argsort of 100000 strings per column and of the 100000 combined integer keys per pair) was 93 percent (35 calls, 28 ms each).

Changed: the string columns go through `sorted_string_codes` (hash-factorise, sort only the distinct values) and the combined pair key through `unique_inverse_dense` (a `bincount` rank table for an integer vector whose range is small; it falls back to `np.unique` above `4 * rows + 1024`). Same sorted unique values and inverse codes, so the same codes, mappings and cell counts: checked against the previous code on a frame with strings, missing values and categoricals (21 crosses equal), and by `tests/feature_selection/fe/test_unique_inverse_dense.py`. Wall 1.05 -> 0.11 s.

Inside the 100000 x 100 GPU fit the family costs about 0.3 s (`_stage_cat_pair` in the per-family walls), so the fit-level gain is small; it grows with the number of categorical columns and the rows.

GPU twin: none. The work is a factorisation of host strings and small-integer keys; a device version would move the 100000-row key vectors across the bus for a counting pass that now takes 3 ms per pair on the host. Not built.

## 2026-11-16: contract measurements (host scoring, device pair pass)

Measured on the GTX 1050 Ti host (sm_61, 112 GB/s); hardware: GPU 78 C, SM 1695/1911 MHz, mem 3504 MHz, no throttle; CPU 2808/2808 MHz, load 19 %.

cProfile, `score_cat_pairs_by_interaction_information` on 100000 rows x 30 string columns (435 pairs, 4 / 8 / 16 / 30 levels): 0.465 s, 9478 function calls. `_codes` 0.420 s over 60 calls (30 columns, each asked twice), of which `column_string_codes` 0.393 s and the content key (`pandas.util.hash_array` over the object strings, 10 ms a column) 0.305 s; the joint-MI pass over all 435 pairs is the remaining 0.05 s. A hit of the column-coding cache costs 15.9 ms (the `to_numpy` of the arrow strings 4.6 ms, the key 10.6 ms) against 18.0 ms for coding without the cache, so the cache stays (tried bypassing it for object columns: slower); numeric columns hit in 0.26 ms against 4.4 ms.

Joint-MI pass over the 435 pairs (`cat_pair_search_backend` sweep inputs, 30 columns of 12 levels): njit prange 0.0718 s at 100000 rows and 0.5568 s at 500000; device (`mi_direct_gpu_batched_pairs`) 0.0092 s and 0.0466 s, 7.8 and 11.9 times faster, so the dispatch takes the device pass.

nvprof of the device pass at 500000 rows (4 calls): `compute_joint_hist_multi_pair_shared_cuda` 56.4 % of the GPU time (25.4 ms a launch, one launch per call, a launch per pair block), host-to-device copy of the codes 22.6 %, `cupy_copy` 12.1 %, `cupy_take` 5.9 % (2.7 ms), a segmented reduce 1.3 %. Launches per call: about 6. Achieved occupancy of the histogram kernel 0.987 at 93.4 GB/s device-memory read (83 percent of the card's 112 GB/s): the pass is bandwidth bound and utilization of the card during the launch is full; `cupy_take` 0.90 occupancy at 23 GB/s, the segmented reduce 0.49 at 93 GB/s.
