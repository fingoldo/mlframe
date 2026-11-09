# FE module GPU / streaming / profiling contract (MRMR)

Opened 2026-10-25. Plan item added by the user: every feature-engineering module that MRMR uses must meet the contract below, and the contract is pinned by meta-tests.

## The contract

1. **CPU kernel and GPU kernel.** A numba CPU kernel and a fused GPU kernel (cupy RawKernel or numba.cuda), both kept as versions. Fused means one launch per unit of work with no intermediate table written to global memory (the pair test: one launch per block pair, the histogram in shared memory, the MI reduced in place); the profile record states the launches per call. The GPU kernel source carries no bare `double` or `float`: values and intermediates use the `{REAL}` placeholder (substituted from `crit_float_dtype()`, one compiled module per type, as in `hermite_fe/_als_kernels_gpu.py`), and the accumulator of a sum or a statistic compared with a threshold uses a separate `{ACC}` placeholder (float64 where the precision matters, with the reason stated), so that `{REAL}` cannot silently lower the precision of a reduction. Counts and indices stay integer types. The generic dispatchers are not a substitute when they are slower on the module's own shape (pair residual test: the generic dispatcher was 12 times slower than the CPU kernel at 80 thousand pairs, 5000 rows).
2. **Column-block streaming with a RAM and a VRAM bound.** No `n x p` temporary. Columns are converted and scored block by block, and a device that runs out of memory hands the block to the CPU kernel instead of failing the fit. First found in `_fe_sufficient_summary_pairs.py`: the wrapper built several `n x p` int64 matrices, about 6 GiB at n=100k, p=2000.
3. **`crit_dtype`** for the stored and scored values (`_crit_np_dtype()`, float32 by default).
4. **Dispatch through the kernel-tuning cache (KTC)**: `kernel_tuner(...)` registration, a sweep over the module's own axes, and a measured fallback heuristic.
5. **Profiled twice and optimized**: cProfile for the host path and a GPU profiler (`nsys`, `ncu` or `nvprof`), each with the numbers kept in `audits/2026-10-22/profiles/<module>.md`. The GPU profile covers the whole-GPU picture, not only the hotspot kernel: achieved occupancy and its limiter, SM throughput, memory throughput (DRAM bandwidth), launch statistics (grid, block, registers, shared memory) and the share of the wall time spent in kernels against copies and host work. `ncu` sections: SpeedOfLight, Occupancy, LaunchStats, MemoryWorkloadAnalysis.

Profilers on the development host (GTX 1050 Ti, Pascal sm_61): `nsys` (Nsight Systems 2022.4.2, not on PATH) and `nvprof` (CUDA 12.8; CUPTI on PATH, counter metrics need an elevated shell) work. `ncu` is installed (2022.3.0 and 2025.2.1) but refuses this GPU ("Profiling is not supported on device 0": Nsight Compute dropped Pascal), so occupancy and throughput come from `nvprof --metrics` here and from `ncu` on Turing or newer. See CUDA.md.

## Status

* Compliant: `_fe_sufficient_summary_pairs` (contract points 1 to 4 in place; the GPU profile of the fused kernel is still to be recorded).
* Everything else: not yet checked against points 2 to 5. The static table below covers point 1 only, and only by text markers.
* Meta-tests: planned in `tests/test_meta/`, as a ratchet over the pending modules (the list only shrinks) plus a hard check on every compliant module.

## Static inventory (GPU markers and dispatcher markers by source text)

| module | kind | GPU markers | dispatcher markers | lines | static verdict |
|---|---|---:|---:|---:|---|
| `_binned_numeric_agg_fe` | module | 4 | 0 | 778 | GPU code, no dispatcher |
| `_cat_pair_fe` | module | 0 | 0 | 590 | no GPU marker |
| `_cat_triple_fe` | module | 0 | 0 | 586 | no GPU marker |
| `_composite_group_agg_fe` | module | 0 | 0 | 651 | no GPU marker |
| `_conditional_gate_fe` | module | 43 | 0 | 957 | GPU code, no dispatcher |
| `_conditional_quantile_rank_fe` | module | 0 | 0 | 307 | no GPU marker |
| `_confirm_predictor_engineered` | module | 0 | 0 | 327 | no GPU marker |
| `_count_freq_interaction_fe` | module | 0 | 0 | 581 | no GPU marker |
| `_extra_fe_families` | module | 3 | 0 | 944 | GPU code, no dispatcher |
| `_family_maxt_gate` | module | 7 | 0 | 652 | GPU code, no dispatcher |
| `_fe_accuracy_gate` | module | 0 | 0 | 332 | no GPU marker |
| `_fe_additive_fusion` | module | 21 | 0 | 526 | GPU code, no dispatcher |
| `_fe_auto_escalation` | module | 5 | 0 | 748 | GPU code, no dispatcher |
| `_fe_batched_mi` | module | 219 | 3 | 986 | GPU code + dispatcher |
| `_fe_budget_learning` | missing | 0 | 0 | 0 | no GPU marker |
| `_fe_cmi_redundancy_gate` | module | 41 | 1 | 874 | GPU code + dispatcher |
| `_fe_deadline` | module | 0 | 0 | 118 | no GPU marker |
| `_fe_family_budget` | module | 0 | 0 | 301 | no GPU marker |
| `_fe_family_timing` | module | 1 | 0 | 128 | GPU code, no dispatcher |
| `_fe_final_subsumption` | module | 0 | 0 | 54 | no GPU marker |
| `_fe_frame_ops` | module | 0 | 0 | 177 | no GPU marker |
| `_fe_gpu_strict` | module | 30 | 1 | 301 | GPU code + dispatcher |
| `_fe_gpu_vram` | module | 52 | 0 | 237 | GPU code, no dispatcher |
| `_fe_linear_explainability` | module | 0 | 0 | 114 | no GPU marker |
| `_fe_pure_form_retention` | module | 28 | 0 | 882 | GPU code, no dispatcher |
| `_fe_raw_redundancy_drop` | module | 13 | 0 | 912 | GPU code, no dispatcher |
| `_fe_raw_redundancy_helpers` | module | 14 | 0 | 480 | GPU code, no dispatcher |
| `_fe_rejection_ledger` | module | 0 | 0 | 394 | no GPU marker |
| `_fe_resident_operands` | module | 26 | 0 | 383 | GPU code, no dispatcher |
| `_fe_retention_subsumption` | module | 27 | 0 | 251 | GPU code, no dispatcher |
| `_fe_roster_attrs` | missing | 0 | 0 | 0 | no GPU marker |
| `_fe_rung_schedule` | module | 0 | 9 | 254 | no GPU marker |
| `_fe_stability_vote` | module | 0 | 0 | 387 | no GPU marker |
| `_fe_stage_cascade_early_a` | missing | 0 | 0 | 0 | no GPU marker |
| `_fe_stage_cascade_early_b` | missing | 0 | 0 | 0 | no GPU marker |
| `_fe_stage_cascade_mid_a` | missing | 0 | 0 | 0 | no GPU marker |
| `_fe_stage_cascade_mid_b` | missing | 0 | 0 | 0 | no GPU marker |
| `_fe_stage_cascade_run` | missing | 0 | 0 | 0 | no GPU marker |
| `_fe_stage_merge` | missing | 0 | 0 | 0 | no GPU marker |
| `_fe_stage_offset_product` | missing | 0 | 0 | 0 | no GPU marker |
| `_fe_stage_oof_warp` | missing | 0 | 0 | 0 | no GPU marker |
| `_fe_stage_row_stat` | missing | 0 | 0 | 0 | no GPU marker |
| `_fe_stage_temporal_agg` | missing | 0 | 0 | 0 | no GPU marker |
| `_fe_subsample` | module | 17 | 0 | 452 | GPU code, no dispatcher |
| `_fe_sufficient_summary` | module | 0 | 0 | 660 | no GPU marker |
| `_fe_synergy_screen` | module | 0 | 0 | 300 | no GPU marker |
| `_fe_usability_signal` | module | 2 | 0 | 456 | GPU code, no dispatcher |
| `_feature_engineering_pairs` | package | 392 | 41 | 8430 | GPU code + dispatcher |
| `_fourier_detect_cap` | module | 0 | 0 | 47 | no GPU marker |
| `_gpu_strict_fe` | package | 147 | 0 | 695 | GPU code, no dispatcher |
| `_group_distance_fe` | module | 0 | 0 | 456 | no GPU marker |
| `_grouped_agg_fe` | module | 0 | 0 | 559 | no GPU marker |
| `_grouped_quantile_fe` | module | 0 | 0 | 682 | no GPU marker |
| `_hinge_basis_fe` | module | 16 | 0 | 919 | GPU code, no dispatcher |
| `_hybrid_orth_family_variants` | missing | 0 | 0 | 0 | no GPU marker |
| `_integer_lattice_fe` | module | 0 | 0 | 428 | no GPU marker |
| `_lof_fe` | module | 1 | 0 | 302 | GPU code, no dispatcher |
| `_mahalanobis_density_fe` | module | 0 | 0 | 223 | no GPU marker |
| `_meta_fe_recommender` | module | 0 | 0 | 450 | no GPU marker |
| `_mi_greedy_cmi_fe` | module | 87 | 0 | 875 | GPU code, no dispatcher |
| `_mi_greedy_fe` | module | 0 | 0 | 692 | no GPU marker |
| `_missingness_fe` | module | 0 | 0 | 577 | no GPU marker |
| `_mrmr_fe_provenance` | module | 0 | 0 | 637 | no GPU marker |
| `_mrmr_fe_step` | package | 105 | 5 | 5173 | GPU code + dispatcher |
| `_mrmr_fe_step_helpers` | module | 9 | 6 | 796 | GPU code + dispatcher |
| `_numeric_decompose_fe` | module | 0 | 0 | 447 | no GPU marker |
| `_offset_product_fe` | module | 4 | 1 | 356 | GPU code + dispatcher |
| `_oof_cell2d_fe` | module | 0 | 0 | 149 | no GPU marker |
| `_oof_warp_fe` | module | 0 | 0 | 171 | no GPU marker |
| `_ordinal_pattern_fe` | module | 0 | 0 | 342 | no GPU marker |
| `_orthogonal_adaptive_arity_fe` | module | 15 | 0 | 840 | GPU code, no dispatcher |
| `_orthogonal_adaptive_degree_fe` | module | 0 | 0 | 385 | no GPU marker |
| `_orthogonal_bootstrap_mi_fe` | module | 0 | 0 | 462 | no GPU marker |
| `_orthogonal_cluster_basis_fe` | module | 0 | 0 | 833 | no GPU marker |
| `_orthogonal_cmim_fe` | module | 0 | 0 | 638 | no GPU marker |
| `_orthogonal_copula_mi_fe` | module | 0 | 0 | 486 | no GPU marker |
| `_orthogonal_dcor_fe` | module | 0 | 0 | 491 | no GPU marker |
| `_orthogonal_diff_basis_fe` | module | 0 | 0 | 680 | no GPU marker |
| `_orthogonal_elasticnet_fe` | module | 0 | 0 | 367 | no GPU marker |
| `_orthogonal_hsic_fe` | module | 0 | 0 | 659 | no GPU marker |
| `_orthogonal_jmim_fe` | module | 0 | 0 | 510 | no GPU marker |
| `_orthogonal_ksg_mi_fe` | module | 0 | 0 | 397 | no GPU marker |
| `_orthogonal_lasso_fe` | module | 0 | 0 | 454 | no GPU marker |
| `_orthogonal_meta_scorer_fe` | module | 0 | 3 | 572 | no GPU marker |
| `_orthogonal_quadruplet_fe` | module | 4 | 0 | 641 | GPU code, no dispatcher |
| `_orthogonal_routing_fe` | module | 0 | 0 | 619 | no GPU marker |
| `_orthogonal_scorer_auto_fe` | module | 0 | 1 | 667 | no GPU marker |
| `_orthogonal_three_gate_mi_fe` | module | 0 | 0 | 710 | no GPU marker |
| `_orthogonal_total_correlation_fe` | module | 0 | 0 | 625 | no GPU marker |
| `_orthogonal_triplet_fe` | module | 6 | 0 | 943 | GPU code, no dispatcher |
| `_orthogonal_univariate_fe` | package | 583 | 6 | 7789 | GPU code + dispatcher |
| `_pair_residual_screen` | module | 0 | 0 | 141 | no GPU marker |
| `_pairwise_modular_fe` | module | 36 | 0 | 588 | GPU code, no dispatcher |
| `_parity_fe` | module | 1 | 0 | 158 | GPU code, no dispatcher |
| `_periodic_fe` | module | 0 | 0 | 607 | no GPU marker |
| `_permutation_null_pair_resident` | module | 45 | 0 | 256 | GPU code, no dispatcher |
| `_random_fourier_features_fe` | module | 0 | 0 | 293 | no GPU marker |
| `_ratio_delta_fe` | module | 0 | 0 | 776 | no GPU marker |
| `_residual_rescue_fe` | module | 0 | 0 | 304 | no GPU marker |
| `_row_stat_fe` | module | 0 | 0 | 243 | no GPU marker |
| `_semi_supervised_fe` | module | 0 | 0 | 210 | no GPU marker |
| `_sliced_inverse_regression_fe` | module | 1 | 0 | 302 | GPU code, no dispatcher |
| `_step_pair_order` | missing | 0 | 0 | 0 | no GPU marker |
| `_step_pairmi` | missing | 0 | 0 | 0 | no GPU marker |
| `_step_pairs_rank` | missing | 0 | 0 | 0 | no GPU marker |
| `_synergy_detector` | module | 0 | 10 | 211 | no GPU marker |
| `_target_encoding_fe` | module | 0 | 0 | 670 | no GPU marker |
| `_target_mix_fe` | module | 0 | 0 | 145 | no GPU marker |
| `_temporal_agg_fe` | module | 0 | 0 | 817 | no GPU marker |
| `_unified_fe_gate` | module | 2 | 0 | 545 | GPU code, no dispatcher |
| `_wavelet_basis_fe_recipes` | module | 0 | 0 | 372 | no GPU marker |
| `_y_encoding` | module | 0 | 0 | 169 | no GPU marker |
| `batch_pair_mi_gpu` | module | 124 | 45 | 841 | GPU code + dispatcher |
| `batch_pair_usability_corr_gpu` | module | 90 | 27 | 743 | GPU code + dispatcher |
| `cat_fe_state` | module | 0 | 0 | 299 | no GPU marker |
| `engineered_recipes` | package | 136 | 0 | 3751 | GPU code, no dispatcher |
| `feature_engineering` | module | 72 | 0 | 723 | GPU code, no dispatcher |
| `polynom_pair_fe` | module | 3 | 2 | 772 | GPU code + dispatcher |

118 modules: no GPU marker=78, GPU code, no dispatcher=29, GPU code + dispatcher=11

## Which of the no-GPU-marker families would repay a kernel (2026-11-06)

Measured per-family walls inside the 100000 x 100 GPU fit (per-family walls in `profiles/stage_sample_100k_gpu_2026_11_05.md` and the per-module notes): `_cat_pair_fe` about 0.3 s, `_cat_triple_fe` about 0.75 s, `_count_freq_interaction_fe` and count encoding well under 0.3 s after the factorisation changes, `_conditional_quantile_rank_fe` and the other `mid_b` host families 20 s together on four stage threads that overlap with the GPU stages. A device kernel for the first three would save under 1 s of a 70 to 83 s fit and would add host-to-device copies of string-derived codes; the `mid_b` families do not shorten when given more host threads (prefetch experiment, same note), so their time is shared with other host work and a kernel is the only lever left for them. That makes `_conditional_quantile_rank_fe` (per-bin sort plus row-level `searchsorted`, the kind of kernel the repo already has for modular and quantile codes) the first candidate if a kernel is written; the cat-pair, cat-triple and count-frequency families are left on the host path.

### group_distance is not a kernel candidate (2026-11-06)

Standalone, host path with the numba prange kernels, 100000 rows, 3 group columns x 8 numeric columns (24 pairs): 0.38 s wall (a first check that passed the arguments in the wrong order grouped by float columns and took 167 s; the signature is `(X, group_cols, num_cols)`). The 5.64 s the family takes inside the 100000 x 100 fit comes from its pair count and from sharing the host threads with the other `mid_b` families, not from per-pair cost, so a segmented-merge Wasserstein kernel would not shorten it; the fit-level lever is the number of pairs it evaluates. The next families by fit wall that have no device twin and a per-call cost worth a kernel are `binned_numeric_agg` (9.4 s), `grouped_quantile` (5.9 s) and `modular` (4.6 s, which has a device path for the codes already).

## GPU coverage measured, not by text markers (2026-11-09, `fam_gpu_attr.py` / `fam_gpu_an.py` in D:/Temp)

Method: default `MRMR`, 100000 x 100 (`family_scale._make`), stage concurrency off (`MLFRAME_FE_STAGE_CONCURRENCY=0`) so families do not overlap, under `nvprof --print-gpu-trace`. Every ledger record (`fe_timed` family, `gate:` lap, `stage:` row) is logged with its start and end; a kernel is charged to the shortest record that contains its GPU start time (fe_timed families before gates and stages). The trace clock is aligned by an anchor kernel launched before the fit. Caveats: launches are asynchronous, so a kernel can start slightly after the record that issued it ended (charged to the next record or to none); nvprof inflates the wall (85.6 s against 71-75 s). The static table above counts text markers and is superseded by this one for the question 'does the family put work on the device'.

Whole run: the `fe_timed` families spent 42.1 s of which 7.3 s were GPU kernel time (17 %).

### Families with at least 0.4 s of wall and under 10 % of it on the device

| family | wall s | GPU kernel s | kernels | GPU % |
|---|---:|---:|---:|---:|
| `orth_mi_auto_scorer` | 2.32 | 0.132 | 110 | 5.7 |
| `orth_mi_meta` | 1.74 | 0.075 | 41 | 4.3 |
| `grouped_quantile` | 0.91 | 0.000 | 0 | 0.0 |
| `wavelet` | 0.86 | 0.052 | 3507 | 6.0 |
| `orth_mi_bootstrap` | 0.86 | 0.005 | 69 | 0.6 |
| `mi_greedy` | 0.76 | 0.000 | 0 | 0.0 |
| `group_distance` | 0.67 | 0.000 | 0 | 0.0 |
| `ordinal_pattern_te` | 0.58 | 0.000 | 0 | 0.0 |
| `composite_group_agg` | 0.58 | 0.000 | 0 | 0.0 |
| `kfold_target_encode` | 0.50 | 0.017 | 69 | 3.4 |
| `orth_mi_tc` | 0.43 | 0.006 | 5 | 1.4 |

### Full-frame stages (outside the shared 30000-row FE subsample) with at least 0.9 s of wall and no device work attributed

| stage | wall s | GPU kernel s | kernels |
|---|---:|---:|---:|
| `stage:screen_predictors` | 8.50 | 0.021 | 21 |
| `stage:fdr_gain_floor` | 4.90 | 0.000 | 0 |
| `stage:assign_support_tail` | 4.56 | 0.000 | 0 |
| `stage:support_fe_max_steps_pass` | 3.59 | 0.002 | 108 |
| `stage:honest_accept_block` | 2.81 | 0.000 | 0 |
| `stage:discretize_inputs` | 2.42 | 0.000 | 0 |
| `stage:operand_pool` | 1.85 | 0.000 | 0 |
| `stage:categorize_dataset` | 1.70 | 0.000 | 0 |
| `stage:accuracy_gate` | 1.47 | 0.000 | 0 |
| `stage:pure_form_side_check` | 1.15 | 0.000 | 0 |
| `stage:dcd_discover_and_swap` | 1.00 | 0.000 | 0 |
| `stage:pair_mis_and_floor` | 1.00 | 0.000 | 0 |

Families with no kernel at all (any wall): `grouped_quantile`, `mi_greedy`, `group_distance`, `ordinal_pattern_te`, `composite_group_agg`, `random_fourier_features_block`, `conditional_residual`, `conditional_dispersion`, `orth_mi_copula`, `grouped_agg`, `rounding`, `diff_basis`, `digit`, `mahalanobis_density`, `rare_category`, `rankgauss`, `triplet_cross_basis`, `modular`, `pairwise_mi_legacy_sweep`, `missingness_pattern`, `missingness_count`, `cluster_basis`, `pairwise_ratio`, `pairwise_log_ratio`, `lagged_diff`, `grouped_delta`

Reading: the families listed are the ones to give a profiled, streamed, KTC-dispatched device twin per `fe_gpu_coverage.md` contract points 1-5, ordered by wall. A zero in the stage table means no kernel STARTED inside the stage: the stage may still call device code that the attribution gave to a finer record, so each stage is checked in its source before a kernel is written.

### Correction to the measured table above (2026-11-09, after reading the sources of the zero-kernel rows)

A zero in the table does not prove that a family has no device path. `grouped_quantile` is listed with 0 kernels, yet `_grouped_quantile_mi_device` (resident plug-in MI batch through `_scorer_device_dispatch.scorer_device_choice`, kernel-tuning entry `grouped_quantile`) exists and was called (0.29 s cumulative in the cProfile of the sequential fit); the 0.9 s that remain in the family are the host side of `_p_added_by_column` (binning the columns for the added-information G-test, 1.4 s cumulative across 3 calls) and the generator. Kernels of that call start after the record that issued them has closed, or are charged to a neighbouring record. The attribution is therefore an upper bound on the missing device work, and each family listed has to be confirmed in its source before a kernel is written. Confirmed gaps so far: the dependence scorers of the bootstrap / ensemble / meta / auto-scorer families (`_orthogonal_xi_fe`, `_orthogonal_copula_mi_fe`, `_orthogonal_dcor_fe`, `_orthogonal_hsic_fe`, `_orthogonal_tail_dependence_fe`: no cupy in the modules); device twins of xi and copula written, profiled, and left behind the tuner because the host `prange` radix kernel is faster on this card (`profiles/_orthogonal_xi_gpu.md`, `profiles/_orthogonal_copula_gpu.md`).

nvprof metrics of those twins at 100000 rows x 40 columns (elevated shell, `nv_xi.ps1` in D:/Temp): `xi_gather` achieved occupancy 0.88, global load efficiency 22 %; `xi_inverse_perm` occupancy 0.84, global store efficiency 12.5 % (the scatter); `xi_walk` occupancy 0.99, load efficiency 27 % (double indirection through `ra[order]`), 9.8 ms per call; `copula_joint` occupancy 0.74, load efficiency 71 %, shared-memory efficiency 5 % (bank conflicts of the 400-bin histogram); cub radix sort 37.6 ms over the 960 sort launches of three xi and three copula calls. What the exact-equality contract rules out: the permutations that break the column ties come from the host generator (pinned by `test_xi_batch_stream.py`, `test_xi_batch_kernel.py`, `test_xi_radix_kernel.py`), and drawing them is 66 of the 99 ms of a 30000 x 100 host batch; a device twin must be fed them over PCIe, so it cannot beat the host while those pins stand. Replacing the generator stream by a counter-based tie-break key would remove the draws on both paths (selection-equivalent, values change for tied columns) and is a decision for the pinned tests, not taken here.
