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
