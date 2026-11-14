# Pending contract modules: inventory (2026-11-14)

Generated from `contract_violations` and the import relation among the 167 pending modules of `tests/test_meta/_fe_gpu_contract.json`. Columns: module, source lines, missing contract points, pending modules that import it (a module imported by one or more others is a candidate carved child of that family).

| module | lines | missing | imported by (pending) |
|---|---:|---|---|
| `_gpu_resident_basis` | 969 | CPU KTC STREAM PROFILE | `_gpu_resident_cross_basis`, `_gpu_resident_fe`, `_orth_gpu_resident`, `_orthogonal_adaptive_degree_device` ... |
| `_extra_fe_families` | 957 | GPU KTC PROFILE | `_extra_fe_families_dispersion`, `_extra_fe_families_dispersion_resident`, `_lof_fe`, `_offset_product_fe` ... |
| `_fe_raw_redundancy_drop` | 947 | CPU GPU KTC STREAM PROFILE | `_fe_additive_fusion`, `_fe_additive_fusion_gpu_resident`, `_fe_raw_redundancy_anchors`, `_fe_raw_redundancy_helpers` ... |
| `_fe_cmi_redundancy_gate` | 937 | CPU GPU KTC STREAM PROFILE | `_fe_additive_fusion`, `_fe_additive_fusion_gpu_resident`, `_fe_auto_escalation`, `_fe_cmi_perm_null_gpu` ... |
| `_hermite_fe_optimise_pair` | 928 | GPU KTC STREAM PROFILE | `_cupy_polynom_optimizer`, `_hermite_fe_optimise` |
| `_grouped_quantile_fe` | 915 | CPU GPU KTC REC:has no cProfile REC:has no GPU profiler (nsys / nc REC:does not state the kernel laun REC:has no occupancy REC:has no whole-GPU throughput /  | `_group_distance_fe`, `_orthogonal_triplet_fe` |
| `_family_maxt_gate` | 903 | GPU KTC STREAM PROFILE | `_periodic_fe` |
| `_fe_pure_form_retention` | 889 | CPU GPU KTC STREAM PROFILE | `_fe_pure_form_retention_gpu_resident`, `_fe_retention_subsumption`, `_fe_subsample` |
| `_conditional_gate_fe` | 876 | GPU KTC STREAM REC:has no cProfile REC:has no GPU profiler (nsys / nc REC:does not state the kernel laun REC:has no occupancy REC:has no whole-GPU throughput /  | `_conditional_gate_fe_helpers`, `_conditional_gate_hybrid`, `_conditional_gate_kernels`, `_conditional_gate_naming` ... |
| `_orthogonal_three_gate_mi_fe` | 870 | CPU GPU KTC STREAM PROFILE |  |
| `_batch_pair_mi_cuda_kernels` | 869 | KTC STREAM PROFILE | `batch_pair_mi_gpu` |
| `batch_pair_mi_gpu` | 862 | CPU GPU STREAM PROFILE | `_batch_pair_mi_cuda_kernels`, `_fe_batch_dispatch`, `_fe_synergy_exhaustive`, `_gpu_pairs` |
| `_numba_polynom_optimizer` | 861 | GPU KTC STREAM PROFILE | `_cupy_polynom_optimizer`, `_hermite_fe_optimise`, `_hermite_fe_optimise_pair`, `_hermite_fe_optimise_parts` |
| `_hermite_fe_optimise` | 843 | GPU KTC STREAM PROFILE | `_hermite_fe_diverse`, `_hermite_fe_optimise_helpers`, `_hermite_fe_optimise_pair`, `_hermite_fe_optimise_parts` ... |
| `_orthogonal_adaptive_arity_fe` | 839 | CPU GPU KTC STREAM PROFILE |  |
| `_orthogonal_triplet_fe` | 838 | GPU KTC STREAM PROFILE | `_orthogonal_triplet_fe_recipes`, `_orthogonal_triplet_recipes` |
| `_temporal_agg_fe` | 834 | GPU KTC STREAM PROFILE | `_ratio_delta_fe`, `_temporal_agg_fe_rolling` |
| `_orthogonal_hsic_fe` | 817 | GPU KTC STREAM PROFILE | `_orth_auto_scorer_fe`, `_orthogonal_meta_scorer_fe`, `_orthogonal_scorer_auto_fe` |
| `_fe_batched_mi` | 802 | CPU KTC REC:has no cProfile REC:does not state the kernel laun REC:has no whole-GPU throughput /  | `_cupy_polynom_optimizer`, `_fe_additive_fusion`, `_fe_additive_fusion_gpu_resident`, `_fe_batched_mi_cmi` ... |
| `_mrmr_fe_step_helpers` | 802 | CPU GPU KTC STREAM PROFILE |  |
| `_orth_basis_recipes` | 801 | GPU KTC STREAM PROFILE | `_extra_basis_resident` |
| `_pairwise_modular_fe` | 800 | GPU KTC STREAM PROFILE | `_conditional_gate_fe`, `_conditional_gate_fe_helpers`, `_conditional_gate_hybrid`, `_orth_mi_backends` ... |
| `_fe_batched_mi_cmi` | 796 | CPU KTC STREAM REC:has no cProfile REC:does not state the kernel laun | `_fe_batched_mi`, `_fe_batched_mi_cmi_hist`, `_fe_batched_mi_cmi_lut`, `_fe_cmi_fused_backend` |
| `_hermite_fe_mi` | 787 | KTC | `_conditional_gate_fe`, `_extra_fe_families_dispersion_resident`, `_fe_batched_mi_guards`, `_fe_edge_mi` ... |
| `polynom_pair_fe` | 786 | CPU GPU KTC STREAM PROFILE | `_fe_subsample` |
| `_target_encoding_fe` | 785 | GPU KTC STREAM PROFILE | `_cat_pair_fe`, `_cat_target_encoding_and_weighted`, `_cat_triple_fe`, `_count_freq_interaction_fe` ... |
| `_ratio_delta_fe` | 780 | CPU GPU KTC STREAM PROFILE | `_missingness_ratio_recipes`, `_unified_fe_gate` |
| `_periodic_fe` | 776 | GPU KTC STREAM REC:has no cProfile REC:has no GPU profiler (nsys / nc REC:does not state the kernel laun REC:has no occupancy REC:has no whole-GPU throughput /  | `_pairwise_modular_fe`, `_parity_fe` |
| `_orth_auto_scorer_fe` | 775 | CPU GPU KTC STREAM REC:has no cProfile REC:has no GPU profiler (nsys / nc REC:does not state the kernel laun REC:has no occupancy | `_orthogonal_scorer_auto_fe`, `_orthogonal_xi_fe`, `_pairwise_modular_fe` |
| `_orth_extra_basis_fe` | 766 | GPU KTC STREAM REC:has no GPU profiler (nsys / nc REC:does not state the kernel laun REC:has no occupancy REC:has no whole-GPU throughput /  | `_extra_fe_families`, `_fe_auto_escalation`, `_fe_auto_escalation_proposers`, `_fourier_detect_batch` ... |
| `_fe_auto_escalation` | 759 | CPU GPU KTC STREAM PROFILE | `_fe_auto_escalation_proposers` |
| `_gpu_resident_fe` | 754 | CPU KTC STREAM REC:has no cProfile REC:does not state the kernel laun REC:has no occupancy REC:has no whole-GPU throughput /  | `_fe_auto_escalation`, `_fe_gpu_batch`, `_gpu_resident_basis`, `_hermite_fe_mi` ... |
| `feature_engineering` | 747 | GPU KTC STREAM PROFILE | `_fe_cpu_batch`, `_feature_engineering_mem_budget`, `_gpu_resident_fe`, `_offset_product_fe` |
| `_mi_greedy_fe` | 739 | CPU GPU KTC STREAM PROFILE | `_mi_greedy_device`, `_missingness_ratio_recipes` |
| `_orth_mi_backends` | 724 | CPU GPU KTC PROFILE | `_conditional_gate_fe`, `_extra_basis_resident`, `_extra_fe_families_dispersion`, `_fe_cpu_batch` ... |
| `_extra_fe_families_dispersion` | 720 | GPU KTC PROFILE | `_extra_fe_families`, `_extra_fe_families_dispersion_resident` |
| `_orthogonal_scorer_auto_fe` | 713 | CPU GPU KTC STREAM PROFILE | `_orth_auto_scorer_fe`, `_orthogonal_elasticnet_fe`, `_orthogonal_xi_fe` |
| `_orthogonal_cmim_fe` | 709 | CPU GPU KTC STREAM PROFILE | `_orthogonal_meta_scorer_fe` |
| `_orthogonal_quadruplet_fe` | 689 | CPU GPU KTC STREAM PROFILE | `_orthogonal_quadruplet_fe_recipes` |
| `_orth_pair_cross_fe` | 688 | CPU GPU KTC STREAM PROFILE | `_gpu_resident_cross_basis`, `_orth_basis_recipes`, `_orthogonal_adaptive_arity_fe`, `_orthogonal_quadruplet_fe` ... |
| `_orthogonal_routing_fe` | 680 | CPU GPU KTC STREAM PROFILE | `_basis_cells_ktc`, `_orthogonal_routing_device` |
| `_orthogonal_meta_scorer_fe` | 671 | GPU KTC STREAM PROFILE |  |
| `_fe_sufficient_summary` | 661 | CPU GPU KTC STREAM PROFILE |  |
| `_fe_batched_mi_cmi_hist` | 650 | CPU KTC STREAM PROFILE | `_fe_batched_mi_cmi` |
| `_fe_cmi_redundancy_null` | 633 | CPU GPU KTC STREAM PROFILE | `_fe_cmi_redundancy_gate` |
| `_cat_triple_fe` | 617 | CPU GPU KTC STREAM REC:has no cProfile REC:has no GPU profiler (nsys / nc REC:does not state the kernel laun REC:has no occupancy REC:has no whole-GPU throughput /  | `_encoding_recipes` |
| `_group_distance_fe` | 613 | GPU KTC STREAM PROFILE |  |
| `_cat_pair_fe` | 603 | CPU GPU KTC STREAM REC:has no cProfile REC:has no GPU profiler (nsys / nc REC:does not state the kernel laun REC:has no occupancy REC:has no whole-GPU throughput /  | `_cat_triple_fe`, `_encoding_recipes` |
| `_orthogonal_copula_mi_fe` | 603 | GPU KTC STREAM PROFILE | `_orth_auto_scorer_fe`, `_orthogonal_copula_gpu`, `_orthogonal_meta_scorer_fe`, `_orthogonal_scorer_auto_fe` ... |
| `_unified_fe_gate` | 594 | CPU GPU KTC STREAM REC:has no cProfile REC:has no GPU profiler (nsys / nc REC:does not state the kernel laun REC:has no occupancy REC:has no whole-GPU throughput /  | `_count_freq_interaction_fe`, `_extra_fe_families`, `_extra_fe_families_dispersion`, `_extra_fe_families_dispersion_resident` ... |
| `_count_freq_interaction_fe` | 582 | CPU GPU KTC STREAM REC:has no cProfile REC:has no GPU profiler (nsys / nc REC:does not state the kernel laun REC:has no occupancy REC:has no whole-GPU throughput /  | `_encoding_recipes`, `_unified_fe_gate` |
| `_missingness_fe` | 580 | GPU KTC DTYPE STREAM PROFILE | `_missingness_ratio_recipes`, `_unified_fe_gate` |
| `_numeric_decompose_fe` | 579 | GPU KTC PROFILE | `_periodic_fe` |
| `_fe_gpu_batch` | 568 | CPU KTC PROFILE | `_extra_basis_resident`, `_extra_fe_families_dispersion_resident`, `_fe_batch_dispatch`, `_fe_cpu_batch` ... |
| `_hermite_prewarp` | 549 | GPU KTC STREAM PROFILE | `_hermite_prewarp_gpu_resident` |
| `_grouped_agg_fe` | 546 | CPU GPU KTC STREAM PROFILE |  |
| `_orth_dedup` | 545 | GPU KTC STREAM PROFILE | `_orth_gpu_resident`, `_orth_scoring_memo`, `_orthogonal_meta_scorer_fe` |
| `_orthogonal_dcor_fe` | 533 | GPU KTC STREAM PROFILE | `_orth_auto_scorer_fe`, `_orthogonal_hsic_fe`, `_orthogonal_meta_scorer_fe`, `_orthogonal_scorer_auto_fe` |
| `_fe_cmi_perm_null_gpu` | 527 | CPU KTC STREAM PROFILE | `_fe_cmi_redundancy_null` |
| `_fe_additive_fusion` | 525 | CPU GPU KTC PROFILE | `_fe_additive_fusion_gpu_resident` |
| `_orthogonal_bootstrap_mi_fe` | 510 | CPU GPU KTC STREAM PROFILE | `_numeric_decompose_fe`, `_periodic_fe` |
| `_fe_additive_fusion_gpu_resident` | 503 | CPU KTC STREAM PROFILE | `_fe_additive_fusion` |
| `_fe_raw_redundancy_helpers` | 480 | CPU GPU KTC STREAM PROFILE | `_fe_additive_fusion`, `_fe_raw_redundancy_anchors`, `_fe_raw_redundancy_drop` |
| `_orth_extra_basis_fe_helpers` | 471 | GPU KTC STREAM PROFILE | `_orth_extra_basis_fe` |
| `_fe_raw_redundancy_anchors` | 465 | CPU GPU KTC STREAM PROFILE | `_fe_raw_redundancy_drop` |
| `_fe_usability_signal` | 457 | GPU KTC STREAM PROFILE | `_cat_pair_fe`, `_categorical_encoding`, `_conditional_gate_fe`, `_count_freq_interaction_fe` ... |
| `_orthogonal_lasso_fe` | 454 | CPU GPU KTC STREAM PROFILE |  |
| `_fe_subsample` | 453 | GPU KTC STREAM PROFILE | `_fe_pure_form_retention`, `_fe_sufficient_summary`, `polynom_pair_fe` |
| `_pairwise_modular_rank_gpu` | 453 | CPU STREAM REC:does not state the kernel laun | `_pairwise_modular_fe` |
| `_feature_engineering_mem_budget` | 444 | CPU GPU KTC DTYPE STREAM PROFILE | `_periodic_fe`, `feature_engineering` |
| `_fe_synergy_exhaustive` | 438 | CPU GPU KTC STREAM PROFILE | `_mrmr_fe_step_helpers` |
| `_fe_stability_vote` | 431 | CPU GPU KTC STREAM PROFILE |  |
| `_orthogonal_ksg_mi_fe` | 421 | CPU GPU KTC STREAM PROFILE | `_orth_auto_scorer_fe`, `_orthogonal_hsic_fe`, `_orthogonal_meta_scorer_fe`, `_orthogonal_scorer_auto_fe` |
| `_gpu_resident_cross_basis` | 415 | CPU KTC STREAM PROFILE | `_extra_basis_resident`, `_orth_pair_cross_fe`, `_orthogonal_adaptive_arity_fe`, `_orthogonal_quadruplet_fe` ... |
| `_hermite_basis_eval` | 409 | GPU KTC STREAM PROFILE |  |
| `_fourier_detect_gpu_resident` | 401 | CPU KTC STREAM PROFILE | `_fourier_detect_batch`, `_fourier_fused_gpu`, `_orth_extra_basis_fe` |
| `_fe_batched_mi_mm_values` | 394 | CPU KTC STREAM PROFILE | `_fe_batched_mi` |
| `_hermite_robust` | 394 | GPU KTC STREAM PROFILE | `_gpu_resident_basis`, `_hermite_prewarp`, `_orth_dedup`, `_orth_scoring_memo` |
| `_wavelet_basis_fe_recipes` | 392 | CPU GPU KTC STREAM PROFILE |  |
| `_wavelet_basis_fe_batched` | 376 | CPU KTC STREAM PROFILE | `_extra_basis_resident`, `_fe_batched_mi` |
| `_residual_rescue_fe` | 375 | GPU KTC STREAM PROFILE |  |
| `_encoding_recipes` | 372 | CPU GPU KTC DTYPE STREAM PROFILE |  |
| `_offset_product_fe` | 371 | CPU GPU KTC STREAM PROFILE |  |
| `_extra_fe_families_dispersion_resident` | 369 | CPU GPU KTC STREAM PROFILE | `_extra_fe_families_dispersion`, `_gpu_resident_cross_basis` |
| `_orthogonal_elasticnet_fe` | 367 | CPU GPU KTC STREAM PROFILE |  |
| `_fe_auto_escalation_proposers` | 362 | CPU GPU KTC STREAM PROFILE | `_fe_auto_escalation` |
| `_orth_gpu_resident` | 360 | CPU GPU KTC STREAM PROFILE |  |
| `_dcd_pair_su_batch` | 359 | GPU KTC STREAM PROFILE |  |
| `_lof_fe` | 314 | CPU GPU KTC STREAM PROFILE |  |
| `_temporal_agg_fe_rolling` | 310 | GPU KTC STREAM PROFILE | `_temporal_agg_fe` |
| `_fe_retention_subsumption` | 306 | CPU GPU KTC STREAM PROFILE |  |
| `_gpu_pairs` | 305 | CPU KTC STREAM PROFILE |  |
| `_fe_synergy_screen` | 299 | GPU KTC STREAM PROFILE | `_synergy_detector` |
| `_fe_matrix_io` | 297 | CPU GPU KTC STREAM PROFILE |  |
| `_hermite_prewarp_gpu_resident` | 292 | CPU GPU KTC STREAM PROFILE | `_hermite_prewarp` |
| `_hermite_fe_optimise_parts` | 290 | CPU GPU KTC STREAM PROFILE | `_hermite_fe_optimise` |
| `_fe_pure_form_retention_gpu_resident` | 289 | CPU KTC STREAM PROFILE | `_fe_pure_form_retention` |
| `_orth_mi_null_max` | 281 | GPU KTC PROFILE | `_unified_fe_gate` |
| `_cupy_polynom_optimizer` | 279 | CPU GPU KTC STREAM PROFILE | `_fe_batched_mi`, `_hermite_fe_optimise_pair` |
| `_cat_target_encoding_and_weighted` | 273 | GPU KTC STREAM PROFILE | `_target_encoding_fe` |
| `_permutation_null_pair_resident` | 258 | CPU GPU KTC STREAM PROFILE | `_mrmr_fe_step_helpers`, `batch_pair_mi_gpu` |
| `_mi_greedy_device` | 248 | CPU GPU REC:does not state the kernel laun REC:has no occupancy REC:has no whole-GPU throughput /  | `_mi_greedy_fe` |
| `_pairwise_modular_resident` | 248 | CPU GPU KTC STREAM PROFILE | `_pairwise_modular_fe` |
| `_row_stat_fe` | 245 | CPU GPU KTC STREAM PROFILE | `_usability_row_stat_pool` |
| `_orthogonal_xi_fe` | 244 | GPU KTC STREAM PROFILE | `_orth_auto_scorer_fe`, `_orthogonal_scorer_auto_fe` |
| `_extra_basis_resident` | 222 | CPU GPU KTC STREAM PROFILE | `_extra_basis_fourier_cuda`, `_orth_basis_recipes` |
| `_orthogonal_copula_gpu` | 220 | CPU STREAM REC:has no cProfile REC:does not state the kernel laun REC:has no whole-GPU throughput /  | `_orth_auto_scorer_fe`, `_orthogonal_copula_mi_fe` |
| `_orthogonal_triplet_recipes` | 219 | CPU GPU KTC STREAM PROFILE | `_orthogonal_triplet_fe` |
| `_conditional_gate_hybrid` | 216 | GPU KTC STREAM PROFILE | `_conditional_gate_fe` |
| `_fe_batched_mi_cmi_lut` | 215 | CPU KTC STREAM PROFILE | `_fe_batched_mi_cmi`, `_fe_batched_mi_cmi_hist` |
| `_parity_fe` | 213 | GPU KTC STREAM PROFILE |  |
| `_screen_predictors_gate` | 213 | CPU GPU KTC DTYPE STREAM PROFILE |  |
| `_synergy_detector` | 213 | CPU GPU KTC STREAM PROFILE |  |
| `_fe_batched_qbin` | 211 | KTC | `_fe_batched_mi` |
| `_class_encoding` | 198 | GPU KTC STREAM PROFILE | `_dcd_pair_su_batch` |
| `_fourier_fused_gpu` | 197 | CPU KTC STREAM PROFILE | `_fourier_detect_gpu_resident` |
| `_orthogonal_tail_dependence_fe` | 194 | GPU KTC STREAM REC:has no cProfile REC:does not state the kernel laun REC:has no occupancy REC:has no whole-GPU throughput /  | `_orth_auto_scorer_fe`, `_orthogonal_scorer_auto_fe` |
| `_oof_warp_fe` | 186 | CPU GPU KTC STREAM PROFILE |  |
| `_target_encoding_order_stats` | 167 | GPU KTC DTYPE STREAM PROFILE | `_target_encoding_fe` |
| `_hermite_oracle` | 159 | CPU GPU KTC STREAM PROFILE | `_hermite_fe_mi` |
| `_orth_scoring_memo` | 158 | CPU GPU KTC STREAM PROFILE |  |
| `_fourier_detect_batch` | 156 | CPU GPU STREAM PROFILE | `_fe_auto_escalation_proposers` |
| `_missingness_ratio_recipes` | 151 | CPU GPU KTC STREAM PROFILE |  |
| `_oof_cell2d_fe` | 148 | CPU GPU KTC STREAM PROFILE |  |
| `_categorical_encoding` | 144 | CPU GPU KTC STREAM PROFILE |  |
| `_orthogonal_routing_device` | 142 | CPU GPU KTC STREAM REC:does not state the kernel laun REC:has no whole-GPU throughput /  | `_basis_cells_ktc`, `_orthogonal_routing_fe` |
| `_pair_residual_screen` | 140 | GPU KTC STREAM PROFILE | `_oof_cell2d_fe`, `_residual_rescue_fe`, `_target_mix_gate` |
| `_orthogonal_adaptive_degree_device` | 136 | CPU GPU KTC STREAM PROFILE |  |
| `_orthogonal_shared` | 130 | CPU GPU KTC DTYPE STREAM PROFILE | `_basis_cells_ktc`, `_orth_pair_cross_fe`, `_orthogonal_adaptive_arity_fe`, `_orthogonal_bootstrap_mi_fe` ... |
| `_extra_basis_fourier_cuda` | 127 | CPU KTC STREAM PROFILE | `_extra_basis_resident` |
| `_orthogonal_quadruplet_fe_recipes` | 117 | CPU GPU KTC DTYPE STREAM PROFILE | `_orthogonal_adaptive_arity_fe`, `_orthogonal_quadruplet_fe` |
| `_basis_cells_ktc` | 115 | CPU GPU DTYPE STREAM PROFILE | `_orthogonal_adaptive_degree_device`, `_orthogonal_routing_device` |
| `_fe_batch_dispatch` | 111 | CPU GPU KTC STREAM PROFILE | `_fe_gpu_batch` |
| `_fe_cpu_batch` | 110 | CPU GPU KTC STREAM PROFILE | `_fe_batch_dispatch`, `_fe_gpu_batch` |
| `_pair_usability_table` | 108 | CPU GPU KTC DTYPE STREAM PROFILE |  |
| `_added_information_gate` | 107 | GPU KTC STREAM PROFILE | `_group_distance_fe`, `_grouped_quantile_fe` |
| `_orthogonal_triplet_fe_recipes` | 105 | CPU GPU KTC DTYPE STREAM PROFILE | `_orthogonal_adaptive_arity_fe` |
| `_relaxmrmr_pair_loop` | 100 | GPU KTC STREAM PROFILE |  |
| `_orth_extra_basis_fe_generate_doc` | 98 | CPU GPU KTC DTYPE STREAM PROFILE |  |
| `_fe_content_memo` | 96 | CPU GPU KTC DTYPE STREAM PROFILE | `_orthogonal_cmim_fe` |
| `_fe_cmi_fused_backend` | 90 | CPU GPU KTC STREAM PROFILE | `_fe_batched_mi_cmi` |
| `_fe_batched_mi_mm_cm` | 87 | CPU KTC STREAM PROFILE |  |
| `_orth_abs_floor` | 82 | CPU GPU KTC DTYPE STREAM PROFILE |  |
| `_row_argmax_gpu` | 82 | CPU KTC STREAM PROFILE | `_conditional_gate_fe` |
| `_fe_pure_form_sides` | 78 | CPU GPU KTC STREAM REC:has no cProfile REC:has no GPU profiler (nsys / nc REC:does not state the kernel laun REC:has no occupancy REC:has no whole-GPU throughput /  | `_fe_pure_form_retention` |
| `_fe_mi_contract` | 75 | CPU GPU KTC DTYPE STREAM PROFILE | `_fe_additive_fusion`, `_fe_edge_mi`, `_orth_mi_backends` |
| `_fe_batched_mi_guards` | 70 | CPU GPU KTC DTYPE STREAM PROFILE | `_fe_batched_mi`, `_fe_batched_mi_mm_cm`, `_fe_batched_mi_mm_values` |
| `_orth_basis_memo` | 68 | CPU GPU KTC DTYPE STREAM PROFILE |  |
| `_conditional_gate_fe_helpers` | 67 | GPU KTC DTYPE STREAM PROFILE | `_conditional_gate_fe`, `_conditional_gate_hybrid` |
| `_target_mix_gate` | 66 | CPU GPU KTC STREAM PROFILE |  |
| `_conditional_gate_naming` | 57 | CPU GPU KTC DTYPE STREAM PROFILE | `_conditional_gate_fe`, `_conditional_gate_hybrid` |
| `_usability_pool_resident_pairs` | 57 | CPU GPU KTC STREAM PROFILE |  |
| `_hermite_fe_diverse` | 54 | CPU GPU KTC STREAM PROFILE | `_hermite_fe_optimise_parts` |
| `_fe_gain_stats` | 50 | CPU GPU KTC STREAM PROFILE | `_offset_product_fe`, `_oof_warp_fe`, `_pair_residual_screen`, `_row_stat_fe` ... |
| `_conditional_gate_kernels` | 48 | GPU KTC DTYPE STREAM PROFILE | `_conditional_gate_fe` |
| `_basis_name_codes` | 43 | CPU GPU KTC DTYPE STREAM PROFILE | `_gpu_resident_cross_basis`, `_orth_auto_scorer_fe`, `_orth_pair_cross_fe`, `_orthogonal_bootstrap_mi_fe` ... |
| `_fourier_core_cycles` | 43 | CPU GPU KTC STREAM PROFILE | `_fourier_detect_gpu_resident`, `_orth_extra_basis_fe`, `_orth_extra_basis_fe_helpers` |
| `_usability_row_stat_pool` | 41 | CPU GPU KTC DTYPE STREAM PROFILE |  |
| `_permutation_null_pair_tiles` | 38 | CPU GPU KTC STREAM PROFILE | `_permutation_null_pair_resident` |
| `_pair_operand_arrays` | 37 | CPU GPU KTC DTYPE STREAM PROFILE | `_mrmr_fe_step_helpers` |
| `_cat_pair_joint_njit` | 26 | GPU KTC STREAM PROFILE | `_cat_pair_fe` |
| `_lattice_gate_hit_shared` | 26 | CPU GPU KTC DTYPE STREAM PROFILE | `_conditional_gate_fe` |
| `_fe_edge_mi` | 23 | GPU KTC DTYPE STREAM PROFILE | `_fe_cpu_batch`, `_fe_gpu_batch`, `_fe_mi_contract`, `_orth_mi_backends` |
| `_fe_env_switch` | 18 | CPU GPU KTC DTYPE STREAM PROFILE | `_fourier_detect_batch`, `_fourier_detect_gpu_resident`, `_wavelet_basis_fe_batched` |
| `_fe_scan_common` | 17 | CPU GPU KTC DTYPE STREAM PROFILE | `_offset_product_fe`, `_oof_warp_fe`, `_residual_rescue_fe`, `_row_stat_fe` |
| `_hermite_fe_optimise_helpers` | 17 | CPU GPU KTC DTYPE STREAM PROFILE | `_hermite_fe_optimise_parts` |
| `_fe_output_clip` | 6 | CPU GPU KTC DTYPE STREAM PROFILE | `_offset_product_fe`, `_row_stat_fe` |
