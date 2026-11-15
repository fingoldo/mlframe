# Pending FE GPU-contract modules after 2026-11-14 (132)
Regenerated from `contract_violations` and a text scan of each module. `kind` says what the module itself contains: an own CUDA kernel, an own numba kernel, or neither (cupy/numpy orchestration over kernels that live elsewhere). The route column is a recommendation, not a decision: **kernel** = write the missing kernel/twin, **carve** = declare under a parent that already meets the contract (the parent's text must name it), **owner** = needs a decision whether glue modules may be classified as exempt.

| kind | modules |
|---|---|
| cupy/numpy orchestration | 94 |
| own numba | 31 |
| own CUDA kernel | 7 |

## cupy/numpy orchestration (94)

| module | lines | missing points |
|---|---|---|
| `_cat_pair_fe` | 603 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; profile record has no cProfile; profile record has no GPU profiler (nsys / ncu / nvprof); profile record does not state the kernel launches per call (; profile record has no occupancy; profile record has no whole-GPU throughput / utilization fig |
| `_cat_triple_fe` | 617 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; profile record has no cProfile; profile record has no GPU profiler (nsys / ncu / nvprof); profile record does not state the kernel launches per call (; profile record has no occupancy; profile record has no whole-GPU throughput / utilization fig |
| `_count_freq_interaction_fe` | 582 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; profile record has no cProfile; profile record has no GPU profiler (nsys / ncu / nvprof); profile record does not state the kernel launches per call (; profile record has no occupancy; profile record has no whole-GPU throughput / utilization fig |
| `_fe_pure_form_sides` | 78 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; profile record has no cProfile; profile record has no GPU profiler (nsys / ncu / nvprof); profile record does not state the kernel launches per call (; profile record has no occupancy; profile record has no whole-GPU throughput / utilization fig |
| `_unified_fe_gate` | 594 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; profile record has no cProfile; profile record has no GPU profiler (nsys / ncu / nvprof); profile record does not state the kernel launches per call (; profile record has no occupancy; profile record has no whole-GPU throughput / utilization fig |
| `_conditional_gate_fe` | 876 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; profile record has no cProfile; profile record has no GPU profiler (nsys / ncu / nvprof); profile record does not state the kernel launches per call (; profile record has no occupancy; profile record has no whole-GPU throughput / utilization fig |
| `_grouped_quantile_fe` | 927 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; profile record has no cProfile; profile record has no GPU profiler (nsys / ncu / nvprof); profile record does not state the kernel launches per call (; profile record has no occupancy; profile record has no whole-GPU throughput / utilization fig |
| `_orth_auto_scorer_fe` | 775 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; profile record has no cProfile; profile record has no GPU profiler (nsys / ncu / nvprof); profile record does not state the kernel launches per call (; profile record has no occupancy |
| `_basis_name_codes` | 43 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_conditional_gate_naming` | 57 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_encoding_recipes` | 372 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_fe_content_memo` | 96 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_fe_env_switch` | 18 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_fe_mi_contract` | 75 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_fe_output_clip` | 6 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_fe_scan_common` | 17 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_feature_engineering_mem_budget` | 444 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_hermite_fe_optimise_helpers` | 17 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_lattice_gate_hit_shared` | 26 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_orth_abs_floor` | 82 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_orth_basis_memo` | 68 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_orth_extra_basis_fe_generate_doc` | 98 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_orthogonal_quadruplet_fe_recipes` | 117 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_orthogonal_routing_device` | 142 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; profile record does not state the kernel launches per call (; profile record has no whole-GPU throughput / utilization fig |
| `_orthogonal_shared` | 130 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_orthogonal_triplet_fe_recipes` | 105 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_pair_operand_arrays` | 37 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_pair_usability_table` | 108 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_screen_predictors_gate` | 213 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_usability_row_stat_pool` | 41 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_basis_cells_ktc` | 115 | no numba CPU kernel; no fused GPU kernel; no crit dtype use; no column-block streaming; no profile record |
| `_categorical_encoding` | 144 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_conditional_gate_fe_helpers` | 67 | no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_cupy_polynom_optimizer` | 279 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_extra_basis_resident` | 221 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_extra_fe_families_dispersion_resident` | 369 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_auto_escalation` | 759 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_auto_escalation_proposers` | 362 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_batch_dispatch` | 111 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_cmi_fused_backend` | 90 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_cmi_redundancy_gate` | 937 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_cmi_redundancy_null` | 633 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_cpu_batch` | 110 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_gain_stats` | 50 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_matrix_io` | 297 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_pure_form_retention` | 889 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_raw_redundancy_anchors` | 465 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_raw_redundancy_drop` | 947 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_raw_redundancy_helpers` | 480 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_retention_subsumption` | 306 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_stability_vote` | 431 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_synergy_exhaustive` | 438 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fourier_core_cycles` | 43 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_grouped_agg_fe` | 546 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_hermite_fe_diverse` | 54 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_hermite_fe_optimise_parts` | 290 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_hermite_oracle` | 159 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_hermite_prewarp_gpu_resident` | 292 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_mi_greedy_device` | 248 | no numba CPU kernel; no fused GPU kernel; profile record does not state the kernel launches per call (; profile record has no occupancy; profile record has no whole-GPU throughput / utilization fig |
| `_mi_greedy_fe` | 739 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_missingness_ratio_recipes` | 151 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_mrmr_fe_step_helpers` | 802 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_offset_product_fe` | 371 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_oof_cell2d_fe` | 148 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_oof_warp_fe` | 186 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orth_gpu_resident` | 360 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orth_pair_cross_fe` | 688 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orth_scoring_memo` | 158 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_adaptive_arity_fe` | 839 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_adaptive_degree_device` | 136 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_bootstrap_mi_fe` | 510 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_cmim_fe` | 709 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_elasticnet_fe` | 367 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_ksg_mi_fe` | 421 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_lasso_fe` | 454 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_quadruplet_fe` | 689 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_routing_fe` | 680 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_scorer_auto_fe` | 713 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_three_gate_mi_fe` | 870 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_triplet_recipes` | 219 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_permutation_null_pair_resident` | 258 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_permutation_null_pair_tiles` | 38 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_ratio_delta_fe` | 780 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_row_stat_fe` | 245 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_synergy_detector` | 213 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_target_mix_gate` | 66 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_usability_pool_resident_pairs` | 57 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_wavelet_basis_fe_recipes` | 392 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `polynom_pair_fe` | 786 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_conditional_gate_hybrid` | 216 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_additive_fusion` | 525 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no profile record |
| `_fe_additive_fusion_gpu_resident` | 503 | no numba CPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_pure_form_retention_gpu_resident` | 289 | no numba CPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orth_mi_backends` | 724 | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no profile record |

## own numba (31)

| module | lines | missing points |
|---|---|---|
| `_periodic_fe` | 776 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; profile record has no cProfile; profile record has no GPU profiler (nsys / ncu / nvprof); profile record does not state the kernel launches per call (; profile record has no occupancy; profile record has no whole-GPU throughput / utilization fig |
| `_orth_extra_basis_fe` | 766 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; profile record has no GPU profiler (nsys / ncu / nvprof); profile record does not state the kernel launches per call (; profile record has no occupancy; profile record has no whole-GPU throughput / utilization fig |
| `_conditional_gate_kernels` | 48 | no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_fe_edge_mi` | 23 | no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_missingness_fe` | 580 | no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_target_encoding_order_stats` | 167 | no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_class_encoding` | 198 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_family_maxt_gate` | 903 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_subsample` | 453 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_usability_signal` | 457 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_hermite_basis_eval` | 409 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_hermite_fe_optimise` | 843 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_hermite_fe_optimise_pair` | 928 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_hermite_prewarp` | 549 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_hermite_robust` | 394 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_numba_polynom_optimizer` | 861 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orth_basis_recipes` | 801 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orth_dedup` | 545 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orth_extra_basis_fe_helpers` | 471 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_dcor_fe` | 533 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_hsic_fe` | 817 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_meta_scorer_fe` | 671 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_triplet_fe` | 838 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_pair_residual_screen` | 140 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_parity_fe` | 213 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_relaxmrmr_pair_loop` | 100 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_target_encoding_fe` | 785 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_temporal_agg_fe` | 834 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_temporal_agg_fe_rolling` | 310 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `feature_engineering` | 748 | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orth_mi_null_max` | 281 | no fused GPU kernel; not registered with the kernel-tuning cache; no profile record |

## own CUDA kernel (7)

| module | lines | missing points |
|---|---|---|
| `_fe_cmi_perm_null_gpu` | 527 | no numba CPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_gpu_pairs` | 305 | no numba CPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_gpu_resident_basis` | 969 | no numba CPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_gpu_resident_cross_basis` | 415 | no numba CPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_row_argmax_gpu` | 82 | no numba CPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_wavelet_basis_fe_batched` | 376 | no numba CPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_gpu_batch` | 568 | no numba CPU kernel; not registered with the kernel-tuning cache; no profile record |
