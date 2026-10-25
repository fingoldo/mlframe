# FE GPU contract walk: the 90 pending modules (2026-10-25)

Static check by `contract_violations` (`mlframe.feature_selection._benchmarks.fe_gpu_coverage_inventory`) over the modules in `tests/test_meta/_fe_gpu_contract.json` under `pending`. Compliant: 1, exempt: 28, pending: 90. The check reads each module's own text and its `profiles/<module>.md`, so a GPU kernel that lives in a sibling file does not count for the module (example: `_pairwise_modular_fe` has its ranking kernel in `_pairwise_modular_rank_gpu.py`, profiled in `profiles/_pairwise_modular_rank_gpu.md`).

## Missing points across the pending modules

| missing point | modules |
|---|---:|
| no profile record | 89 |
| not registered with the kernel-tuning cache | 87 |
| no fused GPU kernel | 84 |
| no column-block streaming | 82 |
| no numba CPU kernel | 63 |
| no crit dtype use | 9 |
| GPU kernel source has a bare double / float (use {REAL}, or {ACC} for an accumulator) | 4 |
| GPU kernel source has no {REAL} placeholder | 2 |
| module not found | 1 |

Violations per module: 1 missing: 1 modules, 3 missing: 6 modules, 4 missing: 20 modules, 5 missing: 56 modules, 6 missing: 7 modules.

## Closest to compliant (three or fewer missing points)

| module | missing |
|---|---|
| `_hybrid_orth_family_variants` | module not found |
| `_binned_numeric_agg_fe` | no fused GPU kernel; not registered with the kernel-tuning cache; no profile record |
| `_extra_fe_families` | no fused GPU kernel; not registered with the kernel-tuning cache; no profile record |
| `_family_maxt_gate` | no fused GPU kernel; not registered with the kernel-tuning cache; no profile record |
| `_feature_engineering_pairs` | GPU kernel source has no {REAL} placeholder; GPU kernel source has a bare double / float (use {REAL}, or {ACC} for an accumulator); no profile record |
| `_orthogonal_univariate_fe` | not registered with the kernel-tuning cache; GPU kernel source has a bare double / float (use {REAL}, or {ACC} for an accumulator); no profile record |
| `batch_pair_usability_corr_gpu` | not registered with the kernel-tuning cache; no column-block streaming; no profile record |

## All pending modules

| module | missing points |
|---|---|
| `_binned_numeric_agg_fe` | no fused GPU kernel; not registered with the kernel-tuning cache; no profile record |
| `_cat_pair_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_cat_triple_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_composite_group_agg_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_conditional_gate_fe` | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_conditional_quantile_rank_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_confirm_predictor_engineered` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_count_freq_interaction_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_extra_fe_families` | no fused GPU kernel; not registered with the kernel-tuning cache; no profile record |
| `_family_maxt_gate` | no fused GPU kernel; not registered with the kernel-tuning cache; no profile record |
| `_fe_additive_fusion` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no profile record |
| `_fe_auto_escalation` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_batched_mi` | no numba CPU kernel; not registered with the kernel-tuning cache; GPU kernel source has a bare double / float (use {REAL}, or {ACC} for an accumulator); no profile record |
| `_fe_cmi_redundancy_gate` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_family_timing` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_fe_gpu_strict` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_fe_gpu_vram` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_fe_pure_form_retention` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_raw_redundancy_drop` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_raw_redundancy_helpers` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_resident_operands` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_fe_retention_subsumption` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_stability_vote` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_subsample` | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_sufficient_summary` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_synergy_screen` | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_fe_usability_signal` | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_feature_engineering_pairs` | GPU kernel source has no {REAL} placeholder; GPU kernel source has a bare double / float (use {REAL}, or {ACC} for an accumulator); no profile record |
| `_fourier_detect_cap` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_gpu_strict_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_group_distance_fe` | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_grouped_agg_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_grouped_quantile_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_hinge_basis_fe` | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_hybrid_orth_family_variants` | module not found |
| `_integer_lattice_fe` | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_lof_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_mahalanobis_density_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_mi_greedy_cmi_fe` | not registered with the kernel-tuning cache; no column-block streaming; GPU kernel source has no {REAL} placeholder; GPU kernel source has a bare double / float (use {REAL}, or {ACC} for an accumulator); no profile record |
| `_mi_greedy_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_missingness_fe` | no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_mrmr_fe_step` | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_mrmr_fe_step_helpers` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_numeric_decompose_fe` | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_offset_product_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_oof_cell2d_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_oof_warp_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_ordinal_pattern_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_adaptive_arity_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_adaptive_degree_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_bootstrap_mi_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_cluster_basis_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_cmim_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_copula_mi_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_dcor_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_diff_basis_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_elasticnet_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_hsic_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_jmim_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_ksg_mi_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_lasso_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_meta_scorer_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_orthogonal_quadruplet_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_routing_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_scorer_auto_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_three_gate_mi_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_total_correlation_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_triplet_fe` | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_orthogonal_univariate_fe` | not registered with the kernel-tuning cache; GPU kernel source has a bare double / float (use {REAL}, or {ACC} for an accumulator); no profile record |
| `_pair_residual_screen` | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_pairwise_modular_fe` | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_parity_fe` | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_periodic_fe` | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_permutation_null_pair_resident` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_random_fourier_features_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_ratio_delta_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_residual_rescue_fe` | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_row_stat_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_semi_supervised_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_sliced_inverse_regression_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_synergy_detector` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_target_encoding_fe` | no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_target_mix_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no crit dtype use; no column-block streaming; no profile record |
| `_temporal_agg_fe` | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_unified_fe_gate` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `_wavelet_basis_fe_recipes` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `batch_pair_mi_gpu` | no numba CPU kernel; no fused GPU kernel; no column-block streaming; no profile record |
| `batch_pair_usability_corr_gpu` | not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `feature_engineering` | no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |
| `polynom_pair_fe` | no numba CPU kernel; no fused GPU kernel; not registered with the kernel-tuning cache; no column-block streaming; no profile record |

## Reading

Almost every pending module lacks a profile record (89 of 90) and a kernel-tuning-cache registration (87). 63 have no numba CPU kernel at all, so they are not candidates for a GPU path until the CPU reference exists. The seven modules with three or fewer missing points are the cheapest to promote; `_hybrid_orth_family_variants` is listed as not found (the registry name does not resolve to a module). Promotion means meeting each point for real, not matching the static patterns.
