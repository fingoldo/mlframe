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

## Measured priority (2026-10-26, every FE flag on, n=20000, 34 columns, `D:/Temp/fe_all_on_probe.py`; fit 217 s -> 110 s)

Wall per family from the lap timer (`gate:` rows); a module without a gated family has no row of its own and is part of a shared path. A module is only profiled when its wall is above about 2 s or its lap looked out of line; the others are recorded by wall alone. The contract points a module misses (fused GPU kernel, KTC registration, streaming, profile record) are not added for a module until a profile shows it dominating the fit.

| wall (s) | module | disposition |
|---:|---|---|
| 4.78 | `_cat_triple_fe` | bulk tokens + high-cardinality column skip (73s, 73x): 11.4 -> 0.8 s |
| 4.38 | `_binned_numeric_agg_fe` | no hotspot (73u), 1.3 s |
| 2.78 | `_composite_group_agg_fe` | profiled: spread over joint entropy 0.6 s, arrow string iteration 0.26 s and a lookup dictcomp, nothing above 0.6 s; no change |
| 2.75 | `_cat_pair_fe` | profiled standalone: 0.25 s (the 2.75 s lap includes the shared token work fixed in 73s); no change |
| 2.03 | `_orthogonal_triplet_fe` | parallel permutation draws (73t): 5.98 -> 1.97 s |
| 1.80 | `_grouped_agg_fe` | hash-based group lookup (73x): 3.79 -> 1.71 s |
| 1.68 | `_grouped_quantile_fe` | whole-fit profile: 1.5 s cumulative in the family, spread; group lookup shared with 73x; no further change |
| 1.40 | `_mi_greedy_fe` | wall only, not profiled; under the 3 s bar at this scale |
| 1.35 | `_orthogonal_adaptive_degree_fe` | whole-fit profile: 0.9 s cumulative, 0.8 s of it candidate building; no hotspot, no change |
| 1.35 | `_group_distance_fe` | whole-fit profile: 1.3 s cumulative, spread; group lookup shared with 73x; no further change |
| 1.29 | `_conditional_gate_fe` | device-born tau grid by default (73r): 13.6 -> 2.1 s at n=20000, p=34 |
| 1.24 | `_lof_fe` | whole-fit profile: 0.8 s cumulative (`_lof_transform` 0.6 s); no change |
| 1.20 | `_mi_greedy_cmi_fe` | whole-fit profile (n=20000): `_sync_free_qbin_codes` 1.3 s over 1002 calls is the largest function, cumulative 1.4 s; no hotspot, no change |
| 1.09 | `_pairwise_modular_fe` | wall only, not profiled; under the 3 s bar at this scale |
| 0.90 | `_orthogonal_cmim_fe` | whole-fit profile: `score_features_by_cmim` 1.3 s cumulative, no function above 0.2 s of its own time; no change |
| 0.85 | `_row_stat_fe` | profiled: 0.6-0.8 s standalone, the search kernel `eval_candidates` is 0.4 s; no change |
| 0.61 | `_orthogonal_hsic_fe` | compiled median heuristic (73v) |
| 0.47 | `_orthogonal_jmim_fe` | wall only, not profiled; under the 3 s bar at this scale |
| 0.46 | `_ordinal_pattern_fe` | wall only, not profiled; under the 3 s bar at this scale |
| 0.37 | `_oof_warp_fe` | wall only, not profiled; under the 3 s bar at this scale |
| 0.34 | `_conditional_quantile_rank_fe` | wall only, not profiled; under the 3 s bar at this scale |
| 0.00 | `_orthogonal_total_correlation_fe` | tail-dependence exact floor shared with the scorer family (73k) |
| 0.00 | `_orthogonal_scorer_auto_fe` | KSG, tail dependence, HSIC fixed (73k, 73v): 49.3 -> 11.9 s |
| 0.00 | `_orthogonal_ksg_mi_fe` | numba Ross estimator (73k) |

54 further pending modules take under 0.3 s here or are not gated families: `_numeric_decompose_fe`, `_orthogonal_dcor_fe`, `_offset_product_fe`, `_orthogonal_quadruplet_fe`, `_orthogonal_adaptive_arity_fe`, `_mahalanobis_density_fe`, `_integer_lattice_fe`, `_orthogonal_diff_basis_fe`, `_orthogonal_cluster_basis_fe`, `polynom_pair_fe`, `feature_engineering`, `batch_pair_usability_corr_gpu`, `batch_pair_mi_gpu`, `_wavelet_basis_fe_recipes`, `_unified_fe_gate`, `_temporal_agg_fe`, `_target_encoding_fe`, `_synergy_detector`, `_sliced_inverse_regression_fe`, `_residual_rescue_fe`, `_ratio_delta_fe`, `_random_fourier_features_fe`, `_permutation_null_pair_resident`, `_periodic_fe`, `_parity_fe`, `_pair_residual_screen`, `_orthogonal_univariate_fe`, `_orthogonal_three_gate_mi_fe`, `_orthogonal_routing_fe`, `_orthogonal_meta_scorer_fe`, `_orthogonal_lasso_fe`, `_orthogonal_elasticnet_fe`, `_orthogonal_copula_mi_fe`, `_orthogonal_bootstrap_mi_fe`, `_oof_cell2d_fe`, `_mrmr_fe_step_helpers`, `_missingness_fe`, `_hinge_basis_fe`, `_fe_usability_signal`, `_fe_synergy_screen`, `_fe_sufficient_summary`, `_fe_subsample`, `_fe_stability_vote`, `_fe_retention_subsumption`, `_fe_raw_redundancy_helpers`, `_fe_raw_redundancy_drop`, `_fe_pure_form_retention`, `_fe_cmi_redundancy_gate`, `_fe_batched_mi`, `_fe_auto_escalation`, `_fe_additive_fusion`, `_family_maxt_gate`, `_extra_fe_families`, `_count_freq_interaction_fe`.
