# FE GPU contract: classification of the 226 pending modules

Generated 2026-11-12 from `tests/test_meta/_fe_gpu_contract.json` by a static scan of `src/mlframe/feature_selection/filters` (no profiling).

The first survey (FG11-01) counted 188 modules without a GPU kernel by looking for `RawKernel` inside each file. That figure was too high: a module can reach the device through helpers in other files. Recounted:

| group | modules | meaning |
|---|---|---|
| own GPU kernel | 38 | RawKernel / numba.cuda in the module; needs contract points 2 to 5 checked |
| delegates to GPU code | 113 | references cupy / `_gpu` / device paths of other modules; needs the contract checks, not a new kernel |
| no GPU reference, computes | 33 | numba kernels or heavy numpy; candidates for a new GPU path |
| no GPU reference, no compute | 42 | orchestration, naming, memo, config, recipes; a GPU path has no meaning |

Signals are heuristics (numba/prange markers and heavy numpy calls), so the split of the last two groups is a plan to be confirmed module by module, not a verdict.

## Candidates for a new GPU path (most compute signals first)

| module | lines | numba markers | heavy numpy ops |
|---|---|---|---|
| _hermite_basis_eval | 409 | 32 | 0 |
| _orth_extra_basis_fe_helpers | 471 | 11 | 8 |
| _dcd_pair_su_batch | 313 | 7 | 0 |
| _conditional_gate_kernels | 48 | 6 | 0 |
| _residual_rescue_fe | 329 | 5 | 7 |
| _cat_target_encoding_and_weighted | 273 | 5 | 0 |
| _class_encoding | 198 | 4 | 1 |
| _missingness_fe | 580 | 4 | 1 |
| _cat_pair_joint_njit | 26 | 4 | 0 |
| _relaxmrmr_pair_loop | 100 | 4 | 0 |
| _fe_accuracy_gate_newton | 148 | 3 | 5 |
| _temporal_agg_fe | 834 | 3 | 5 |
| _pair_residual_screen | 140 | 3 | 4 |
| _target_encoding_order_stats | 167 | 3 | 4 |
| _conditional_gate_hybrid | 216 | 3 | 1 |
| _orthogonal_meta_scorer_fe | 671 | 3 | 1 |
| _orthogonal_tail_dependence_fe | 182 | 3 | 1 |
| _conditional_gate_fe_helpers | 67 | 3 | 0 |
| _pairs_gate_dedupe | 91 | 3 | 0 |
| _fe_synergy_screen | 299 | 2 | 0 |
| _hermite_fe_optimise_parts | 290 | 2 | 0 |
| _hermite_robust | 394 | 1 | 3 |
| _added_information_gate | 107 | 1 | 1 |
| _numeric_decompose_fe | 579 | 1 | 0 |
| _pairs_operand_floor | 86 | 1 | 0 |
| _temporal_agg_fe_rolling | 310 | 1 | 0 |
| _fe_sufficient_summary | 661 | 0 | 7 |
| _ratio_delta_fe | 780 | 0 | 4 |
| _fe_interaction_prerank | 345 | 0 | 3 |
| _categorical_encoding | 144 | 0 | 2 |
| _count_freq_interaction_fe | 582 | 0 | 2 |
| _oof_cell2d_fe | 148 | 0 | 2 |
| _target_mix_gate | 66 | 0 | 2 |

## No-compute modules (to be recorded as exempt with a per-module reason after a read)

- _basis_name_codes (19 lines)
- _cat_pair_fe (603 lines)
- _cat_triple_fe (617 lines)
- _conditional_gate_naming (57 lines)
- _encoding_recipes (372 lines)
- _fe_content_memo (96 lines)
- _fe_env_switch (18 lines)
- _fe_gain_stats (50 lines)
- _fe_output_clip (6 lines)
- _fe_pure_form_sides (78 lines)
- _fe_scan_common (17 lines)
- _feature_engineering_mem_budget (441 lines)
- _fourier_core_cycles (43 lines)
- _grouped_agg_fe (546 lines)
- _hermite_fe_diverse (54 lines)
- _hermite_fe_optimise_helpers (17 lines)
- _hermite_oracle (148 lines)
- _lattice_gate_hit_shared (26 lines)
- _missingness_ratio_recipes (151 lines)
- _oof_warp_fe (186 lines)
- _orth_abs_floor (82 lines)
- _orth_basis_memo (68 lines)
- _orth_extra_basis_fe_generate_doc (98 lines)
- _orth_scoring_memo (158 lines)
- _orthogonal_bootstrap_mi_fe (510 lines)
- _orthogonal_elasticnet_fe (367 lines)
- _orthogonal_ksg_mi_fe (421 lines)
- _orthogonal_lasso_fe (454 lines)
- _orthogonal_quadruplet_fe_recipes (117 lines)
- _orthogonal_scorer_auto_fe (713 lines)
- _orthogonal_triplet_fe_recipes (105 lines)
- _orthogonal_triplet_recipes (219 lines)
- _pair_operand_arrays (37 lines)
- _pair_usability_table (108 lines)
- _pairs_code_bins (29 lines)
- _pairs_gates (356 lines)
- _pairs_score_recompute (386 lines)
- _row_stat_fe (245 lines)
- _screen_predictors_gate (213 lines)
- _synergy_detector (213 lines)
- _usability_row_stat_pool (41 lines)
- _wavelet_basis_fe_recipes (392 lines)
