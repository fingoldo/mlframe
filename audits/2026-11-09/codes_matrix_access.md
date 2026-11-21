# Readers of the codes matrix

129 (function, parameter) pairs subscript a matrix parameter.

| reader kind | leans feature-major | leans row-major | neutral |
|---|---|---|---|
| cuda.jit | 5 | 0 | 0 |
| njit | 18 | 8 | 14 |
| python | 58 | 0 | 26 |

## Functions that lean feature-major

| file | function | param | kind | column | rows_inner | row | cols_inner |
|---|---|---|---|---|---|---|---|
| _batch_pair_mi_cuda_kernels.py | _kernel | factors_data | cuda.jit | 0 | 2 | 0 | 0 |
| _batch_pair_mi_cuda_kernels.py | _hist_kernel | factors_data | cuda.jit | 0 | 2 | 0 | 0 |
| _batch_pair_mi_cuda_kernels.py | _hist_kernel_shared | factors_data | cuda.jit | 0 | 2 | 0 | 0 |
| friend_graph_gpu.py | _kernel | sub | cuda.jit | 0 | 2 | 0 | 0 |
| friend_graph_gpu.py | _kernel | codes | cuda.jit | 0 | 1 | 0 | 0 |
| info_theory/_batch_kernels.py | _triple_mi_scalar | factors_data | njit | 0 | 3 | 0 | 0 |
| info_theory/_batch_kernels.py | _triple_mi_perm_scalar | factors_data | njit | 0 | 3 | 0 | 0 |
| _cat_kway_materialize.py | _scatter_factorize_lookup | factors_data | njit | 0 | 2 | 0 | 0 |
| _cat_target_encoding_and_weighted.py | _pair_search_kernel_weighted_njit | factors_data | njit | 0 | 2 | 0 | 0 |
| _mrmr_fe_step/_conditional_maxt.py | _row_range | codes | njit | 0 | 2 | 0 | 1 |
| _permutation_null.py | _pairwise_occupied_joint_k_njit | factors_data | njit | 0 | 2 | 0 | 0 |
| cat_interactions.py | _pair_search_kernel_njit | factors_data | njit | 0 | 2 | 0 | 0 |
| correlated_features.py | _su_pairs_njit | codes | njit | 0 | 2 | 0 | 0 |
| info_theory/_batch_kernels.py | _pair_mi_scalar | factors_data | njit | 0 | 2 | 0 | 0 |
| info_theory/_batch_kernels.py | _pair_mi_perm_scalar | factors_data | njit | 0 | 2 | 0 | 0 |
| info_theory/_class_encoding.py | merge_vars | factors_data | njit | 1 | 1 | 0 | 0 |
| info_theory/_class_encoding.py | joint_freqs_2var | factors_data | njit | 0 | 2 | 0 | 0 |
| info_theory/_class_encoding.py | joint_entropy_2var | factors_data | njit | 0 | 2 | 0 | 0 |
| info_theory/_entropy_kernels.py | _merge_vars_freqs | factors_data | njit | 0 | 2 | 0 | 0 |
| _mrmr_artifact_entropy.py | _column_histograms_njit | data | njit | 0 | 1 | 0 | 0 |
| _mrmr_fe_step/_conditional_maxt.py | _gather_transposed | data | njit | 0 | 1 | 0 | 0 |
| _numba_utils_cross_module.py | _relevance_mi_1var_fused | factors_data | njit | 0 | 1 | 0 | 0 |
| info_theory/_entropy_kernels.py | _entropy_x_onto_classes | factors_data | njit | 0 | 1 | 0 | 0 |
| _screen_predictors_gate.py | compute_selection_gate | factors_data | python | 4 | 0 | 0 | 0 |
| _mrmr_fit_impl/_fit_impl_stages/_binning.py | _rebin_continuous_targets | data | python | 3 | 0 | 0 | 0 |
| _mrmr_fit_impl/_operand_information.py | operand_adds_information | data | python | 3 | 0 | 0 | 0 |
| _cat_interactions_step.py | _occupied_pair_cells | data | python | 2 | 0 | 0 | 0 |
| _fe_raw_redundancy_drop.py | _drop_redundant_raw_o_step2_adversarial_multi_contamination | data | python | 2 | 0 | 0 | 0 |
| _feature_engineering_pairs/_pairs_gate_dedupe.py | distinct_columns | codes | python | 2 | 0 | 0 | 0 |
| _gain_stats_codes_gpu.py | _main_effect_z | codes | python | 2 | 0 | 0 | 0 |
| _honest_mdlp_null.py | _njit_null_columns_block | factors_data | python | 2 | 0 | 0 | 0 |
| _mrmr_artifacts.py | _fill_marginals_and_su | data | python | 2 | 0 | 0 | 0 |
| _mrmr_fe_step/_step_inert_operands.py | _has_inert_leaf | data | python | 2 | 0 | 0 | 0 |
| _mrmr_fe_step/_step_pairs_prevalence.py | _get_col_codes_i64 | data | python | 2 | 0 | 0 | 0 |
| _mrmr_fe_step/_step_pairs_rank.py | _pair_perm_null_admission | data | python | 2 | 0 | 0 | 0 |
| _mrmr_fit_impl/_assign_support_tail.py | _assign_support_tai_signal_aware_subsumption | data | python | 2 | 0 | 0 | 0 |
| _mrmr_fit_impl/_fallback_maxt.py | fallback_maxt_floor | data | python | 2 | 0 | 0 | 0 |
| _mrmr_fit_impl/_finalise_group_demotion.py | _demote_group_leaks_final | data | python | 2 | 0 | 0 | 0 |
| _mrmr_fit_impl/_friend_graph_and_redundancy/_group4.py | _pick_linear_usable_floor_raw | data | python | 2 | 0 | 0 | 0 |
| _pair_mi_perm_cuda.py | pair_mi_perm_matrix_cuda | factors_data | python | 2 | 0 | 0 | 0 |
| correlated_features.py | _su_pairs_python | codes | python | 2 | 0 | 0 | 0 |
| friend_graph_gpu.py | _cupy_relevance | factors_data | python | 2 | 0 | 0 | 0 |
| friend_graph_gpu.py | friend_graph_stats_cuda | factors_data | python | 2 | 0 | 0 | 0 |
| info_theory/_cmi_cuda.py | conditional_mi_batched_dispatch | factors_data | python | 2 | 0 | 0 | 0 |
| _batch_pair_mi_cuda_fm.py | _launch_pairs | factors_data | python | 1 | 0 | 0 | 0 |
| _batch_pair_mi_cuda_shared_fused.py | _launch_pairs | factors_data | python | 1 | 0 | 0 | 0 |
| _cat_interactions_step.py | _run_cat_interactio_cache_active | data | python | 1 | 0 | 0 | 0 |
| _cat_interactions_step_stages.py | _stage_candidates | data | python | 1 | 0 | 0 | 0 |
| _cat_kway_materialize.py | _build_kway_chained_lookup | factors_data | python | 1 | 0 | 0 | 0 |
| _cat_weighted_pair_gpu.py | weighted_pair_search_gpu | factors_data | python | 1 | 0 | 0 | 0 |
| _cluster_aggregate.py | run_cluster_aggregate_step | data | python | 1 | 0 | 0 | 0 |
| _fe_raw_redundancy_anchors.py | _build_raw_redundancy_step2_def_raw_codes | data | python | 1 | 0 | 0 | 0 |
| _fe_raw_redundancy_anchors.py | _build_raw_redundan_cont_none_np_asarray | data | python | 1 | 0 | 0 | 0 |
| _fe_raw_redundancy_drop.py | _no_harm_guard_result | data | python | 1 | 0 | 0 | 0 |
| _fe_raw_redundancy_drop.py | _drop_redundant_raw_o_step3_profiles_already_drop | data | python | 1 | 0 | 0 | 0 |
| _fe_raw_redundancy_drop.py | _drop_redundant_raw_o_step2_st_gmi_payload | data | python | 1 | 0 | 0 | 0 |
| _fe_sufficient_summary.py | _validated_target | data | python | 1 | 0 | 0 | 0 |
| _fe_sufficient_summary.py | _raw_code_matrix | data | python | 1 | 0 | 0 | 0 |
| _fe_sufficient_summary.py | _selected_columns_summary | data | python | 1 | 0 | 0 | 0 |
| _honest_mdlp_null.py | _njit_null_columns | factors_data | python | 1 | 0 | 0 | 0 |
| _honest_mdlp_null.py | honest_fdr_floor | factors_data | python | 1 | 0 | 0 | 0 |
| _honest_oj_fused.py | _assemble_pool | factors_data | python | 1 | 0 | 0 | 0 |
| _honest_oj_fused.py | _floor_body | factors_data | python | 1 | 0 | 0 | 0 |
| _interaction_information.py | pooled_pair_ii_null_floor | factors_data | python | 1 | 0 | 0 | 0 |
| _mrmr_artifacts.py | compute_mrmr_artifacts | data | python | 1 | 0 | 0 | 0 |
| _mrmr_fe_step/_step_inert_operands.py | _is_dominated_by_the_other_leaves | data | python | 1 | 0 | 0 | 0 |
| _mrmr_fit_impl/_fit_impl_gains.py | _resolve_screen_shared_idx | data | python | 1 | 0 | 0 | 0 |
| _mrmr_fit_impl/_fit_impl_stages/_binning.py | _install_honest_null_hook | data | python | 1 | 0 | 0 | 0 |
| _mrmr_fit_impl/_fit_impl_stages/_prelude.py | _resolve_min_relevance_gain | data | python | 1 | 0 | 0 | 0 |
| _mrmr_fit_impl/_friend_graph_and_redundancy/_group3.py | _encoded_feature_gate_pass | data | python | 1 | 0 | 0 | 0 |
| _mrmr_fit_impl/_friend_graph_and_redundancy/_group3.py | _score_encoded_features_heldout | data | python | 1 | 0 | 0 | 0 |
| _mrmr_fit_impl/_friend_graph_and_redundancy/_group3.py | _genuine_child_bins | data | python | 1 | 0 | 0 | 0 |
| _mrmr_fit_impl/_friend_graph_and_redundancy/_group3.py | _drop_redundant_raw_operands_sweep | data | python | 1 | 0 | 0 | 0 |
| _mrmr_fit_impl/_friend_graph_and_redundancy/_raw_dtype.py | raw_column_is_numeric | data | python | 1 | 0 | 0 | 0 |
| _mrmr_fit_impl/_main_effect_restore.py | run_main_effect_restore | data | python | 1 | 0 | 0 | 0 |
| _mrmr_fit_impl/_selection_compaction.py | run_selection_compaction | data | python | 1 | 0 | 0 | 0 |
| _permutation_null.py | _order1_candidate_stats | factors_data | python | 1 | 0 | 0 | 0 |
| _permutation_null.py | pooled_permutation_null_gain_floor | factors_data | python | 1 | 0 | 0 | 0 |
| _permutation_null.py | _validate_code_ranges | data | python | 1 | 0 | 0 | 0 |
| cat_interactions.py | _restore_cached_marginal_mis | factors_data | python | 1 | 0 | 0 | 0 |
| friend_graph_gpu.py | friend_graph_stats_cupy | factors_data | python | 1 | 0 | 0 | 0 |

## Functions that lean row-major

| file | function | param | kind | row | cols_inner | column | rows_inner |
|---|---|---|---|---|---|---|---|
| _batch_pair_mi_cuda_shared_fused.py | _col_min_max_pass | factors_data | njit | 0 | 3 | 0 | 0 |
| _numba_utils_cross_module.py | quantile_bin_block | codes | njit | 0 | 2 | 0 | 0 |
| _orthogonal_triplet_fe.py | _triple_mi_kernel | codes | njit | 0 | 2 | 0 | 0 |
| _feature_engineering_pairs/_pairs_gate_dedupe.py | _column_fingerprints | codes | njit | 0 | 1 | 0 | 0 |
| _mrmr_fe_step/_conditional_maxt.py | _conditional_mi_columns | codes | njit | 0 | 1 | 0 | 0 |
| _mrmr_fe_step/_conditional_maxt.py | _shift_rows_int16 | codes | njit | 0 | 1 | 0 | 0 |
| _numba_utils_cross_module.py | quantile_codes_block | codes | njit | 0 | 1 | 0 | 0 |
| _orthogonal_univariate_fe/_orth_mi_null_max.py | _mi_from_codes_rows_njit | codes | njit | 0 | 1 | 0 | 0 |

## Files with CUDA source strings (read by hand)

- _abs_corr_tiled_gpu.py
- _added_information_gpu.py
- _batch_mi_noise_gate_bits.py
- _batch_pair_mi_cuda_fm.py
- _batch_pair_mi_cuda_shared_fused.py
- _bin_fingerprint_gpu.py
- _binned_numeric_agg_resident.py
- _bootstrap_rank_mi_gpu.py
- _bootstrap_rank_mi_tied_gpu.py
- _cat_pair_joint_gpu.py
- _cat_weighted_pair_gpu.py
- _cell_moments_kernel.py
- _cell_validity_gpu.py
- _cmi_null_strata_gpu.py
- _cmim_min_gpu.py
- _conditional_bin_moments.py
- _cupy_polynom_optimizer.py
- _dcd_pair_su_gpu.py
- _dcor_gpu.py
- _degenerate_combo_gpu.py
- _extra_fe_families_dispersion_resident.py
- _family_maxt_counts_device.py
- _family_maxt_cuda_bits.py
- _fe_accuracy_gate_kernel.py
- _fe_batched_mi.py
- _fe_batched_mi_cmi.py
- _fe_batched_mi_cmi_hist.py
- _fe_batched_mi_cmi_lut.py
- _fe_batched_mi_mm_values.py
- _fe_batched_qbin.py
- _fe_pure_form_retention_gpu_resident.py
- _fe_sufficient_summary_pairs.py
- _feature_engineering_pairs/_pairs_abs_corr_gpu.py
- _feature_engineering_pairs/_pairs_analytic_gpu.py
- _fleuret_confirm_gpu.py
- _gain_stats_codes_gpu.py
- _gpu_kernel_sources.py
- _gpu_pairs.py
- _gpu_resident_basis.py
- _gpu_resident_basis_factories.py
- _gpu_resident_discretize.py
- _gpu_resident_extval_fused.py
- _gpu_resident_fe.py
- _gpu_resident_materialise.py
- _gpu_resident_operand_build.py
- _gpu_resident_rank_bin.py
- _gpu_resident_select_f32_compact.py
- _gpu_resident_select_kernels.py
- _gpu_resident_shared.py
- _group_distance_gpu.py
- _grouped_quantile_gpu.py
- _hermite_fe_mi.py
- _hinge_precheck_gpu.py
- _honest_null_accept_bits.py
- _honest_null_device.py
- _honest_null_k3.py
- _honest_null_k4.py
- _honest_oj_fused_gpu.py
- _hsic_gpu.py
- _ksg_cd_gpu.py
- _lof_gpu.py
- _meta_fingerprint_gpu.py
- _mi_greedy_cmi_fe_gpu.py
- _modular_codes_gpu.py
- _mrmr_fe_step/_conditional_maxt_gpu.py
- _mrmr_fit_impl/_eng_dedup_spearman_gpu.py
- _null_max_perm_gpu.py
- _numeric_decompose_gpu.py
- _offset_product_gpu.py
- _order_stats_gpu.py
- _orthogonal_copula_gpu.py
- _orthogonal_univariate_fe/_extra_basis_fourier_cuda.py
- _orthogonal_univariate_fe/_extra_basis_resident.py
- _orthogonal_univariate_fe/_fourier_batch_gpu.py
- _orthogonal_univariate_fe/_fourier_fused_gpu.py
- _orthogonal_xi_gpu.py
- _pair_mi_perm_cuda.py
- _pair_residual_gpu.py
- _pairwise_modular_rank_gpu.py
- _parity_gpu.py
- _perm_labels_device.py
- _permutation_null_joint_kernel.py
- _quantile_codes_gpu.py
- _ratio_delta_screen_gpu.py
- _relaxmrmr_pair_gpu.py
- _resident_candidate_mi.py
- _residual_rescue_gpu.py
- _rolling_window_gpu.py
- _row_argmax_gpu.py
- _strata_shuffle_gpu.py
- _synergy_combo_gpu.py
- _tail_dependence_gpu.py
- _triple_mi_gpu.py
- _usability_greedy_clf_batched.py
- _wavelet_legs_fused_gpu.py
- discretization/_discretization_cuda.py
- hermite_fe/_als_fused_gpu.py
- hermite_fe/_als_kernels_gpu.py
- hermite_fe/_basis_fused_gpu.py
- hermite_fe/_nonpoly_cells_gpu.py
- info_theory/_cmi_cuda.py
