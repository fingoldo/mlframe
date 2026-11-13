# Tiny-cupy-loops, low severity: triage list (2026-11-12)

The detector's `low` hits in the current source, one row per function; the same dispositions as the high table (FALSE POSITIVE, FALLBACK, REAL, LEGACY).

Rows: 50.

| file | function | line(s) | disposition |
|---|---|---|---|
| data_valuation/_propagate_gpu_ktc.py | nearest_neighbor_value_resident | 73 | FALSE POSITIVE: a blocked loop (`for start in range(0, n_full, batch_full)`), each trip one batched operation over its block |
| feature_selection/filters/_batch_mi_noise_gate_bits.py | bits_gate_mi | 194 | FALSE POSITIVE: a blocked loop (`for lo in range(0, k_total, group)`), each trip one batched operation over its block |
| feature_selection/filters/_batch_pair_mi_cuda_kernels.py | batch_pair_mi_cuda_row_chunked | 709 | FALSE POSITIVE: a blocked loop (`for pair_start in range(0, n_pairs, pair_subchunk_rows)`), each trip one batched operation over its block |
| feature_selection/filters/_binned_agg_cheap_mi.py | _cheap_mi_group_selection_device | 130 | FALSE POSITIVE: a blocked loop (`for start in range(0, len(usable), width)`), each trip one batched operation over its block |
| feature_selection/filters/_binned_numeric_agg_resident.py | _binagg_device_scores | 678 | FALSE POSITIVE: a blocked loop (`for s0 in range(0, len(col_specs), block)`), each trip one batched operation over its block |
| feature_selection/filters/_bootstrap_rank_mi_gpu.py | distinct_walk_gpu | 147 | FALSE POSITIVE: a blocked loop (`for c0 in range(0, k, block_cols)`), each trip one batched operation over its block |
| feature_selection/filters/_bootstrap_rank_mi_tied_gpu.py | tied_replicate_mi_stream | 304 | FALSE POSITIVE: a blocked loop (`for c0 in range(0, k, block)`), each trip one batched operation over its block |
| feature_selection/filters/_family_maxt_gate.py | _family_codes | 564 | FALSE POSITIVE: a blocked loop (`for j0 in range(0, n_cols, block)`), each trip one batched operation over its block |
| feature_selection/filters/_family_maxt_gate.py | family_codes_of_rows | 595 | FALSE POSITIVE: a blocked loop (`for j0 in range(0, n_cols, block)`), each trip one batched operation over its block |
| feature_selection/filters/_family_maxt_gate.py | family_null_max | 462 | FALSE POSITIVE: a blocked loop (`for start in range(0, int(n_perm), block)`), each trip one batched operation over its block |
| feature_selection/filters/_family_maxt_gate_device.py | null_max | 156 | FALSE POSITIVE: a blocked loop (`for start in range(0, int(m), block)`), each trip one batched operation over its block |
| feature_selection/filters/_fe_batched_qbin.py | batched_quantile_bin_gpu | 205 | FALSE POSITIVE: a blocked loop (`for c0 in range(0, K, block_cols)`), each trip one batched operation over its block (a full radix select over n x block elements plus one fused coder launch: 10.9 ms and 9.5 ms at 100000 x 400, against about 25 edge-table kernels of 2 us); the loop only runs when the working set of the whole matrix does not fit the VRAM cushion |
| feature_selection/filters/_feature_engineering_pairs/_pairs_analytic_gpu.py | resident_observed_mi_and_bins | 157 | FALSE POSITIVE: a blocked loop (`for start in range(0, k_total, step)`), each trip one batched operation over its block |
| feature_selection/filters/_gpu_nan_quantile_codes.py | nan_aware_quantile_codes_gpu | 50 | FALSE POSITIVE: a blocked loop (`for c0 in range(0, k, block)`), each trip one batched operation over its block |
| feature_selection/filters/_gpu_resident_fe.py | gpu_resident_pair_candidate_mi | 687 | FALSE POSITIVE: a blocked loop (`for start in range(0, len(_COMBOS), k_chunk)`), each trip one batched operation over its block |
| feature_selection/filters/_gpu_resident_k_chunk_ktc.py | gpu_resident_pair_candidate_mi_vram_fraction | 94 | FALSE POSITIVE: a blocked loop (`for start in range(0, len(_COMBOS), k_chunk)`), each trip one batched operation over its block |
| feature_selection/filters/_gpu_resident_materialise.py | gpu_discretize_codes_host | 809 | FALSE POSITIVE: a blocked loop (`for start in range(0, K, k_chunk)`), each trip one batched operation over its block |
| feature_selection/filters/_gpu_resident_materialise.py | gpu_materialise_discretize_codes_host | 578 | FALSE POSITIVE: a blocked loop (`for start in range(0, K, k_chunk)`), each trip one batched operation over its block |
| feature_selection/filters/_gpu_resident_pair_mi_parts.py | _grand_fused_pair_mi_fused_part1 | 222 | FALSE POSITIVE: a blocked loop (`for start in range(0, len(_COMBOS), k_chunk)`), each trip one batched operation over its block |
| feature_selection/filters/_gpu_resident_pair_mi_parts.py | _grand_fused_pair_mi_part1 | 67 | FALSE POSITIVE: a blocked loop (`for start in range(0, len(_COMBOS), k_chunk)`), each trip one batched operation over its block |
| feature_selection/filters/_gpu_resident_select.py | radix_quantiles_grouped | 414 | FALSE POSITIVE: a blocked loop (`for s in range(0, int(fr.size), int(group))`), each trip one batched operation over its block |
| feature_selection/filters/_group_distance_gpu.py | group_stats_device | 162 | FALSE POSITIVE: a blocked loop (`for c0 in range(0, k, block)`), each trip one batched operation over its block |
| feature_selection/filters/_grouped_quantile_fe.py | _grouped_quantile_mi_device | 679 | FALSE POSITIVE: a blocked loop (`for i in range(0, len(cols), step)`), each trip one batched operation over its block |
| feature_selection/filters/_grouped_quantile_fe.py | _prebinned_codes | 709 | FALSE POSITIVE: a blocked loop (`for i in range(0, len(keep), step)`), each trip one batched operation over its block |
| feature_selection/filters/_honest_null_accept_bits.py | _launch_pre | 357, 362 | FALSE POSITIVE: a blocked loop (`for b0 in range(0, n_perm, _PERM_CHUNK)`), each trip one batched operation over its block |
| feature_selection/filters/_hsic_gpu.py | hsic_biased_columns_gpu | 104 | FALSE POSITIVE: a blocked loop (`for start in range(0, k, width)`), each trip one batched operation over its block |
| feature_selection/filters/_ksg_cd_gpu.py | mi_cd_columns_device | 113 | FALSE POSITIVE: a blocked loop (`for c0 in range(0, n_cols, block)`), each trip one batched operation over its block |
| feature_selection/filters/_lof_gpu.py | lof_transform_gpu | 125 | FALSE POSITIVE: a blocked loop (`for s in range(0, n, block)`), each trip one batched operation over its block |
| feature_selection/filters/_mrmr_fe_step/_step_batched_marginals.py | batched_device_marginals | 46 | FALSE POSITIVE: a blocked loop (`for start in range(0, len(cols), per_chunk)`), each trip one batched operation over its block |
| feature_selection/filters/_null_max_perm_gpu.py | null_max_mi_perms_gpu | 260, 277 | FALSE POSITIVE: a blocked loop (`for p0 in range(0, len(perms), 64)`), each trip one batched operation over its block |
| feature_selection/filters/_orthogonal_copula_gpu.py | copula_columns_device | 132 | FALSE POSITIVE: a blocked loop (`for c0 in range(0, kk, block)`), each trip one batched operation over its block |
| feature_selection/filters/_orthogonal_copula_mi_fe.py | _copula_mi_batch | 338 | FALSE POSITIVE: a blocked loop (`for s0 in range(0, len(fast), block)`), each trip one batched operation over its block |
| feature_selection/filters/_orthogonal_univariate_fe/_streamed_gpu_mi.py | mi_classif_streamed_fill | 275 | FALSE POSITIVE: a blocked loop (`for k in range(n_blocks)`), each trip one batched operation over its block |
| feature_selection/filters/_orthogonal_xi_fe.py | xi_correlation_batch | 224 | FALSE POSITIVE: a blocked loop (`for s0 in range(0, finite.size, block)`), each trip one batched operation over its block |
| feature_selection/filters/_orthogonal_xi_gpu.py | xi_columns_device | 176 | FALSE POSITIVE: a blocked loop (`for c0 in range(0, kk, block)`), each trip one batched operation over its block |
| feature_selection/filters/_parity_fe.py | hybrid_parity_fe_with_recipes | 183 | FALSE POSITIVE: a blocked loop (`for start in range(0, len(cols), block)`), each trip one batched operation over its block |
| feature_selection/filters/_parity_gpu.py | parity_mi_grids_gpu | 101 | FALSE POSITIVE: a blocked loop (`for c0 in range(0, len(columns), COLUMN_BLOCK)`), each trip one batched operation over its block |
| feature_selection/filters/_permutation_null_pair_tiles.py | _pooled_pair_null_perm_tiles | 18 | FALSE POSITIVE: a blocked loop (`for p0 in range(0, nperm, perm_chunk)`), each trip one batched operation over its block |
| feature_selection/filters/_permutation_null_resident.py | pooled_gain_floor_perms_cupy | 99, 116 | FALSE POSITIVE: a blocked loop (`for c0 in range(0, ncand, cand_chunk)`), each trip one batched operation over its block |
| feature_selection/filters/_qbin_block_gpu.py | resident_qbin_codes_block | 164 | FALSE POSITIVE: a blocked loop (`for s0 in range(0, len(miss), step)`), each trip one batched operation over its block |
| feature_selection/filters/_qbin_block_gpu.py | sync_free_qbin_codes_block | 59 | FALSE POSITIVE: a blocked loop (`for c0 in range(0, k, _COMPARE_COLUMNS)`), each trip one batched operation over its block |
| feature_selection/filters/_quantile_codes_gpu.py | quantile_codes_block_gpu | 214 | FALSE POSITIVE: a blocked loop (`for c0 in range(0, k, block)`), each trip one batched operation over its block |
| feature_selection/filters/_quantile_codes_gpu.py | quantile_codes_resident_f32 | 269 | FALSE POSITIVE: a blocked loop (`for c0 in range(0, k, block)`), each trip one batched operation over its block |
| feature_selection/filters/_tail_dependence_gpu.py | tail_dependence_columns_device | 164 | FALSE POSITIVE: a blocked loop (`for c0 in range(0, k, block)`), each trip one batched operation over its block |
| feature_selection/filters/_triple_mi_gpu.py | triple_mi_gpu | 113 | FALSE POSITIVE: a blocked loop (`for s in range(0, m, TRIPLES_PER_LAUNCH)`), each trip one batched operation over its block |
| feature_selection/filters/_usability_pool_resident_pairs.py | _score_one_pair_combos | 37 | FALSE POSITIVE: a blocked loop (`for c0 in range(0, nc, combo_chunk)`), each trip one batched operation over its block |
| feature_selection/filters/discretization/_discretization_cuda.py | discretize_2d_array_cuda_row_chunked | 300, 312, 347 | FALSE POSITIVE: a blocked loop (`for row_start in range(0, n_rows, row_chunk_rows)`), each trip one batched operation over its block |
| feature_selection/shap_proxied_fs/_shap_proxy_cluster_su.py | _pairwise_su_edges_gpu | 238 | FALSE POSITIVE: a blocked loop (`for i_start in range(0, n_features, pair_chunk_size)`), each trip one batched operation over its block |
| metrics/_gpu_metrics.py | gpu_multiple_pr_auc_scores | 483 | FALSE POSITIVE: a blocked loop (`for c0 in range(0, M, block_cols)`), each trip one pass over its block of columns; its inner per-column loop only sorts (72 of 154 ms at 100000 x 100, memory bound, a flat sort is only 15% faster) |
| metrics/_gpu_metrics.py | gpu_multiple_roc_auc_scores | 427 | FALSE POSITIVE: a blocked loop (`for c0 in range(0, M, block_cols)`), each trip one flat rank pass over its block of columns |
