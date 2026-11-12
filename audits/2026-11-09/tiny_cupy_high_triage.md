# Tiny-cupy-loops, high severity: triage list (2026-11-12)

The detector's `high` hits in the current source, one row per function. The baseline (`tests/test_meta/_tiny_cupy_loops_baseline.json`) holds these keys.
Dispositions to fill in: **FALSE POSITIVE** (the loop is a chunk, fold or fallback loop with a few trips and real work per trip), **FALLBACK** (the cupy loop runs only when a batched path is not taken, measured), **REAL** (a launch-bound loop, to restructure), **OPEN** (not yet read).

Rows: 115.

| file | function | line(s) | disposition |
|---|---|---|---|
| feature_engineering/transformer/random_features.py | compute_rff_features | 190 | FALSE POSITIVE: the loop is the Mode A fold loop (5 to 10 trips), each a full fit and projection (GPU11-59) |
| feature_selection/filters/_batch_mi_noise_gate_screen_v1.py | batch_mi_with_noise_gate_cupy_v1 | 139 | LEGACY: no caller in src beyond the re-export and `__all__` of batch_mi_noise_gate_gpu (tests only); a host loop over permutations, not on a fit path |
| feature_selection/filters/_binned_numeric_agg_resident.py | _binagg_null_max_vector | 592 | FALLBACK: runs only when the batched null is disabled or fails (counted 0 calls by default; GPU11-62) |
| feature_selection/filters/_binned_numeric_agg_resident.py | _pair_fold_stat_tables | 372 | FALSE POSITIVE: the fallback fold loop, one full pass per fold (GPU11-60) |
| feature_selection/filters/_binned_numeric_agg_resident.py | build_binagg_oof_matrix_gpu | 432 | REAL, measured partly launch-bound: 32 us per column launch at 100000 rows, batching open (GPU11-61) |
| feature_selection/filters/_cat_confirm_permutation.py | _confirm_pairs_via_permutation | 919 | REAL, not yet measured: per surviving pair, a permutation null with launches per permutation; the same class as the batched-permutation item (GPU11-62) |
| feature_selection/filters/_cat_confirm_permutation.py | _count_nfailed_joint_indep_cupy | 313 | REAL: one permutation per trip, about eight cupy launches per trip over N rows, launch-bound; batching the permutations (as the null path does) is the restructure |
| feature_selection/filters/_conditional_gate_fe.py | _gate_flush_pending | 717 | FALSE POSITIVE: one trip per pending candidate group (a few), each a host argmax and, for the groups that clear the margin, one fixed 12-permutation null; not a per-element loop |
| feature_selection/filters/_conditional_gate_fe.py | _rank_and_prune | 615 | REAL, not yet measured: a per-column device loop in the main path (the column gathers and medians), not only in the fallback |
| feature_selection/filters/_conditional_gate_fe.py | cheap_row_argmax_scan | 513 | REAL, deliberate and budgeted: one device argmax per 3-column combination, the source states that the launch count is kept on purpose until the suite is decoupled from pool ordering |
| feature_selection/filters/_cupy_polynom_optimizer.py | _score_generation_gpu | 171 | FALSE POSITIVE: loops over the fixed basis families (about nine), each one batched over the whole population |
| feature_selection/filters/_cupy_polynom_optimizer.py | run_cupy_kernel_search | 251 | FALSE POSITIVE: loops over the sequential generations of the search, each trip one whole batched generation |
| feature_selection/filters/_extra_fe_families_dispersion_resident.py | build_dispersion_matrix_gpu | 129 | REAL, not yet measured: one device build per output column, the same class as the binned-aggregate builder (GPU11-61) |
| feature_selection/filters/_extra_fe_families_dispersion_resident.py | build_residual_abs_matrix_gpu | 196 | REAL, not yet measured: one device build per output column, the same class as build_dispersion_matrix_gpu |
| feature_selection/filters/_family_maxt_gate.py | _family_null_max_device_bits | 356 | FALSE POSITIVE: loops over permutation blocks (n_perm over the block size), each trip one batched device null over a block of permutations |
| feature_selection/filters/_family_maxt_stream.py | streamed_null_max.null_max | 89 | FALSE POSITIVE: loops over the column groups admitted by the stream, each trip one full device null over its group |
| feature_selection/filters/_fe_additive_fusion.py | _collect_relevant_halves | 204 | REAL, not yet measured: a full permutation null per pre-candidate, per-candidate launch work |
| feature_selection/filters/_fe_additive_fusion.py | _subsumed_raw_operand_check | 489 | FALSE POSITIVE: loops over the few raw operand tokens of a pair, each one a full check |
| feature_selection/filters/_fe_additive_fusion_gpu_resident.py | _gpu_relevant_halves | 207 | REAL, not yet measured: a full permutation null per half, the same class as _collect_relevant_halves |
| feature_selection/filters/_fe_additive_fusion_gpu_resident.py | _subsumed_raw_operand_check_gpu | 467 | FALSE POSITIVE: loops over the few raw operand tokens of the halves, each one a check (the twin of the host check) |
| feature_selection/filters/_fe_additive_fusion_gpu_resident.py | propose_additive_fusions_gpu | 422, 428 | REAL, not yet measured: one pair evaluation (OLS fits and MI on the device) per token-disjoint half pair, quadratic in the halves and capped by the fusion limit |
| feature_selection/filters/_fe_auto_escalation.py | _apply_pair_maxt_floor | 722 | REAL, not yet measured: a full permutation null per candidate, the same class as _collect_relevant_halves |
| feature_selection/filters/_fe_auto_escalation.py | _best_admitted_column | 118 | REAL, not yet measured: one marginal MI per admitted column, launch work per column |
| feature_selection/filters/_fe_auto_escalation.py | _propose_pair_candidates | 411, 440 | FALSE POSITIVE: loops over the few eligible pairs, each trip one whole candidate proposer (heavy per-pair work) |
| feature_selection/filters/_fe_auto_escalation.py | _slice_admitted_pool | 295 | REAL when the device path is taken, not yet measured: each admitted feature's marginal CMI takes the device route on its heuristic, one launch set per feature |
| feature_selection/filters/_fe_auto_escalation.py | find_underdelivering_pairs | 215 | REAL, not yet measured: per pair a best-column scan, two conditional MIs and a permutation null |
| feature_selection/filters/_fe_cmi_redundancy_gate.py | _gate_round | 486 | REAL, not yet measured: one resident CMI and permutation null per remaining candidate |
| feature_selection/filters/_fe_cmi_redundancy_null.py | _conditional_cpu_null | 440 | REAL when the device path is taken, not yet measured: each shuffle's CMI takes the device route on its heuristic, one launch set per shuffle |
| feature_selection/filters/_fe_pure_form_retention_gpu_resident.py | _batched_gate_scalars | 128 | REAL, not yet measured: a device operation per pair over P pairs |
| feature_selection/filters/_fe_raw_redundancy_anchors.py | _build_raw_redundancy_step2_def_raw_codes | 262 | REAL, not yet measured: one device code build per engineered column |
| feature_selection/filters/_fe_raw_redundancy_drop.py | _drop_redundant_raw_o_step1_full_composite_bin | 708 | REAL, not yet measured: per-consumer device composite bins; the selection-exact short circuit skips some work |
| feature_selection/filters/_fe_raw_redundancy_drop.py | _some_consumer_adds_information | 370 | REAL, not yet measured: one excess and permutation floor per consumer, with an early return |
| feature_selection/filters/_fe_stability_vote.py | _vote_recipe_over_folds | 222 | FALSE POSITIVE: loops over the few folds, each trip a full replay and scoring over the fold's rows |
| feature_selection/filters/_fe_sufficient_summary_pairs.py | best_pair_streamed | 412, 415 | FALSE POSITIVE: loops over the streamed column blocks, each trip a full block product |
| feature_selection/filters/_feature_engineering_pairs/_pairs_emit_extval.py | _emit_pair_features_step2_read_here_once | 222 | FALSE POSITIVE: loops over the few transformation configurations, each trip a full feature emission |
| feature_selection/filters/_gpu_resident_basis.py | _gpu_evaluate_basis_matrix | 808 | FALSE POSITIVE: loops over the few (basis, robust) groups, each trip one batched build |
| feature_selection/filters/_gpu_resident_basis.py | _gpu_route_bases_batched | 949 | FALSE POSITIVE: loops over the few candidate bases, each one batched over the columns |
| feature_selection/filters/_gpu_resident_extval.py | _candidate_matrix | 79, 81 | FALLBACK: the per-candidate loop runs only when `MLFRAME_EXTVAL_FUSED` is off; the default fused kernel builds all candidates in one launch |
| feature_selection/filters/_gpu_resident_operand_build.py | _batched_unary_prepass | 222 | FALSE POSITIVE: loops over the few batch-key groups, each one kernel over its group |
| feature_selection/filters/_gpu_resident_operand_build.py | _build_resident_operand_table_part1 | 268 | REAL, not yet measured: one device build per column that is not in a batch |
| feature_selection/filters/_gpu_strict_fe/_state.py | build | 82 | FALSE POSITIVE: loops over the few devices, each one bulk upload |
| feature_selection/filters/_hinge_basis_fe.py | _detect_hinge_breakpoints_for_columns | 551 | REAL, not yet measured: one hinge scan per column, budgeted by the deadline |
| feature_selection/filters/_hinge_detect_gpu_resident.py | _hinge_full_scan | 193 | FALSE POSITIVE: loops over the few breakpoint rounds, each a full projection |
| feature_selection/filters/_hinge_detect_gpu_resident_batch.py | _batched_precheck_gpu | 75 | FALSE POSITIVE: the loop is over the fixed quantile levels (a few trips), each a K-wide vector operation |
| feature_selection/filters/_hinge_detect_gpu_resident_batch.py | detect_hinge_breakpoints_gpu_batch | 147 | REAL, not yet measured: one single-column detection per column that passed the batched precheck |
| feature_selection/filters/_honest_oj_fused_gpu.py | picks | 342 | FALSE POSITIVE: loops over the few folds, each a full array operation |
| feature_selection/filters/_mi_greedy_cmi_fe_steps.py | _per_candidate_cmi_argmax | 877 | REAL when the device path is taken, not yet measured: one conditional MI per scanned candidate |
| feature_selection/filters/_mi_greedy_cmi_fe_steps.py | _renumber_joint_gpu | 282 | FALSE POSITIVE: loops over the few conditioning columns, each a full-size refactorisation |
| feature_selection/filters/_mi_greedy_device.py | score_candidates_device | 142 | REAL, not yet measured: one device op plus a cast per candidate transform, then the MI is batched per block; thousands of candidates mean thousands of small elementwise launches |
| feature_selection/filters/_modular_codes_gpu.py | modular_family_codes_gpu | 131 | FALSE POSITIVE: loops over the streamed column blocks, each one full block build |
| feature_selection/filters/_mrmr_fe_step/_pair_admission_batch.py | _block_verdicts | 66 | REAL, not yet measured: one conditional permutation null per column that does not apply the batched route |
| feature_selection/filters/_mrmr_fe_step/_pair_admission_batch.py | _score_groups | 125 | REAL, not yet measured: one device block per anchor group, blocked by the column budget |
| feature_selection/filters/_mrmr_fe_step/_pair_admission_batch.py | batched_asymmetric_relax_verdicts | 169 | REAL, not yet measured: one device block per partner group, blocked by the column budget |
| feature_selection/filters/_mrmr_fe_step/_step_pairs_rank.py | _prepass_gate_and_usability_candidates | 538 | REAL, not yet measured: one prevalence-gate resolution per sorted pair; the relax verdicts are batched up front, the per-pair gate is not |
| feature_selection/filters/_mrmr_fe_step/_step_score.py | _score_candidates_one_by_one | 57, 60 | REAL, not yet measured: per candidate one upload, one device quantile binning and one marginal MI over the whole CMI-gate pool |
| feature_selection/filters/_mrmr_fe_step/_step_score_parts2.py | _escalation_admitted_pool | 480, 483 | FALLBACK: the loop runs only when the batched escalation pool is not done (the iterable is empty otherwise) |
| feature_selection/filters/_mrmr_fe_step/_step_score_parts2.py | _fused_compound_codes | 608 | REAL, not yet measured: one device code conversion per fused feature |
| feature_selection/filters/_mrmr_fit_impl/_fe_stage_concurrency.py | merge_stage_state | 107, 108 | FALSE POSITIVE: merges host attributes of the stage copies, with no device launch in the loop body |
| feature_selection/filters/_offset_product_gpu.py | scan_offset_products_device | 332, 335 | REAL but small: 0.1% of a 100000-row fit (profile 2026-11-05); restructuring not justified by the measured share |
| feature_selection/filters/_orth_auto_scorer_fe.py | _batched_bootstrap_values | 352 | FALLBACK: the sequential resample loop is the else branch, used when the batched path is not taken |
| feature_selection/filters/_orthogonal_adaptive_arity_fe.py | _best_cell_per_tuple | 166 | FALSE POSITIVE: the loop is over arity (2 to max_arity, two or three trips), and each trip is one full batched MI call |
| feature_selection/filters/_orthogonal_cmim_fe.py | _cmim_scores_device | 391 | REAL, not yet measured: one device update per selected reference over the candidate block |
| feature_selection/filters/_orthogonal_cmim_fe.py | cmim_score | 306 | FALLBACK: the per-candidate host CMI path; score_features_by_cmim runs the batched device scores first (line 510) |
| feature_selection/filters/_orthogonal_cmim_fe.py | score_features_by_cmim | 511 | FALSE POSITIVE: a host loop that assembles results from device scores computed once before it |
| feature_selection/filters/_orthogonal_jmim_fe.py | _jmim_scores_device | 223 | REAL, not yet measured: one device update per selected reference over the candidate block, the same class as _cmim_scores_device |
| feature_selection/filters/_orthogonal_routing_device.py | routing_scores_on_device | 110 | FALSE POSITIVE: loops over the few candidate bases, each one batched over the columns |
| feature_selection/filters/_orthogonal_univariate_fe/_extra_basis_resident.py | _build_extra_basis_matrix_gpu | 156 | REAL, not yet measured: one device build per named column |
| feature_selection/filters/_orthogonal_univariate_fe/_fourier_batch_gpu.py | _run_group | 281 | FALSE POSITIVE: the sequential frequency-peeling iterations (max_freqs, a few), each batched over the group's members |
| feature_selection/filters/_orthogonal_univariate_fe/_fourier_batch_gpu.py | detect_fourier_freqs_batch_gpu | 360 | FALSE POSITIVE: loops over the few size groups, each one batched over its members |
| feature_selection/filters/_orthogonal_univariate_fe/_fourier_detect_gpu_resident.py | detect_fourier_freqs_for_col_gpu | 356 | REAL, not yet measured: the single-column detector, several small launches per iteration per column; the batched twin exists |
| feature_selection/filters/_orthogonal_univariate_fe/_gpu_resident_cross_basis.py | _prefetch_legs | 260 | FALSE POSITIVE: loops over the few degrees, each one stack of resident legs |
| feature_selection/filters/_orthogonal_univariate_fe/_gpu_resident_cross_basis.py | build_leg_product_matrix_gpu | 179, 182 | REAL, not yet measured: one product per output column over its legs |
| feature_selection/filters/_orthogonal_univariate_fe/_orth_dedup.py | _pc_corr_cupy | 199 | FALSE POSITIVE: the memory-tile loop, each trip one tile GEMM |
| feature_selection/filters/_orthogonal_univariate_fe/_orth_mi_null_max.py | _null_max_strict_gpu | 180 | REAL, not yet measured: one label upload and one batched MI per permutation, the same class as the pre-batched null (GPU11-62); whether a batched path precedes it is not checked |
| feature_selection/filters/_orthogonal_univariate_fe/_streamed_gpu_mi.py | mi_classif_streamed | 157 | FALSE POSITIVE: the planned streaming loop over column blocks, each a full block MI |
| feature_selection/filters/_orthogonal_univariate_fe/_streamed_gpu_mi.py | null_max_streamed | 221, 241 | FALSE POSITIVE: the planned streaming loop over column blocks, each a full block null |
| feature_selection/filters/_pairwise_modular_resident.py | residue_grid_mis_resident | 143 | FALSE POSITIVE: loops over the few effective-bin groups, each one batched MI |
| feature_selection/filters/_permutation_null_pair_resident.py | pooled_pair_permutation_null_joint_mi_floor_cupy | 144 | REAL, not yet measured: a bincount and several small ops per pair over n_pairs |
| feature_selection/filters/_resident_candidate_mi.py | gate_grid_mi_resident | 376 | REAL, not yet measured: one tau-grid matrix build (several ops) per spec before the batched MI |
| feature_selection/filters/_unified_fe_gate.py | unified_second_pass_gate | 570 | REAL, not yet measured: one conditional MI per remaining candidate per greedy round |
| feature_selection/filters/_usability_aware_selection.py | _build_usability_cand_step1_ua_codes_none | 483, 484, 485 | REAL, not yet measured: one device op per (unary, unary, binary) combination |
| feature_selection/filters/_usability_aware_selection.py | _build_usability_cand_step2_sub_quantum_cpu | 515 | FALSE POSITIVE: a host loop that builds a recipe per kept candidate, with no device launch per trip |
| feature_selection/filters/_usability_aware_selection.py | _build_usability_cand_step3_pidx_n1_n2 | 376 | REAL, not yet measured: one candidate build per operand pair, each running the combination loops |
| feature_selection/filters/_usability_aware_selection.py | build_usability_candidate_pool | 252 | REAL, not yet measured: one marginal MI per raw column |
| feature_selection/filters/_usability_greedy_clf_batched.py | _newton | 100 | FALSE POSITIVE: the sequential Newton iterations, each one batched over the candidates |
| feature_selection/filters/_usability_greedy_clf_batched.py | cv_candidates_binary | 34 | FALSE POSITIVE: loops over the few folds, each one batched over the candidates |
| feature_selection/filters/_usability_greedy_clf_gpu_resident.py | _usability_greedy_c_range_min | 555, 560 | FALSE POSITIVE: the sequential greedy rounds, each one batched cross-validation call |
| feature_selection/filters/_usability_greedy_clf_gpu_resident.py | _usability_greedy_clf_multinomial_fns._fit_multinomial | 338 | FALSE POSITIVE: the sequential Newton iterations, each full-size |
| feature_selection/filters/_usability_greedy_clf_gpu_resident.py | _usability_greedy_clf_step1_def_fit_binary._fit_binary | 394 | FALSE POSITIVE: the sequential Newton iterations, each full-size |
| feature_selection/filters/_usability_greedy_clf_gpu_resident.py | _usability_greedy_clf_step3_def_shortlist_sel._shortlist | 519 | FALSE POSITIVE: a host shortlist loop over scored candidates with host correlation vectors |
| feature_selection/filters/_usability_greedy_clf_gpu_resident.py | usability_greedy_clf_gpu_resident._cv_candidate | 266 | FALSE POSITIVE: loops over the few folds, each one device fit |
| feature_selection/filters/_usability_greedy_gpu_resident.py | _usability_greedy_g_range_min | 418 | FALSE POSITIVE: the sequential greedy rounds, each one batched cross-validation call |
| feature_selection/filters/_usability_greedy_gpu_resident.py | _usability_greedy_gpu_step2_def_shortlist_sel._shortlist | 273 | FALSE POSITIVE: a host shortlist loop with host correlation vectors |
| feature_selection/filters/_usability_greedy_gpu_resident.py | _usability_greedy_gpu_step3_def_cv_candidates._cv_candidates | 353, 359 | REAL, not yet measured: several device ops per candidate over the folds, with one D2H per candidate |
| feature_selection/filters/_usability_greedy_steps.py | _usability_greedy_step4_consistency_across_folds | 249 | FALSE POSITIVE: the sequential greedy rounds, each one batched scoring call |
| feature_selection/filters/_usability_njit_pool.py | _gpu_marginal_mi | 218 | REAL, not yet measured: several small device ops per candidate row for the entropies |
| feature_selection/filters/_usability_njit_pool.py | _pair_combo_mi_cupy | 296 | FALSE POSITIVE: loops over the few distinct binary operations, each one op over its columns |
| feature_selection/filters/_usability_njit_pool.py | _pair_combo_mi_cupy_loop | 350 | REAL, not yet measured: one apply per combination; the grouped sibling batches by operation |
| feature_selection/filters/_usability_pool_resident.py | score_pair_combos_table_resident | 251 | REAL, not yet measured: one scoring call per pair, with the D2H already collapsed to one per table |
| feature_selection/filters/_wavelet_basis_fe.py | _heldout_incremental_mi_from_prep | 661 | FALSE POSITIVE: the fixed 12-permutation null, a few trips |
| feature_selection/filters/_wavelet_basis_fe.py | _heldout_incremental_mi_prep | 590 | FALSE POSITIVE: the fixed 12-permutation baseline, a few trips |
| feature_selection/filters/_wavelet_basis_fe_batched.py | _per_leg_matrices | 133 | FALLBACK when the fused builder fits one block; in the blocked mode each trip is an O(n) leg build over at least the block budget, so launch overhead is a small share (GPU11-68) |
| feature_selection/filters/_wavelet_basis_fe_batched.py | _select_wavelet_legs_batched_device | 228 | FALSE POSITIVE: the byte-bounded block loop (GPU11-68), each trip one batched MI over its legs |
| feature_selection/filters/batch_pair_mi_gpu.py | batch_pair_mi_cupy | 133 | REAL, not yet measured: several device ops per pair over n_pairs in the plain cupy path |
| feature_selection/filters/hermite_fe/__init__.py | _polyeval_cuda_pick_devices | 314 | FALSE POSITIVE: loops over the devices (one to a few), each a memory query |
| feature_selection/filters/hermite_fe/__init__.py | _polyeval_try_cuda | 398 | FALSE POSITIVE: loops over the devices with per-device fault isolation, each trip one launch |
| feature_selection/filters/hermite_fe/_als_fused_gpu.py | als_sweep_fused | 206 | FALSE POSITIVE: the sequential ALS iterations, each a few fused kernels (launches are documented launch-ok) |
| feature_selection/filters/hermite_fe/_hermite_prewarp.py | warm_start_als_seed | 318 | FALSE POSITIVE: the sequential ALS iterations of a host numpy seed, a few trips |
| feature_selection/filters/hermite_fe/_hermite_prewarp_gpu_resident.py | _als_sweep_gpu | 173 | FALSE POSITIVE: the sequential ALS iterations, each a few fused device ops |
| feature_selection/shap_proxied_fs/_shap_proxy_cluster.py | _edges_blocked | 122 | REAL, not yet measured: a where and slice per row of a block, two or more launches per row |
| feature_selection/shap_proxied_fs/_shap_proxy_gpu.py | brute_force_top_n_gpu | 141 | FALSE POSITIVE: loops over the few combination sizes, each one batched |
| metrics/_gpu_metrics.py | compute_batch_aucs | 597 | FALLBACK: the per-column branch; the batched call over all columns is the default |
| metrics/_gpu_metrics.py | gpu_multiple_pr_auc_scores | 462 | REAL, not yet measured: a sort and several ops per column over M columns |
| metrics/_gpu_metrics.py | gpu_multiple_roc_auc_scores | 398 | REAL, not yet measured: a rank pass with ties per column over M columns |
| models/ensembling/_correlation.py | _probs_pairs | 94 | FALSE POSITIVE: loops over the classes on host numpy arrays |
