# Wave 20 audit 2026-11-02 -- tracker

Six read-only reports by Opus 5.5 low-effort agents on packages no earlier wave read: `feature_engineering`, `preprocessing` / `data_valuation` / `inspection` / `signal`, `reporting` / `integrations`, `inference` / `core` / `competition` / `votenrank` / `system`, the top-level feature-selection modules, and the info-theory / discretization / GPU kernels. The agents could not write files, so the reports are their returned text, condensed. Every claim was re-checked on HEAD before a fix. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Severity | Location |
|---|---|---|---|
| **RESOLVED** | W20FE-1 | High | `leakage_safe_aggregate` fast path let one NaN poison every later sum / mean of an entity and counted the NaN row |
| **RESOLVED** | W20FE-2 | High | causal step 2 of `two_step_recency_weighted_target_encode` underflowed to 0/0 = NaN for events > ~1000 half-lives before the entity's last one; now a per-step carry-forward kernel |
| **RESOLVED** | W20FE-3 | Medium | `nearest_past_join` fallback-tier sparsity check counts future right rows; needs a per-row cumulative group count |
| **RESOLVED** | W20FE-4 | Medium | `polars_dynamic_window` labels windows at their start, so a join back by label sees the following period; switching to `label="right"` changes the output keys, so it needs an explicit option |
| **RESOLVED** | W20FE-5 | Low | integer / float time columns are read as microseconds; the fix (reject or require a unit) changes the accepted inputs |
| **RESOLVED** | W20PRE-1 | Medium-High | PELT l2 cost from raw prefix sums cancels catastrophically far from zero; the kernel centres the signal first |
| **RESOLVED** | W20PRE-2 | Medium | `dro_reweight_fit` returned a model trained on the previous round's weights (the documented final full-data fit was missing) |
| **RESOLVED** | W20PRE-3 | Medium | `unseen_category_imputer` nearest mode maps to a NaN-mean category for large or NaN queries |
| **RESOLVED** | W20PRE-4 | Low-Medium | `smoothed_target_encode_column` shuffled its OOF folds with a fresh seed per call; default is now 0 |
| **RESOLVED** | W20PRE-5 | Low | the cleaning helper counted the NaN key among the rare values, so a column of one real value plus a lone NaN was flagged constant and dropped; the NaN key no longer counts toward the all-rare test (`tests/preprocessing/test_cleaning_nan_key_is_not_a_rare_value.py`; with the NaN key excluded, no column can be entirely rare, since the counts sum to the row count) |
| **RESOLVED** | W20PRE-6 | Low | `gp_smoothing` raises KeyError on a NaN entity |
| **RESOLVED** | W20INF-1 | High | `Leaderboard.elect_all` built its grid from the default settings, not the caller's `use_methods` |
| **RESOLVED** | W20INF-2 | Medium | `find_weights_for_condorcet` accepted a weighted tie as feasible while `condorcet_election` needs a strict win; it now demands a 1e-6 weighted margin over every rival (`tests/votenrank/test_condorcet_weights_require_a_strict_win.py`) |
| **RESOLVED** | W20INF-3 | Medium | the `get_meta_leaderboard` table was created object dtype and filled by position; it is now float64 NaN, each group's ranking is reindexed onto the model index, so a model a ranking omits is NaN instead of shifting the column (`tests/votenrank/test_meta_leaderboard_table_is_float_and_complete.py`) |
| **RESOLVED** | W20INF-4 | Medium | `collapse_predictions_by_group` lost a prediction (unweighted) or crashed (weighted) on a null group key; both now raise |
| **RESOLVED** | W20INF-5 | Low-Medium | `get_models_raw_predictions` returned `proba[:, 1]` as the positive class without looking at `classes_`; a binary model whose second class is not label 1 now logs a throttled warning naming the class the column belongs to (`tests/inference/test_raw_predictions_warn_when_positive_column_is_not_label_one.py`) |
| **RESOLVED** | W20INF-6 | Low | `segment_rank_splice` ranks used an unstable argsort |
| **PARTIAL** | W20INFO-1 | Medium | `mi_miller_madow(..., entropy_consistent=True)` and `mi_miller_madow_correct(..., k_xy=...)` give the entropy-consistent correction `I_plugin + (k_x + k_y - k_xy - 1)/(2n)` (`tests/feature_selection/info_theory/test_mi_miller_madow_entropy_consistent.py`); the default stays the product form until a selection-equivalence benchmark shows the consistent form does not move selections |
| **RESOLVED** | W20INFO-2 | Medium | `mixed_ksg_mi` jitter 1e-10 is absolute and vanishes above \|x\| ~ 1e6; scale it by the data spread or implement the tie-count rule |
| **RESOLVED** | W20INFO-3 | Low-Medium | the fused entropy kernels skipped the class-range check `merge_vars` has |
| **RESOLVED** | W20INFO-4 | Low | `conditional_mi_batched_cuda` launched the joint-histogram kernel without checking the shared-memory limit (only its dispatcher did); a direct call now raises a `ValueError` naming the histogram size (`tests/feature_selection/gpu/test_cmi_cuda_refuses_joint_histograms_beyond_shared_memory.py`) |
| **PARTIAL** | W20FS-1 | Medium | `select_features_noise_floor(rerank='permutation')` re-ranks features per shuffled y (opt-in, `tests/feature_selection/wrappers/test_noise_floor_reranks_per_permutation.py`); the default stays `'none'` because the re-ranked null falls with N and the plateau rule then keeps all 43 features on the 3-informative/40-noise control (n_perm 3 and 30), so a different stop rule has to come before any default flip |
| **RESOLVED** | W20FS-2 | Medium | polars `null_count()` ignores NaN, so an all-NaN float column survived the pre-screen that pandas drops |
| **RESOLVED** | W20FS-3 | Low-Medium | knockoffs imputed NaNs in the knockoff slot but left them in the real slot, which identified the real column; both slots are now mean-imputed (`tests/feature_selection/wrappers/test_knockoff_seeds_and_nan_slots.py`) |
| **RESOLVED** | W20FS-4 | Low | knockoffs and swap coins shared one seed; they now draw from two children of one `SeedSequence` (`tests/feature_selection/wrappers/test_knockoff_seeds_and_nan_slots.py`) |
| **RESOLVED** | W20REP-1 | High | decile gain / lift table breaks tied scores by row order; the fix (split tie groups across deciles proportionally) changes the table for every user with quantised scores |
| **RESOLVED** | W20REP-2 | Medium | risk-coverage curve breaks confidence ties by row order |
| **RESOLVED** | W20REP-3 | Medium | multiclass risk-coverage compared the argmax column index with the raw label value; labels now map to columns through their sorted order (labels already 0..K-1 keep their meaning), string labels work and an unmatchable label set raises (`tests/reporting/test_risk_coverage_multiclass_raw_labels.py`) |
| **RESOLVED** | W20REP-4 | Low | the decile caption said the top decile is 10% when fewer bins were built; it now states 100/bins percent (`tests/reporting/test_decile_caption_matches_bins.py`) |
| **RESOLVED** | W20REP-5 | Low | `flatten_classification_report` assumed every key held a dict; a top-level scalar is now kept under its own name (`tests/integrations/test_flatten_classification_report_scalars.py`) |

## Not covered by this wave

The `feature_engineering/transformer/` subpackage and most other feature_engineering modules; the rest of `inference/`, all of `core/`, the other `competition/` modules, the remaining votenrank blend modules and `system/`; `shap_proxied_fs/`, `importance.py` and the rest of `boruta_shap/`; the info-theory modules after `_cmi_cuda`, all of `discretization/` and the `_gpu_*` / `_batch_*` kernels; most of `reporting/` beyond the charts listed.

## Implemented

| ID | Regression test |
|---|---|
| W20FE-1 | `tests/feature_engineering/test_as_of_aggregate_skips_nan_history.py` |
| W20FE-2 | `tests/feature_engineering/test_two_step_causal_weights_do_not_underflow.py` |
| W20PRE-1, 4 | `tests/test_wave20_numeric_fixes.py` |
| W20PRE-2 | `tests/data_valuation/test_dro_final_model_matches_returned_weights.py` |
| W20INF-1, 6 | `tests/test_votenrank_fixes_wave20.py` |
| W20INF-4 | `tests/inference/test_entity_collapse_rejects_null_group_keys.py` |
| W20INFO-3 | `tests/feature_selection/info_theory/test_fused_entropy_kernels_check_class_range.py` |
| W20FS-2 | `tests/feature_selection/test_pre_screen_polars_counts_nan_like_pandas.py` |
| W20INFO-2 | `tests/feature_selection/test_mixed_ksg_tie_jitter_on_large_values.py` |
| W20FE-3 | `tests/feature_engineering/test_nearest_past_join_sparsity_counts_only_past_rows.py` |
| W20PRE-6 | `tests/signal/test_gp_smoothing_null_entity_is_skipped.py` |
| W20PRE-3 | `tests/preprocessing/test_unseen_category_imputer_nearest_ignores_nan_means.py` |
| W20REP-1 | `tests/reporting/test_decile_table_tied_scores.py` (decision of 2026-11-03: tied scores are averaged, no opt-out) |
| W20REP-2 | `tests/reporting/test_risk_coverage_tied_confidence.py` (decision of 2026-11-03: tied blocks are averaged) |
| W20FE-4, W20FE-5 | `tests/feature_engineering/test_polars_dynamic_window_label_and_time_unit.py` (decision of 2026-11-03: `label="right"` default, numeric time needs `time_unit`) |
