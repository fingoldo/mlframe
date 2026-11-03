# Wave 20 audit 2026-11-02 -- tracker

Six read-only reports by Opus 5.5 low-effort agents on packages no earlier wave read: `feature_engineering`, `preprocessing` / `data_valuation` / `inspection` / `signal`, `reporting` / `integrations`, `inference` / `core` / `competition` / `votenrank` / `system`, the top-level feature-selection modules, and the info-theory / discretization / GPU kernels. The agents could not write files, so the reports are their returned text, condensed. Every claim was re-checked on HEAD before a fix. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Severity | Location |
|---|---|---|---|
| **RESOLVED** | W20FE-1 | High | `leakage_safe_aggregate` fast path let one NaN poison every later sum / mean of an entity and counted the NaN row |
| **RESOLVED** | W20FE-2 | High | causal step 2 of `two_step_recency_weighted_target_encode` underflowed to 0/0 = NaN for events > ~1000 half-lives before the entity's last one; now a per-step carry-forward kernel |
| **RESOLVED** | W20FE-3 | Medium | `nearest_past_join` fallback-tier sparsity check counts future right rows; needs a per-row cumulative group count |
| **FUTURE** | W20FE-4 | Medium | `polars_dynamic_window` labels windows at their start, so a join back by label sees the following period; switching to `label="right"` changes the output keys, so it needs an explicit option |
| **FUTURE** | W20FE-5 | Low | integer / float time columns are read as microseconds; the fix (reject or require a unit) changes the accepted inputs |
| **RESOLVED** | W20PRE-1 | Medium-High | PELT l2 cost from raw prefix sums cancels catastrophically far from zero; the kernel centres the signal first |
| **RESOLVED** | W20PRE-2 | Medium | `dro_reweight_fit` returned a model trained on the previous round's weights (the documented final full-data fit was missing) |
| **RESOLVED** | W20PRE-3 | Medium | `unseen_category_imputer` nearest mode maps to a NaN-mean category for large or NaN queries |
| **RESOLVED** | W20PRE-4 | Low-Medium | `smoothed_target_encode_column` shuffled its OOF folds with a fresh seed per call; default is now 0 |
| **FUTURE** | W20PRE-5 | Low | cleaning helper counts the NaN key as a merged rare value and can drop a column that would be a binary indicator |
| **RESOLVED** | W20PRE-6 | Low | `gp_smoothing` raises KeyError on a NaN entity |
| **RESOLVED** | W20INF-1 | High | `Leaderboard.elect_all` built its grid from the default settings, not the caller's `use_methods` |
| **FUTURE** | W20INF-2 | Medium | `find_weights_for_condorcet` accepts a weighted tie as feasible while `condorcet_election` needs a strict win |
| **FUTURE** | W20INF-3 | Medium | `two_step_ranking` meta table is object dtype and can be partial |
| **RESOLVED** | W20INF-4 | Medium | `collapse_predictions_by_group` lost a prediction (unweighted) or crashed (weighted) on a null group key; both now raise |
| **FUTURE** | W20INF-5 | Low-Medium | `predict` returns `proba[:, 1]` without checking `classes_` |
| **RESOLVED** | W20INF-6 | Low | `segment_rank_splice` ranks used an unstable argsort |
| **FUTURE** | W20INFO-1 | Medium | `mi_miller_madow_correct` subtracts the product-form null bias `(k_x-1)(k_y-1)/(2n)`; the entropy-consistent Miller-Madow MI is `+(k_x+k_y-k_xy-1)/(2n)` and differs when joint cells are unoccupied. The product form is documented for the FE joint-prevalence gate; changing it moves selection and needs a selection-equivalence run |
| **RESOLVED** | W20INFO-2 | Medium | `mixed_ksg_mi` jitter 1e-10 is absolute and vanishes above \|x\| ~ 1e6; scale it by the data spread or implement the tie-count rule |
| **RESOLVED** | W20INFO-3 | Low-Medium | the fused entropy kernels skipped the class-range check `merge_vars` has |
| **FUTURE** | W20INFO-4 | Low | `_cmi_cuda` joint-histogram launch does not check the shared-memory limit |
| **FUTURE** | W20FS-1 | Medium-High | `select_features_noise_floor` ranks and scores on the same rows and does not redo the ranking per permutation |
| **RESOLVED** | W20FS-2 | Medium | polars `null_count()` ignores NaN, so an all-NaN float column survived the pre-screen that pandas drops |
| **FUTURE** | W20FS-3 | Low-Medium | knockoffs: NaNs imputed in the knockoff but not in the real slot |
| **FUTURE** | W20FS-4 | Low | knockoffs and swap coins share one seed |
| **RESOLVED** | W20REP-1 | High | decile gain / lift table breaks tied scores by row order; the fix (split tie groups across deciles proportionally) changes the table for every user with quantised scores |
| **FUTURE** | W20REP-2 | Medium | risk-coverage curve breaks confidence ties by row order |
| **FUTURE** | W20REP-3 | Medium | multiclass risk-coverage compares the argmax column index with raw labels |
| **FUTURE** | W20REP-4 | Low | decile caption says 10% when fewer bins are built |
| **FUTURE** | W20REP-5 | Low | `flatten_classification_report` assumes every key is a dict |

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
