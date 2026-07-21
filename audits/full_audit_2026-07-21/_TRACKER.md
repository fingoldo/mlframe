# full_audit_2026-07-21 -- master tracker

39-agent parallel read-only audit (33 subsystem clusters + 6 cross-cutting specialists: OSS hygiene/packaging, CI/CD & dependencies, test-suite architecture & coverage, cross-package architecture/API consistency, security/robustness, ML-practice meta-review). Scope: all of `mlframe` **except** `src/mlframe/feature_selection/filters/**` (MRMR engine) and `src/mlframe/feature_selection/shap_proxied_fs/**`, which are owned by the parallel `mrmr_audit_2026-07-20/` effort.

Unlike `mrmr_audit_2026-07-20`, this pass is **single-agent-per-cluster, not adversarially verified** (no 3-vote majority pass). Every finding below is one agent's read of the code, not yet cross-checked by a second pair of eyes -- treat as a strong lead, not a confirmed bug, until someone (human or a follow-up verification pass) re-derives it independently. Two clusters (`training_neural`, `reporting_charts`) hit tooling hiccups (a malformed summary block and a Write-tool refusal respectively); both reports were still fully produced and are on disk, recovered/reconstructed from the agents' full responses where needed -- noted in those reports' own text.

Totals across the 39 reports (files_reviewed/loc_reviewed include some cross-referenced context files agents read beyond their strict assigned scope, so these exceed the raw ~1015-file/~291k-LOC partition):

- **1270 files reviewed, 326508 LOC reviewed**
- **20 P0**, **165 P1**, **263 P2** findings
- **254 proposals** (missing tests, refactors, perf ideas, ML-practice improvements)

## Clusters with P0 findings -- read these first

20 P0 findings across 16 of the 39 reports, worst first by P0 count:

| Cluster | P0 | Report | Headline |
|---|---|---|---|
| fe_top_a | 3 | [fe_top_a.md](fe_top_a.md) | `compute_cross_sectional_neighbor_features` silently corrupts its neighbor-aggregate and distance-ratio features (confirmed empirically) whenever the distinct-snapshot count is <= k -- e.g. the default k=10 on a modest dataset -- because unmatched-neighbor placeholders (-1/inf) are used as real indices/distances instead of being masked out. |
| metrics_all | 2 | [metrics_all.md](metrics_all.md) | `fast_regression_metrics_block` / `fast_regression_metrics_block_extended` silently return R2 = -inf (poisoning NSE too) instead of sklearn's 0.0 for a constant y_true with any nonzero residual, and the existing regression test actually pins the wrong sklearn convention for the neighboring perfect-fit case. |
| preprocessing | 2 | [preprocessing.md](preprocessing.md) | `collapse_rare_categories` and `match_missingness_rate` crash or silently corrupt train/test category-space consistency on realistic inputs (no fit/apply split; mixed-dtype frame crash), and three other preprocessing helpers (`impute_with_missing_indicator`, `regime_conditioned_median_fill`, `apply_gaussian_power_transform`) similarly fuse fit+transform with no persisted state, risking train/serve statistic mismatch or leakage. |
| training_composite_blocks | 1 | [training_composite_blocks.md](training_composite_blocks.md) | `CompositeTargetEstimator.fit()` silently corrupts the training target for any grouped+recurrent transform (`ewma_residual_grouped` / `rolling_quantile_ratio_grouped` / `frac_diff_grouped`) whenever the domain filter drops even one row, because the per-group labels are pre-compacted while the recurrent forward runs on the full-length sequence, misaligning indices and leaving part of the output buffer uninitialized -- and no existing test exercises this combination. |
| training_composite_loose_b | 1 | [training_composite_loose_b.md](training_composite_loose_b.md) | `calendar_anomaly.py`'s anomaly-correction formula divides low-side outliers by their deviation ratio instead of multiplying, silently pushing low anomalies further from baseline instead of correcting them toward it. |
| training_core_b | 1 | [training_core_b.md](training_core_b.md) | `ctx.slug_to_original_target_type` is built as a dead local in `_main_train_suite.py` and never attached to `ctx`, so the persisted metadata's target-type slug map is always empty, causing every disk-loaded `predict_mlframe_models_suite` call to silently fall back to raw slug strings and skip multiclass-probability renormalisation. |
| training_neural | 1 | [training_neural.md](training_neural.md) | `PytorchLightningEstimator.get_params()` omits 10 of 23 constructor params (`random_state`, `class_weight`, `use_ema`, `label_smoothing`, `focal_loss_gamma`/`alpha`, etc.), so `sklearn.base.clone()` silently reverts them to defaults in any `cross_val_score`/`GridSearchCV`/`StackingClassifier` workflow -- confirmed via the codebase's own (stale) regression test for this exact contract. |
| training_baselines | 1 | [training_baselines.md](training_baselines.md) | A single NaN in a classification target's `train_y` crashes `compute_dummy_baselines` outright (uncaught ValueError from `np.bincount` on an int64-cast NaN), and the regression-side equivalent silently wipes the whole baseline table to `strongest=None` with a misleading "both splits degenerate" message -- both reproduced live against the public API. |
| training_targets | 1 | [training_targets.md](training_targets.md) | The polars aggregation path in `target_temporal_audit` silently deflates the target rate for any classification target with null values (empirically confirmed: 0.65-0.69 vs the honest 1.0), reproducing on the preferred >1M-row backend a bug already found and fixed on the pandas twin. |
| training_reporting_infra | 1 | [training_reporting_infra.md](training_reporting_infra.md) | `report_regression_model_perf`'s prediction-envelope clip (documented to apply BEFORE metrics AND chart) only reaches the numeric metrics/title -- the scatter chart, residual audit, and the function's own returned preds (later stored as `entry.test_preds` and consumed by every downstream diagnostic) all silently keep the raw, unclipped predictions. |
| training_misc_small | 1 | [training_misc_small.md](training_misc_small.md) | `build_slice_eval_sets(source="both", group_ids=...)` silently falls back to plain `KFold` instead of `GroupKFold`, breaking ranking query-group boundaries with no warning (unlike the `source="random"` path which explicitly guards and warns against this exact risk). |
| fe_transformer_c | 1 | [fe_transformer_c.md](fe_transformer_c.md) | `KeyBank`'s disk-cache fingerprint omits the `projection` build parameter, so reusing the same `cache_dir` with a different `projection=` silently serves a stale key-bank built under a different projection method with no error. |
| fe_top_b | 1 | [fe_top_b.md](fe_top_b.md) | `holiday_calendar_features.py` computes `is_holiday`/`is_eve` via exact datetime64 equality against midnight-normalized holiday dates with no normalization of the input, so any timestamped (non-midnight) date column silently gets `is_holiday`/`is_eve` always False with zero error or warning. |
| feature_selection_nonmrmr | 1 | [feature_selection_nonmrmr.md](feature_selection_nonmrmr.md) | `pre_screen.py`'s sparse-column variance check ignores fill-value mass, so it silently and permanently drops informative sparse rare-event/flag columns (e.g. fraud indicators) before any model or selector ever sees them. |
| core_infra_a | 1 | [core_infra_a.md](core_infra_a.md) | `fit_bin_smoother`/`apply_bin_smoother`/`bin_smooth` crash with `IndexError` on any constant-valued column even under fully default arguments, confirmed by direct reproduction and uncovered by any existing test. |
| x_ml_correctness_meta | 1 | [x_ml_correctness_meta.md](x_ml_correctness_meta.md) | `combine_probs`'s 'median' ensemble flavour feeds a per-row `sample_weight` vector into `np.quantile`'s `axis=0` (per-member) weights, an unfixed instance of a bug the same codebase already found and fixed at a sibling call site, and it has zero test coverage. |

## Full per-report index

| Report | Files | LOC | P0 | P1 | P2 | Proposals |
|---|---:|---:|---:|---:|---:|---:|
| [training_composite_discovery.md](training_composite_discovery.md) | 47 | 15715 | 0 | 2 | 4 | 6 |
| [training_composite_blocks.md](training_composite_blocks.md) | 54 | 14606 | 1 | 3 | 3 | 7 |
| [training_composite_loose_a.md](training_composite_loose_a.md) | 39 | 11619 | 0 | 4 | 14 | 10 |
| [training_composite_loose_b.md](training_composite_loose_b.md) | 38 | 11599 | 1 | 6 | 7 | 8 |
| [training_loose_a.md](training_loose_a.md) | 27 | 11283 | 0 | 4 | 7 | 5 |
| [training_loose_b.md](training_loose_b.md) | 27 | 11273 | 0 | 3 | 7 | 7 |
| [training_loose_c.md](training_loose_c.md) | 27 | 11277 | 0 | 7 | 11 | 7 |
| [training_core_a.md](training_core_a.md) | 36 | 13201 | 0 | 1 | 3 | 4 |
| [training_core_b.md](training_core_b.md) | 30 | 11227 | 1 | 2 | 6 | 5 |
| [training_neural.md](training_neural.md) | 44 | 12501 | 1 | 6 | 9 | 8 |
| [training_feature_handling.md](training_feature_handling.md) | 31 | 8564 | 0 | 8 | 9 | 8 |
| [training_baselines.md](training_baselines.md) | 24 | 6693 | 1 | 2 | 7 | 7 |
| [training_pipeline.md](training_pipeline.md) | 22 | 5522 | 0 | 4 | 5 | 7 |
| [training_targets.md](training_targets.md) | 16 | 4275 | 1 | 3 | 7 | 6 |
| [training_reporting_infra.md](training_reporting_infra.md) | 28 | 9087 | 1 | 2 | 6 | 6 |
| [reporting_charts.md](reporting_charts.md) | 40 | 14081 | 0 | 2 | 6 | 5 |
| [training_misc_small.md](training_misc_small.md) | 34 | 9434 | 1 | 4 | 3 | 8 |
| [fe_transformer_a.md](fe_transformer_a.md) | 40 | 6965 | 0 | 7 | 7 | 4 |
| [fe_transformer_b.md](fe_transformer_b.md) | 39 | 6912 | 0 | 16 | 13 | 8 |
| [fe_transformer_c.md](fe_transformer_c.md) | 40 | 6967 | 1 | 8 | 6 | 6 |
| [fe_top_a.md](fe_top_a.md) | 39 | 10911 | 3 | 12 | 9 | 9 |
| [fe_top_b.md](fe_top_b.md) | 39 | 10939 | 1 | 2 | 10 | 6 |
| [feature_selection_nonmrmr.md](feature_selection_nonmrmr.md) | 32 | 8285 | 1 | 2 | 7 | 7 |
| [feature_selection_wrappers.md](feature_selection_wrappers.md) | 38 | 9475 | 0 | 4 | 5 | 8 |
| [metrics_all.md](metrics_all.md) | 39 | 14086 | 2 | 6 | 3 | 6 |
| [calibration.md](calibration.md) | 18 | 4823 | 0 | 2 | 7 | 7 |
| [evaluation.md](evaluation.md) | 21 | 4759 | 0 | 3 | 7 | 7 |
| [models_all.md](models_all.md) | 41 | 9266 | 0 | 2 | 6 | 6 |
| [preprocessing.md](preprocessing.md) | 23 | 4155 | 2 | 8 | 6 | 10 |
| [competition.md](competition.md) | 34 | 3795 | 0 | 7 | 4 | 9 |
| [votenrank.md](votenrank.md) | 36 | 4226 | 0 | 3 | 12 | 5 |
| [core_infra_a.md](core_infra_a.md) | 35 | 5990 | 1 | 4 | 5 | 7 |
| [core_infra_b.md](core_infra_b.md) | 56 | 6878 | 0 | 3 | 8 | 6 |
| [x_oss_hygiene_packaging.md](x_oss_hygiene_packaging.md) | 22 | 2255 | 0 | 1 | 6 | 5 |
| [x_cicd_dependencies.md](x_cicd_dependencies.md) | 20 | 3222 | 0 | 4 | 3 | 4 |
| [x_test_suite_architecture.md](x_test_suite_architecture.md) | 27 | 2831 | 0 | 4 | 6 | 7 |
| [x_architecture_api_consistency.md](x_architecture_api_consistency.md) | 17 | 7345 | 0 | 0 | 9 | 4 |
| [x_security_robustness.md](x_security_robustness.md) | 27 | 3737 | 0 | 1 | 4 | 4 |
| [x_ml_correctness_meta.md](x_ml_correctness_meta.md) | 23 | 6729 | 1 | 3 | 6 | 5 |
| **Total** | **1270** | **326508** | **20** | **165** | **263** | **254** |

## Notes on tooling hiccups

- **training_neural**: the agent's final message ended without a clean fenced ```json``` summary block (extra prose after it), so the orchestrator's parser fell back to a raw-text capture. The report file itself was written successfully and is complete; counts above were hand-recovered from the agent's own prose summary (verified against the report file's own tables).
- **reporting_charts**: the agent's Write tool call was refused by the tool runtime ("Subagents should return findings as text, not write report files") on this one agent -- not observed on any of the other 38. The full report content was recovered verbatim from the agent's final response text and written to disk by the orchestrating session afterward; the report itself notes this at the end of its Coverage notes section.
