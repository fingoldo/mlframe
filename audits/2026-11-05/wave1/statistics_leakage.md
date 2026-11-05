# Wave 1 (2026-11-05), attack class A5: statistical correctness and data leakage

Read-only audit by one Opus 5.5 low-effort agent. Each defect was confirmed by reading the cited code on the current tree; coverage was partial (a first pass over the modules where this class shows up most). The report is the agent's returned text, condensed; each finding is re-checked on HEAD before it is fixed.

## Findings

| ID | Severity | Location | Defect |
|---|---|---|---|
| S5-01 | High | `feature_engineering/transformer/baseline_surprise.py:80` and nine sibling sites (listed below) | The inner baseline OOF loops are hard-coded `KFold(3, shuffle=True)` and ignore the groups the outer splitter carries. Only `_baseline_oof.fit_baseline_predict_oof` was made group-aware. |
| S5-02 | Medium | `training/composite/count_weighted_blend.py:115` | The auto-k CV for the entity-vs-global blend uses shuffled row-level KFold over an entity-keyed problem, so held-out entities are almost never unseen and the chosen `k` favours the entity model. |
| S5-03 | Medium | `training/composite/hpo_ensembling.py:162` | `honest_score` uses shuffled row KFold even when the pooled OOF came from `PurgedTimeSeriesSplit` (`hpo.py:384`), so the selection fold is temporally adjacent to the scoring fold and the "honest" number stays optimistic. |
| S5-04 | Low | `training/composite/count_weighted_blend.py:139` | The auto-k selection error is an unweighted mean even when `sample_weight` is passed to the fits, so `k` is chosen on a different loss than the model is trained on. |

### S5-01: inner baseline folds are not group-aware outside `_baseline_oof`

These sites build the train-side baseline predictions with an inner `KFold(n_splits=3, shuffle=True, random_state=seed+11).split(Xt)` and never receive `groups_t`:

- `baseline_surprise.py:80`
- `disagreement_band.py:93`
- `fisher_weighted_residual.py:63`
- `ib_baseline_codes.py:70`
- `multi_baseline_hard_row.py:95`
- `residual_stratified_distance.py:92`
- `pairwise_kl_divergence.py:88`
- `decision_region_depth.py:87`
- `sign_residual_baseline.py:90`
- `_residual_oof.py:42` (aux residual splitter)

The siblings `variance_baseline`, `y_quintile_baseline_knn`, `target_kmeans_codebook` and `predictive_info_delta` already pass `fold_train_groups(splitter, train_idx)` into `fit_baseline_predict_oof`. These ten do not.

Failure scenario: 5,000 patients x 10 visits, with the outer splitter bound to patient groups through the `_utils.py:381` wrapper. The outer folds hold out whole patients correctly. Inside each outer-train complement, the inner KFold puts visits of the same patient on both sides, so the train-row baseline predictions are near in-sample: residuals, surprise and disagreement come out too small. The query and validation rows get true held-out-group predictions. The engineered columns therefore have one distribution on train and another on query; their train-side signal is inflated and a downstream model learns cut-points that do not transfer. The same applies to `PurgedTimeSeriesSplit`-style temporal splitters, where the inner folds also shuffle across time.

### S5-02: CountWeightedBlendEnsemble picks `k` on a split with no cold-start rows

`_select_k_via_cv` (lines 113-141) uses `KFold(shuffle=True)` over rows. The blend weight is `counts/(counts+k)`, where `counts` is the entity's frequency in the fold-train rows. With shuffled rows, almost every validation entity also appears in fold-train with about (k_cv-1)/k_cv of its rows, so the CV never sees the situation `k` exists to handle: few or zero prior observations, as for a new entity at predict time. The entity model is also fitted on that same entity's other rows, so its validation error is optimistic.

Failure scenario: 2,000 stores x 20 rows each, deployed to score new or low-history stores. The CV MSE is minimised at small `k`, which trusts the entity model heavily; on held-out stores a larger `k` would be better, and the `k_cv_scores_` values are optimistic too. A group-aware split by `entity_col`, or the shared `make_discovery_splitter`, is what this needs.

### S5-03: `honest_score` ignores the HPO's time and purge structure

`select_oof_pool_ensemble` documents that `score` is optimistic and offers `honest_score` as the fix. `_nested_selection_score` splits rows with shuffled KFold, while `optimize_composite` builds its OOF with `PurgedTimeSeriesSplit` when `time_ordering` is given (the code at lines 116-117 even handles that splitter's warm-up NaN rows).

Failure scenario: a daily series with autocorrelated residuals. The stepwise selection is fitted on rows interleaved with the scoring rows: day t is used to select and day t+1 to score. Members whose OOF errors happen to be low over a regime are selected and then scored on the same regime, so `honest_score` stays biased toward `score`. Contiguous or forward folds, or `discovery_splits(..., time_aware=True)`, would remove that.

### S5-04: auto-k ignores sample weights

At line 139 the per-fold `sample_weight` is sliced and passed to both member fits, but the selection error stays an unweighted mean: `sq_err_by_k[k].append((pred - y_val) ** 2)`. With importance or recency weights, `k` is tuned for the unweighted population while the models target the weighted one.

## Checked and found sound

- `trainer._compute_oof_preds` takes the group and time branches before falling back to the iid splitter.
- `threshold_optimizer` stability now fits on the train side and scores on the held-out side.
- `smoothed_override_backtest._heldout_safe_mae` is a proper held-out estimate.
- `_honest_decision_threshold` selects on OOF.
- `make_discovery_splitter` precedence is group, then time, then KFold.
- The `TimeSeriesSplit` path for target-encoder OOF is in place.
- With replacement, the multiplicity of `stepwise_ensemble_selection` picks is preserved in `combined_oof`.

## Not covered

- Other composite modules: `gated_regression_mixture`, `dual_direction`, `hurdle`, `classification_discovery` and `feature_subset_bagging` also use raw shuffled KFold with no `groups` parameter. They were confirmed not to route through `make_discovery_splitter`; whether their callers can pass grouped or temporal data was not checked.
- Target encoder with a grouped `cv_splitter`: `target_encoders._kfold_encode` calls `kf.split(cats)` without groups. That would raise on a `GroupKFold` rather than leak; not traced further.
- Permutation nulls: whether the `cv_curve` nulls in `feature_selection/filters` and `wrappers/_noise_floor` permute y within groups or blocks was not checked.
- Calibration (`calibration/` fitters, `_calibration_oof_mask`, `_regression_calibration`) was not reviewed.
- Not reviewed: the `evaluation/` bootstrap cluster resampling, the `models/ensembling/score_*` stages, the `rfecv` `_cv_setup` group plumbing, `preprocessing/auto_transform_select` and `category_support`, and the `_aggregate_cv_early_stopping` early-stopping-versus-reported-metric separation.
