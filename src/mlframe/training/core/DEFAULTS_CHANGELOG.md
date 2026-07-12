# train_mlframe_models_suite — default-flip changelog

Explicit user directive (2026-07-12): flip every newly-wired opt-in extension (added during a large
session-long wiring effort connecting previously-ISOLATED mlframe utilities into `train_mlframe_models_suite`)
to be **ON by default**, including behavior-changing ones, accepting the resulting runtime/results change for
existing callers of the suite. This file records what was flipped, what was deliberately left opt-in with a
stated reason, and the bugs the flip work surfaced and fixed along the way.

## Flipped to default-ON (2026-07-12)

- **`OutputConfig.run_diagnostics`**: default changed from `None` to all 6 registered
  `mlframe.evaluation` diagnostics (`cv_informativeness`, `compare_cv_schemes`, `group_leakage`,
  `constant_group_leak`, `adversarial_fold_selection`, `subpopulation_drift`). Results land in
  `metadata["diagnostics"]`; a diagnostic that errors reports `{"error": ...}` rather than aborting the
  suite. `adversarial_fold_selection` is the one materially slower diagnostic (fits a full adversarial
  train-vs-val classifier) — kept in the default list per the "flip everything" directive, but it's the one
  worth dropping via an explicit `run_diagnostics=[...]` override on latency-sensitive runs.
- **`FeatureSelectionConfig.use_forward_select_fs` / `use_greedy_backward_elimination_fs` /
  `use_zero_importance_pruning_fs` / `use_cascade_select_fs`**: all four changed from `False` to `True`.
  Each is an additive branch in `_build_pre_pipelines` (its own entry in `pre_pipelines`, scored as an
  independent candidate alongside MRMR/RFECV/BorutaShap/ShapProxiedFS/ACE) — enabling all four does not
  conflict with or duplicate-prune against the other selectors, it adds four more scored candidates for the
  suite's own model-selection step to compare. Real, accepted runtime cost: four extra CV-scored search
  loops per training run.
- **`TrainingBehaviorConfig.auto_optimize_threshold`**, **`TrainingBehaviorConfig.check_isotonic_overfit_risk`**,
  **`RegressionCalibrationConfig.apply_confidence_shrinkage`**: all flipped to `True`.
- **`"gated_outlier"` (`GatedOutlierEstimator`) auto-inclusion**: NOT blanket-added to the default
  `mlframe_models` allowlist (`["cb","lgb","xgb","mlp","linear"]`). Instead, gated on a runtime auto-detection:
  when the caller left `mlframe_models` at its top-level `None` default (`mlframe_models_is_default_allowlist`)
  AND the current target is regression AND the current target's train split shows a genuine point mass
  (>=5% of rows share the single most common value — the "zero-inflated"/"no purchase" pattern
  `GatedOutlierEstimator` is designed for), `gated_outlier` is auto-appended as an extra candidate model.
  A caller who passes an **explicit** `mlframe_models=[...]` allowlist never gets this auto-append — an
  explicit allowlist is a deliberate model-set decision the suite must not silently expand, per the
  documented "mlframe_models filters which models train" contract. Blanket inclusion on every regression
  target (rather than gating on an actual point-mass signal) would waste compute and risk crashes on
  ordinary continuous targets (too few/no rows in the classifier gate's positive class).

## Deliberately left opt-in (not flipped)

- **`SegmentedModelFactory`**, **`GatedRegressionMixture`**, **`RegimeSplitEnsemble`**,
  **`CountWeightedBlendEnsemble`** (as `mlframe_models` string keys): each requires a task-specific
  constructor argument (`segment_keys`, a required `subpop_label` fit-time positional arg, `regime_fn`,
  `entity_col`) with no generic value the suite can auto-derive for an arbitrary dataset. Registering them
  with a fabricated default would silently produce wrong/meaningless behavior rather than a real win —
  left reachable only via the generic estimator-instance path.
- **`cv_delta_triage` (`triage_cv_delta`)** diagnostic: needs paired per-fold score history accumulated
  across *repeated* CV runs, which the suite doesn't build up at the diagnostics call site. Wiring it would
  mean fabricating fold-score history rather than reusing something the suite already computes.
- **Conformal-surface extension assessment (2026-07-12)**: reviewed three candidate `mlframe.evaluation`/
  `mlframe.calibration` extensions against `ConformalConfig`'s existing seam (`_conformal_on_calib_slice` in
  `_phase_finalize.py`, which only ever reads already-stamped `calib_preds`/`calib_target`/`test_preds`/
  `test_target`/`calib_probs`/`test_probs` arrays per model entry — no feature matrix `X` is available at
  that call site, and it is deliberately additive/read-only, never re-fitting anything). None cleared the
  bar for a safe, generically-applicable default-ON addition:
  - **`CVDeltaHistory`** (`mlframe.evaluation.cv_delta_triage`): same blocker as `cv_delta_triage` above —
    a pooled-variance estimator that only accrues value across *repeated, caller-persisted* suite calls.
    Confirmed: genuinely not single-call wireable; stays external/opt-in (caller instantiates one
    `CVDeltaHistory` per project and passes it into direct `triage_cv_delta` calls themselves).
  - **`cv_score_equivalence_band`'s `n_comparisons` Bonferroni correction** (`mlframe.evaluation.noise_band`):
    a different statistical object than a conformal interval — it widens the "practically equal" band for
    comparing two candidates' *mean CV scores* across a multi-candidate selection search (RFECV/MRMR greedy
    loops), not a per-observation prediction interval/set. The conformal call site has no "how many
    candidates has this search already compared" count to feed it, and conflating the two would either be a
    no-op (`n_comparisons=1`, unchanged) or a fabricated number with no principled value. Confirmed: does not
    fit the conformal surface; the Bonferroni knob already exists and defaults correctly (`n_comparisons=1`)
    for its actual home (model/feature-selection loops), no change needed there.
  - **`imputation_sensitivity_check`'s `shift_split` extension** (`mlframe.evaluation.imputation_sensitivity_check`):
    requires `X_variants` — 2+ already-imputed feature matrices for the SAME rows — to compare fold-score and
    shift-gap stability across imputation choices. The suite resolves exactly ONE `imputer_strategy` per run
    (`PreprocessingBackendConfig.imputer_strategy`, wired into `create_polarsds_pipeline`) and never builds
    alternate imputed copies of `X`, and the conformal call site has no `X` at all (see above). Wiring this
    would mean the suite re-imputing the training data 2+ extra ways and re-fitting/re-scoring a cloned
    estimator per variant purely to feed a diagnostic — real extra compute and a real extra dataset-specific
    input (which imputation choices to compare), not a "pure added output, only reads already-stamped
    arrays" addition like the rest of `ConformalConfig`. Confirmed: does not fit; stays reachable only via a
    direct call with caller-supplied `X_variants`.
  - Net: no code change to `ConformalConfig`/`_conformal_on_calib_slice`. This is a documented negative
    result per the same "left opt-in with a stated reason" pattern as `cv_delta_triage` above, not a gap —
    all three candidates fail on the same root cause (they need data the conformal call site structurally
    doesn't have: cross-call history, a selection-search comparison count, or extra feature matrices), and
    fabricating that data to force a fit would violate the surface's own "additive, read-only" contract.

## Bugs found and fixed while verifying the flip (all confirmed via a live `train_mlframe_models_suite` run,
## not just unit-level mocking)

1. **`GatedOutlierEstimator` + LightGBM early-stopping crash**: the default `configs.LGB_GENERAL_PARAMS`
   bakes in early-stopping params requiring an `eval_set`, but `GatedOutlierEstimator.fit` calls
   `self.regressor_.fit(reg_X, reg_y)` with no `eval_set` (it fits standalone on the non-point-mass training
   rows). Fixed by using the same standalone-safe `LGBMRegressor(n_estimators=200, num_leaves=31,
   verbose=-1, random_state=0)` default the E3 distribution-driven composite base
   (`_estimator_dispatch._default_base_estimator`) already uses.
2. **Point-mass auto-detection couldn't find `train_target`**: the original detection logic looked for the
   train-target array in `common_params["train_target"]` / `config_params["train_target"]` / the bare
   `train_target` kwarg — all three are `None` at the point `configure_training_params` runs in the real
   suite call path (the OD-filtered `train_target` entry gets populated into `common_params` *after* this
   call, not before). Fixed by deriving it directly from the `target` series sliced by `train_idx`, both of
   which ARE real, populated parameters at this call site.
3. **`sorted_mlframe_models` couldn't see dynamically-added model keys**: `ctx.sorted_mlframe_models` (the
   list the per-target model-fitting loop actually iterates) is a suite-level constant computed once at
   setup time, before any target's data exists — it can never contain a model key that
   `configure_training_params` only decides to register per-target (like the auto-detected `gated_outlier`).
   Without a fix, `models_params["gated_outlier"]` would be built and then silently never visited by the fit
   loop (`if _model_entry not in models_params: skip` only prunes the *static* list, it never adds to it).
   Fixed in `_phase_train_one_target_body.py`: after loading `models_params`, extend (not replace)
   `sorted_models` with any `models_params` keys not already present, preserving the original tier-sort
   order for every statically-known model.
4. **`strategy_by_model` `KeyError` on the newly-extended keys**: `strategy_by_model` is looked up by
   `id(_model_entry)` and was built only from the original static `mlframe_models` list (`_phase_config_setup.py`).
   A dynamically-appended model key (from bug #3's fix) has no entry there and raised `KeyError` the first
   time the loop reached it. Fixed alongside #3: compute + register a strategy for each dynamically-added
   key (via the same `get_strategy()` call `_phase_config_setup.py` uses for the static list), on a copy of
   `strategy_by_model` so the suite-level `ctx.strategy_by_model` (shared across pre_pipelines/targets) is
   never mutated by a single target's per-target discovery.

## Flipped to default-ON (2026-07-12, wave 2)

- **`PreprocessingExtensionsConfig.row_wise_summary_stats_enabled` / `row_wise_extreme_columns_enabled`**:
  both `True`. Purely additive per-row aggregates (`mlframe.feature_engineering.row_wise_summary` /
  `row_wise_extremality`) computed over the sklearn-bridge's already-resolved numeric column subset — no
  dataset-specific column names or entity IDs needed, so this is generically safe on any frame. Wired as a
  new step 1.5 in `apply_preprocessing_extensions`, before scaler/kbins/polynomial/dim_reducer so the new
  columns participate in every later step like an ordinary engineered feature. A caller who never touches
  `preprocessing_extensions` still gets these two steps (fixed in `_phase_fit_pipeline.py`: `None` now
  constructs a default `PreprocessingExtensionsConfig()` instead of skipping the whole extensions block).
- **`RegressionCalibrationConfig.apply_confidence_shrinkage`**: `True` (superseding the earlier wave-1 note —
  the wave-1 flip landed in `_reporting_configs.py` but was not actually staged/committed until wave 2).
  Strictly safety-improving: only ever pulls a low-confidence target's predictions toward neutral, never
  degrades a genuinely discriminative one.
- **`TrainingBehaviorConfig.recommend_diversity_additions_in_leaderboard`**: `True`. Runs
  `mlframe.votenrank.correlation_diversity_ablation.recommend_diversity_additions` over the same
  `ens_models` pool `score_ensemble` already blended, via new `_diversity_recommendations.py` adapter wired
  into `_finalize_per_target_ensembling`. Purely observational (never changes which models/ensembles the
  suite selects) — results land in `metadata["diversity_recommendations"][target_type][target_name]`.
  Silently no-ops (not a WARN) when OOF preds are unavailable, since that's a caller config choice.
- **`gated_outlier` strategy registration**: `"gated_outlier": _TREE_STRATEGY` added to `MODEL_STRATEGIES`
  (`training/strategies/__init__.py`) — the wave-1 fix that dynamically extends `sorted_mlframe_models` with
  per-target-discovered keys only works end-to-end once the key also resolves a pipeline strategy; this was
  the missing piece, now covered by `tests/training/test_gated_outlier_registry_key.py`'s live e2e test.
- **`mlframe_models_is_default_allowlist` threading**: `_phase_config_setup.py` now computes this flag
  (`True` only when the caller left top-level `mlframe_models=None`) and threads it through
  `_setup_per_target_mlframe_models` → `select_target` → `configure_training_params`, gating the
  `gated_outlier` point-mass auto-detection so it only ever fires for the implicit default allowlist, never
  silently extending an explicit caller-supplied `mlframe_models=[...]` list.

## Left opt-in (wave 2)

- **`TrainingBehaviorConfig.oof_n_splits` / `oof_has_time` / `oof_random_seed`**: default `0` (no OOF,
  byte-identical to pre-wiring behavior). Was previously unreachable from the public suite entry point at
  all — `train_eval.py`'s trainer already supported these kwargs but no config surface ever set them. NOT
  flipped default-ON: enabling means a real K-fold retrain per model, genuine extra compute a caller must
  opt into (e.g. to unlock `recommend_diversity_additions_in_leaderboard`'s OOF-based diversity signal, or
  `score_ensemble`'s OOF-preferred quality gate).

## Batch D audit: `CompositeTargetDiscoveryConfig` (2026-07-12) — nothing to wire, confirmed already integrated

Investigated per the "wire the isolated 129" directive, specifically the composite-target-discovery config surface.
Verdict: this is **not** an isolated/unreachable utility — it predates and falls outside the current wiring
campaign. Traced the full call graph and confirmed it is already deeply wired end-to-end:

- `CompositeTargetDiscoveryConfig` (`_composite_target_discovery_config.py`, fields carved into
  `_composite_target_discovery_config_base.py`) is accepted as `train_mlframe_models_suite`'s
  `composite_target_discovery_config` parameter (`_main_train_suite.py`), normalised in `_phase_config_setup.py`
  (`_ensure_config(..., CompositeTargetDiscoveryConfig, {})`), and consumed by
  `_phase_composite_discovery.run_composite_target_discovery` — a real phase in the main suite pipeline that
  gates on `composite_target_discovery_config.enabled and TargetTypes.REGRESSION in target_by_type`, feeds
  `_phase_composite_post.py` / `_phase_composite_post_moe.py` / `_phase_composite_wrapping.py` downstream, and is
  exercised by 17+ dedicated test files under `tests/training/composite/discovery/` (unit, biz_value, no-leak,
  parallel, spec-fixes, AR-skip, cache, diagnostic-charts, all-opt-in coverage) plus perf benches under
  `tests/perf/bench_composite_discovery_*`.
- Nearly every one of its ~70 sub-flags already carries its own inline "Default ON per the enable-corrective-
  mechanisms-by-default convention" / "Default OFF, benchmark showed no win" rationale, written in the exact
  style this campaign's changelog entries use (e.g. `detect_base_leakage`, `dedup_x_remaining_for_mi_baseline`,
  `auto_base_structural_boost`, `honest_oof_selection`, `structural_fragility_gate_enabled`, `moe_gate_enabled`,
  `ar1_failsafe_val_crosscheck`, `mi_gain_fdr_control`, all `True`; `region_adaptive_enabled`,
  `use_stacked_discovery(_residual)`, `stacking_aware_gate_enabled`, `ct_ensemble_dedup_enabled` deliberately
  `False` with a named benchmark or dataset-specific blocker each). This work was done in earlier sessions, not
  part of the current isolated-129 list.
- The only knob genuinely left off is the master `enabled: bool = False` switch itself. NOT flipped: unlike a
  pure diagnostic/additive step, composite-target discovery changes what the models are actually trained to
  predict (residual/ratio/log-ratio transforms of `y`) and its value is dataset-specific (needs a real
  dominant-AR-feature / base relationship to pay off) — flipping it globally would be a behavior change with
  real per-target compute cost (MI screening + tiny-model rerank + honest-OOF re-scoring) and no generic
  safety guarantee for datasets with no such structure, so it stays the caller's opt-in decision.

No code change. No new tests needed — this batch is a documented negative result, not a wiring gap.
