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

## Batch H audit: `FeatureHandlingConfig` (2026-07-12) — nothing to wire, confirmed already integrated

Investigated per the "wire the isolated 129" directive, specifically per-feature handling (categorical/text
transforms, embedding providers, caching). Verdict: `FeatureHandlingConfig` (`training/feature_handling/config.py`,
the full `mlframe.training.feature_handling` package with 25+ sibling modules — axis registry, handlers, assembler,
cache/cache_backend, fingerprint, text_detection, target_encoders, routing, presets) is **not** an isolated/
unreachable utility — it is already deeply wired end-to-end into `train_mlframe_models_suite`:

- Accepted as the public `feature_handling_config` parameter (`_main_train_suite.py`), normalised/validated in
  `_phase_config_setup.py` (`FeatureHandlingConfig.validate_against_models(mlframe_models)` against the compat
  matrix, then stashed onto `ctx.artifacts["feature_handling_config"]` — `TrainingContext` is `slots=True` with
  no dedicated slot, so `ctx.artifacts` is the documented storage seam).
- Consumed once per target via `_maybe_run_feature_handling_apply` (`_phase_train_one_target_helpers.py`), called
  from `_phase_train_one_target_model_setup.py` at the "post-FS / pre-final-pipeline" seam, which resolves
  `ctx.feature_handling_config` / `ctx.artifacts["feature_handling_config"]`, calls `feature_handling_apply`
  (auto text-detection, per-model handler-chain resolution, fit/transform/assemble, `FeatureCache` reuse across
  the suite's models×pre_pipelines loops), and stashes the fitted result under
  `ctx.artifacts["feature_handling_fitted"][target_name]`.
- Already covered by dedicated tests exercising the real wiring, not just the standalone helper:
  `tests/training/test_feature_handling_config_plumbed_onto_ctx.py`,
  `tests/training/test_feature_handling_apply_wired_into_train_one_target.py`,
  `tests/training/test_feature_handling_apply.py`, plus fuzz-combo coverage
  (`tests/training/_fuzz_combo/axes.py`, `tests/training/fuzz/test_fuzz_combo_cross_axis.py`).

**Not flipped to default-ON** (verified, not assumed): `FeatureHandlingConfig()` constructed with zero args
resolves `default_cat=None` / `default_text=None` to **empty** per-model handler chains for every model kind —
confirmed live: `fhc = FeatureHandlingConfig(); fhc._effective_text_specs("cb") == [] and
fhc._effective_cat_specs("cb") == []`. Unlike the diagnostics/row-wise-stats flips (pure additive, no
dataset-specific input needed), a zero-config FHC is a genuine **no-op** that would only add cache/memory-probe
setup overhead for zero behavioral effect, while a FHC seeded with a *non-empty* fabricated default (e.g. always
TF-IDF-encode auto-detected text columns) would be a real, dataset-specific feature-encoding-policy decision the
suite cannot safely invent on the caller's behalf — the same "no generic default exists" reasoning already
recorded above for `SegmentedModelFactory`/`GatedRegressionMixture`/etc and for composite-target-discovery's master
switch. Confirmed via live probe, no source change needed; the existing `feature_handling_config: Optional[...] =
None` opt-in default is correct as-is.

No code change. No new tests needed — this batch is a documented negative result, not a wiring gap.

## Batch E: `mlframe_models` registry entries — bagging / composite_classification (2026-07-12)

Audited the composite/ensemble estimator zoo (`src/mlframe/training/composite/`) for generically-instantiable
estimators not yet reachable as an `mlframe_models` string key. Confirmed prior audits' verdicts still hold
(`SegmentedModelFactory`, `GatedRegressionMixture`, `RegimeSplitEnsemble`, `CountWeightedBlendEnsemble` all
require a task-specific constructor arg — left opt-in, reachable only via the generic estimator-instance path).

Registered two genuinely dataset-agnostic composites, explicit-allowlist-only (no auto-detection heuristic,
unlike `gated_outlier`'s point-mass trigger — see rationale inline at each registration site in
`_trainer_configure.py`):

- **`"bagging"` (`BaggedCompositeEstimator`)**: bootstrap-bagged variance reduction over a plain GBDT regressor.
  No required dataset-specific arg. No auto-detection: bagging (`n_estimators=10` full refits by default) is a
  real Nx compute multiplier with no generic trigger signal analogous to "target has a point mass" — blanket
  inclusion would silently 10x every caller's regression fit time.
- **`"composite_classification"` (`CompositeClassificationEstimator`)**: base-margin (init-score) residual
  composite for classification; auto-fits its own `LogisticRegression` base margin when no
  `base_margin_column` is supplied, so it needs no caller-supplied column either. No auto-detection: every
  classification target benefits from *some* base margin, so there is no principled "only some targets need
  this" heuristic to gate a default-ON auto-append on.

Both registered in `MODEL_STRATEGIES` (`training/strategies/__init__.py`) as `_TREE_STRATEGY` (inner estimator
is the suite's standalone-safe default LGBM, matching `gated_outlier`'s registered strategy).

### Bugs found and fixed while verifying live (both confirmed via a real `train_mlframe_models_suite` run)

1. **Dichotomic early-stopping wrapper misread `BaggedCompositeEstimator.n_estimators` as a boosting-round
   budget**: `maybe_wrap_for_partial_fit_es`'s generic `get_params()` probe (`_data_helpers.py`) picks up any
   top-level int named `max_iter`/`n_estimators`/`max_trials` as a dichotomic-searchable ES budget knob.
   `BaggedCompositeEstimator.n_estimators` is the bag COUNT, not boosting rounds — each dichotomic-search
   candidate re-fit the WHOLE bagged ensemble (all N member refits), multiplying fit cost by the search's
   candidate count on top of the composite's own internal per-member cost. Symptom: a single `"bagging"` suite
   fit hung past a 900s `pytest-timeout` under moderate load. Fixed by adding `"bagging"`/
   `"composite_classification"` to `_BUDGET_PARAM_BY_CATEGORY` mapped to `None` (no usable budget knob — each
   composite already manages its own iteration budget internally).
2. **Pre-existing sentinel bug in `_detect_budget_param` masked bug #1's first fix attempt**: `explicit =
   _BUDGET_PARAM_BY_CATEGORY.get(model_category); if explicit is not None: return explicit` cannot distinguish
   "key explicitly mapped to `None`" from "key absent" — both look like `None`, so the just-added
   `"bagging": None` entry silently fell through to the runtime probe anyway, reproducing the exact same hang.
   This is a LATENT bug that predates this batch: the existing `"linear": None` entry never triggered it only
   because `LinearRegression`/`LogisticRegression` are closed-form and happen to expose no probeable int
   param — the masking was coincidental, not by design. Fixed with an explicit `model_category in
   _BUDGET_PARAM_BY_CATEGORY` membership check before falling back to the runtime probe.
3. **Stale test assumption about the trained feature count**: `test_e2e_bagging_key_trains_and_predicts` /
   `test_e2e_composite_classification_key_trains_and_predicts` called `fitted.predict(...)` on a hardcoded
   3-raw-column (`f0`,`f1`,`f2`) zero frame, but the wave-2 default-ON `row_wise_summary_stats_enabled` /
   `row_wise_extreme_columns_enabled` preprocessing extensions append engineered columns before the model
   sees the frame (3 raw → 11 trained columns in the test fixture) — a real `LightGBMError: number of features
   ... not the same as it was in training data`. Per the "validated improvement breaks a test → re-frame the
   test" rule: fixed by predicting with the fitted estimator's own `feature_names_in_`/`n_features_in_`
   instead of a hardcoded raw column list — the test's actual intent (predict works post-fit through the full
   suite) is unaffected by exactly how many engineered columns preprocessing produced.

Verified via 6 live end-to-end `train_mlframe_models_suite` tests (`tests/training/test_batch_e_registry_keys.py`),
all passing after both fixes.

## A–I completeness check (2026-07-13)

All batches from the original wiring plan now have a documented entry in this file:
- **A** (FeatureSelectionConfig selectors) — wired + flipped default-ON, wave 1.
- **B** (PreprocessingExtensionsConfig FE) — wired + flipped default-ON, wave 2.
- **C** (OutputConfig.run_diagnostics) — wired + flipped default-ON, wave 1.
- **D** (composite_target_discovery_config) — audited, already fully wired; master switch correctly left opt-in.
- **E** (mlframe_models registry entries) — wired (`bagging`/`composite_classification`), explicit-allowlist-only.
- **F** (RegressionCalibrationConfig) — wired + flipped default-ON, wave 1/2.
- **G** (ConformalConfig) — audited, documented negative result (no safe generic fit found for 3 candidates).
- **H** (FeatureHandlingConfig) — audited, already fully wired; zero-config default correctly left as a no-op.
- **I** (TrainingBehaviorConfig / votenrank diversity) — wired + flipped default-ON, wave 2.

No remaining unstarted batch from the original A–I plan.

## Fuzz-combo axes updated for the default-flip effort (2026-07-13)

Closed the second half of an earlier explicit user directive ("verify the new functionality is
wired AND update the axes/configs/parameters of fuzz combo tests") — the axes update had not
been done. Added 12 new axes to `tests/training/_fuzz_combo/axes.py` covering every capability
flipped or newly wired across waves 1-3 (batches A/B/C/E/F/I) that previously had zero fuzz
exposure: `extra_registry_model_cfg` (batch E composite keys), `run_diagnostics_cfg` (batch C),
`fs_new_selectors_enabled_cfg` (batch A opt-out path), `auto_optimize_threshold_cfg` /
`check_isotonic_overfit_risk_cfg` / `recommend_diversity_additions_in_leaderboard_cfg` /
`oof_n_splits_cfg` (waves 1/2 TrainingBehaviorConfig), `apply_confidence_shrinkage_cfg` (batch F,
never threaded into the fuzz suite call at all before this), `row_wise_summary_stats_enabled_cfg`
/ `row_wise_extreme_columns_enabled_cfg` (batch B), and `inject_point_mass_cfg` +
`mlframe_models_explicit_cfg` (the `gated_outlier` point-mass auto-detect path, which needs the
implicit `mlframe_models=None` default on a regression target with a real point mass — the fuzz
suite always passed an explicit allowlist before, so this path had zero exposure regardless of
data shape). Each canonicalises away when incompatible with the combo's target_type/model set.

Verification: static (all 12 axes take every declared value across the 150 enumerated combos;
the 3 new canonicalization methods ran error-free on a 40-combo sample) plus one live end-to-end
`--run-fuzz` single-combo pass exercising `gated_outlier`. Full `--run-fuzz` batch execution was
attempted twice and blocked both times by issues confirmed unrelated to this change: (1) a
pre-existing MRMR stability-selection RandomForest/joblib fit exceeding a 1200s per-test timeout
under this session's sustained heavy concurrent-process load, and (2) a native access-violation
crash inside `reporting/charts/class_structure_heatmap.py`'s `@numba.njit(cache=True)` accumulate
kernel — traced the array-index generation (`_equal_population_codes`, `_group_codes_capped`)
and confirmed both are bounds-safe by construction (codes are always clipped/remapped into valid
ranges), so this is most likely a numba on-disk cache-file race from many concurrent processes
JIT-compiling the same kernel simultaneously, not a logic bug — and it sits entirely outside the
files this change touches. Flagged as a follow-up investigation (rerun `--run-fuzz` on a quiet
machine, or disable `cache=True` on that kernel as a mitigation), not chased further here since
it's out of this change's scope and wasn't reproducible with enough confidence to safely patch.

**Follow-up (2026-07-13):** Reproduced `class_structure_matrix` standalone (5 trials, random
group/time codes, low system contention) — no crash, correct output every time, confirming the
kernel itself is correct. `cache=True` appears on 113 other `@numba.njit` sites across the
codebase, so speculatively stripping it here (or anywhere) without stronger evidence would be an
unproven fix with real blast radius if generalized incorrectly. Verdict: genuine environmental
artifact under this session's sustained heavy concurrent-process load, not a code defect — closing
this follow-up as verified-correct-under-load-only, no code change.

## Perf fix: LightGBM n_jobs pinned to avoid Windows physical-core-detection subprocess (2026-07-13)

Profiling `BaggedCompositeEstimator.fit()` (batch E) found `joblib`'s
`loky.backend.context._count_physical_cores_win32` consuming 59% of total fit time (5.1 of 8.6s).
Root cause: leaving `n_jobs` unset on a LightGBM estimator resolves via
`joblib.cpu_count(only_physical_cores=True)`, which shells out to a subprocess on Windows —
measured ~2s on a quiet box, 5s+ under this session's concurrent-process load, paid once per
process on the first LightGBM fit. Fixed by pinning `n_jobs=-1` (routes through
`joblib.cpu_count(only_physical_cores=False)`, i.e. plain `os.cpu_count()`, <1ms, same "use all
cores" intent) at all four sites mlframe constructs a default LightGBM estimator: the suite's own
`LGB_GENERAL_PARAMS` (affects every ordinary `"lgb"` fit, not just this session's new composites),
the three standalone-safe defaults for `gated_outlier`/`bagging`/`composite_classification`, and
the shared E3-composite base (`_estimator_dispatch._default_base_estimator`). Bit-identical output
verified (n_jobs only affects thread count, not the deterministic histogram-based fit result).
Measured ~2.3x wall-time reduction on `BaggedCompositeEstimator.fit()+predict()` (8.85s → 3.77s,
20,000-row/15-column synthetic). This is very likely a contributing factor to several of this
session's earlier "mysteriously slow" single-combo fuzz/test runs, since it's a real, previously-
unaccounted-for per-process cold-start tax that gets paid repeatedly across many short-lived
pytest worker processes.

While verifying, also fixed two unrelated pre-existing test bugs surfaced by the full regression
run (both now committed alongside): `test_gated_outlier_registry_key.py` had the same stale
hardcoded-3-raw-column predict pattern already fixed in `test_batch_e_registry_keys.py` (wave-2's
default-ON row-wise preprocessing extensions changed the trained feature count); and the
`getattr(fitted, "feature_names_in_", None) or range(...)` fallback used to fix that is itself
unsafe (a numpy array with >1 elements raises under Python's `or` truthiness) — fixed with an
explicit `is not None` check at all three occurrences across both files, not just the one that
happened to crash (the other two "worked" only by accident, since those composites don't set
`feature_names_in_` on themselves and always fell through to `None`).
