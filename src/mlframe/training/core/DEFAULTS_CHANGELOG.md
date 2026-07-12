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
