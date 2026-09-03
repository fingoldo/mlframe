# training_core

Files reviewed: 39 | LOC: 16,558 (of ~120 files in the cluster; the rest covered by targeted AST/grep sweeps)

## Summary

The cluster is disciplined about the failure modes it has already been burned by: polars `pl.Enum` over
`Categorical`, no whole-frame copies on any hot path, `orjson`/`OPT_SORT_KEYS` on every cross-process cache key,
the thread-flag restore now reachable from a real `finally` at the suite boundary, and val/test/OOF honesty
encoded in `_ensemble_chooser` and `_honest_decision_threshold`.

The defects that remain are almost all one shape: **a read that silently resolves to a default and therefore
no-ops.** `TrainingContext` is `slots=True`, so `getattr(ctx, "<not-a-slot>", None)` returns None with no error --
three defensive mechanisms in the default-ON unsupervised pre-screen, and the composite y-scale chart path, are
dead for exactly this reason. The same shape appears at config level.

## Findings

### TRAINING_CORE-1 [P1] silent-fallback / broad-except

**File:** `src/mlframe/training/_gpu_probe.py` :14-23

**Summary:** A bare `except Exception` around the module-level numba/CUDA probe caches `CUDA_IS_AVAILABLE = False`
for the entire process lifetime on any transient failure, logged only at `debug`.

**Failure scenario:** A transient device/driver fault while importing `numba.cuda` or running
`is_cuda_available()` -- a driver reset, a contended GPU, a `CudaSupportError`/`OSError` from libcuda -- raises
something other than `ImportError`. `CUDA_IS_AVAILABLE` is bound to False at import and never re-probed.
`_helpers_training_configs.py` :118-119 then resolves `has_gpu = CUDA_IS_AVAILABLE` for every suite call, so
CatBoost `task_type`, `_xgb_device` (:244) and `_lgb_device` (:381) all resolve to CPU for the rest of the
process, with nothing above `debug` saying why. This is byte-for-byte the class documented for
`_select_mi_backend`.

**Suggested fix:** Narrow the outer except to `ImportError` (the genuine "numba absent" case). Any other exception
should `logger.warning` naming the type and still leave `CUDA_IS_AVAILABLE` optimistic, letting the per-library
probes (`_probe_xgb_gpu_support`, which independently checks `build_info()['USE_CUDA']`) make the real decision.
Same treatment for the inner except at :18.

**Evidence:** Read in full; traced `CUDA_IS_AVAILABLE` to its four consumers and confirmed :118-119 is the sole
default resolution for the whole suite.

**Disposition:** RESOLVED in the P0 pass. `_gpu_probe.py` narrows the outer handler to `ImportError`; anything else logs at warning and leaves `CUDA_IS_AVAILABLE` optimistic so the per-library probes decide.

### TRAINING_CORE-2 [P1] dead-knob / optimistic-uncertainty

**File:** `src/mlframe/training/_preprocessing_configs.py` :132, :206-213; `src/mlframe/training/_conformal_split.py` :1-167

**Summary:** `TrainingSplitConfig.conformal_size` is accepted, range-validated and included in the split-budget
sum check, but no code in `src/` ever reads it; the structure-aware carver written to consume it has zero
production call sites.

**Failure scenario:** A user sets `conformal_size=0.05` to get the documented behaviour -- "`calib_size` fits the
recalibration map g, `conformal_size` scores g(model) so the interval reflects what ships (sharing one slice
makes residuals in-sample for g -> optimistic coverage)". No conformal slice is ever carved; finalize silently
takes the fallback of reusing the calib slice, i.e. exactly the optimistic-coverage regime the field exists to
avoid. Reported intervals are narrower than the truth and nothing warns.

**Suggested fix:** Either wire `conformal_size` into `splitting.py`'s carve path, or -- if the feature is
deferred -- raise or warn at config validation when `conformal_size` is non-zero, and mark the field and
`_conformal_split.py` as not-yet-wired.

**Evidence:** `grep -rn "conformal_size|conformal_idx|conformal_df|conformal_frac" src/` returns only the field
declaration, the sum validator, one docstring mention, and `_conformal_split.py`'s own internals. The four public
carvers are imported only by `tests/training/conformal/test_conformal_split_carving.py`; the production path is
`splitting.py` :831 -> `_split_helpers._carve_calib_from_train` (calib only, no conformal slice, no
purge/embargo).

**Disposition:** RESOLVED as fail-closed, wiring deferred. `TrainingSplitConfig` now REFUSES a non-zero `conformal_size` with a message naming what would otherwise happen; the field and `_conformal_split.py` are marked not-yet-wired, and `_regression_calibration.py`'s docstring no longer promises the separate slice it does not get. Carving a slice with no consumer would only shrink train for nothing, so the honest resolution is to stop the setting reading as configured. `tests/training/test_conformal_size_is_not_silently_ignored.py` includes an AST check that fails the moment someone gives the field a real consumer, pointing them at the refusal to lift.

### TRAINING_CORE-3 [P1] stale-cache / wrong-weights

**File:** `src/mlframe/training/cb/_cb_pool_build.py` :234-235, reached from
`src/mlframe/training/core/_phase_train_one_target_body.py` :534-548

**Summary:** On a CatBoost Pool cache hit the weight is re-applied only when `sample_weight is not None`; a
subsequent uniform-weight fit on the same feature frame silently trains against the previous schema's
non-uniform weights.

**Failure scenario:** The extractor supplies `sample_weights = {"recency": w, "uniform": None}`. The loop
iterates in insertion order, so "recency" fits first and builds the Pool with `weight=w`. The "uniform" iteration
sets `sample_weight = None`; the cache signature contains no weight component, so the cached Pool is returned.
:234 is false, `set_weight` is never called, and the Pool still carries `w`. The "uniform" model is trained
recency-weighted, its metrics are compared against the recency model as if they were different schemas, and the
leaderboard / ensemble pick is made on a duplicate.

**Suggested fix:** Add the `else` branch the LGB and XGB shims already have: on a cache hit with
`sample_weight is None`, call `cached.set_weight(np.ones(cached.num_row(), dtype=np.float32))`. Verify CatBoost
has no all-ones short-circuit -- LightGBM does, which is why `_lgb_shim_helpers._reset_weight_to_uniform`
bypasses `set_weight` via `set_field`.

**Evidence:** :201-261 (hit path) versus :297-305 (build path with `weight=sample_weight`). Both sibling caches
handle this and document why: `lgb_shim.py` :378-386 ("plain `set_weight(ones)` silently NO-OPS when a real prior
weight is already set at the C++ side") and `xgb_shim.py` :523-528. Reuse is capability-gated only, no config
opt-out. Note: `cb/` is one directory outside the stated cluster path; reported because the triggering loop is
in-cluster and the two in-cluster shims prove the intended contract.

**Disposition:** RESOLVED. The weight is re-applied unconditionally on a cache hit, resetting to uniform when the fit asks for no weights; the built Pool records its own weight state. Tested against a fake Pool because this CatBoost build has no `Pool.set_label`, so the reuse path cannot activate on this machine and a skip would have proved nothing. `tests/training/test_cb_pool_reuse_resets_weights.py`.

### TRAINING_CORE-4 [P2] dead-guard behind slots=True

**File:** `src/mlframe/training/core/_phase_train_one_target_pre_screen.py` :65-81

**Summary:** All three "defensive double-source" mechanisms that protect the group-ID and timestamp columns from
the default-ON unsupervised pre-screen read attribute names that exist on none of the objects they probe, so the
protected set never contains a group or ts column.

**Failure scenario:** `FeatureSelectionConfig.pre_screen_unsupervised` defaults to True, so this runs on every
stock suite. :66 probes `ctx.group_id_col` / `ctx.ts_field` -- neither is a `TrainingContext` slot, and because
the dataclass is `slots=True` the `getattr(..., None)` returns None silently rather than raising. :70 probes
`ctx.extractor` / `ctx.features_and_targets_extractor` -- also not slots, so the whole FTE block at :71-75 never
executes. :78-81 probes `split_config.group_field` / `timestamps_column` / `ts_column`; `TrainingSplitConfig`
declares none of these (its real field is `time_column`). Net: `_protected` only ever holds target names plus
cat/text/embedding features, so a group or time column that is constant (single-group frame, single-date
snapshot) or >99% null is dropped from every train/val/test mirror. The safety hatch at :83-95
(`if _split_cfg_use_groups and not _protected: skip`) can never fire either, because `_protected` is non-empty by
:53 on every real suite.

**Suggested fix:** Probe the names that exist: `split_config.time_column`; add an `extractor` (or the resolved
group/ts column names) to `TrainingContext` as a real slot and populate it in `setup_configuration`; drop the
`ctx.group_id_col`/`ctx.ts_field` probe or add those slots. Add a meta-test asserting every
`getattr(ctx, "<name>", ...)` literal in `core/` is a declared `TrainingContext` field -- the same AST sweep that
found this.

**Evidence:** Enumerated `TrainingContext`'s 124 annotated slots and cross-checked every `getattr(ctx, "...")`
literal in `src/mlframe/training/`: the four names are absent. Enumerated `TrainingSplitConfig`'s 18 fields:
`group_field`, `timestamps_column`, `ts_column` are absent; `use_groups` and `time_column` are present.

**Disposition:** RESOLVED, though not by the fix the finding implies. Confirmed the premise: none of `group_id_col`, `ts_field`, `extractor`, `features_and_targets_extractor` is a `TrainingContext` slot, and `TrainingSplitConfig` declares none of `group_field` / `timestamps_column` / `ts_column`. But there is no corrected NAME to substitute either -- the context addresses these as ARRAYS (`group_ids_raw`, `group_ids`, `timestamps`), never by column name, so the intended two-source resolution has no second source to resolve against. The three dead blocks are replaced by the one real source available at this point: a pandas Series' own `.name` on those three slots. The `use_groups and not _protected` skip remains the backstop for everything else. `tests/training/test_dead_slot_reads_and_stale_contracts.py`.

### TRAINING_CORE-5 [P2] dead-read / lost diagnostic output

**File:** `src/mlframe/training/core/_phase_train_one_target_post.py` :247

**Summary:** `plot_file=getattr(ctx, "plot_file", None)` reads a non-existent slot, so the per-model composite
y-scale TEST chart is never written to disk.

**Failure scenario:** On any composite-target run the call receives `plot_file=None` unconditionally. Inside
`_phase_composite_wrapping.py` :115-122, `if plot_file:` is false -> `_plot_path = ""` ->
`report_model_perf(..., plot_file="")` produces no saved chart. The metric log line still prints, so the run
looks healthy; the operator never gets the chart the block exists to emit, and nothing warns. The real value
lives at `ctx.output_config.plot_file`.

**Suggested fix:** `plot_file=getattr(getattr(ctx, "output_config", None), "plot_file", None)`.

**Evidence:** Same slot enumeration as TRAINING_CORE-4; `_phase_composite_wrapping.py` :115-134 shows
`plot_file` is the sole source of `_plot_path` and the empty-string branch.

**Disposition:** RESOLVED exactly as suggested: `getattr(getattr(ctx, "output_config", None), "plot_file", None)`. `tests/training/test_dead_slot_reads_and_stale_contracts.py` pins that `plot_file` is not a context slot while `output_config` is, so the test fails if the slot layout ever makes the old read look valid again.

### TRAINING_CORE-6 [P2] leakage-adjacent contract drift

**File:** `src/mlframe/training/core/_ar1_failsafe_veto.py` :1-13, :61-64;
`src/mlframe/training/_composite_target_discovery_config_base.py` :89

**Summary:** A default-ON deployment decision is made on the VAL split, which both the module docstring and the
config comment describe as "the SAME honest-holdout regime as test" -- but val is the early-stopping split, and
its bias points in the same direction as the decision the veto makes.

**Failure scenario:** `ar1_failsafe_val_crosscheck` defaults True. `decide_ar1_failsafe_val_veto` compares
`lag_predict` (zero-parameter, no early stopping, so its val RMSE is unbiased) against the best trained component
(early-stopped ON val, so its val RMSE is optimistically biased low). The veto fires at :63 when
`bt_val < lp_val / (1 + tol)` and replaces the deployed lag baseline with the trained model. The ES optimism
inflates exactly the quantity that triggers the swap, so the veto fires more often than an honest comparison
would justify -- the 10% tolerance is the only thing absorbing it, and nothing measures how much ES optimism is
actually present. :62's own comment acknowledges "val is used for ES" while the module header claims the
opposite.

**Suggested fix:** Either move the cross-check to the honest test/OOS slice with the selection cost documented
and a one-time WARN (the pattern `_ensemble_chooser._choose_ensemble_flavour` already uses at :164-175), or keep
val but fix both docstrings to state that val is ES-biased in favour of the trained component, and size the
tolerance from a measured train-versus-val ES gap rather than reusing `lag_predict_failsafe_tolerance`.

**Evidence:** Read in full; default at `_composite_target_discovery_config_base.py` :92, tolerance at
`_composite_target_discovery_config.py` :524, call site at
`core/_phase_composite_post_xt_ensemble/__init__.py` :913-919.

**Disposition:** RESOLVED as a documentation-and-observability fix; the split itself is unchanged, and deliberately so. The finding is right that val is the early-stopping split and that its bias points the same way as the decision -- the module docstring's claim that val is "the SAME honest-holdout regime as test" is simply false, and the sibling config comment repeated it. Both now state plainly that val is group-disjoint but ES-biased, that lag_predict is zero-parameter and therefore unbiased, that the tolerance is the only thing absorbing the difference, and that the residual error direction is the mild one. Moving the decision off val is not available: there is no fourth split, and using test would leak the honest estimate into a deployment decision. The finding's other half -- "nothing measures how much ES optimism is actually present" -- is closed by logging both val RMSEs, the veto threshold, the tolerance and the outcome, so the realised headroom is observable per run rather than assumed. `tests/training/test_dead_slot_reads_and_stale_contracts.py`.

### TRAINING_CORE-7 [P3] undeclared config knob

**File:** `src/mlframe/training/core/_phase_temporal_audit.py` :120, :132

**Summary:** The code reads and documents `behavior_config.target_temporal_audit_unit`, but the field is declared
nowhere; setting it works only via pydantic's `extra="allow"` escape hatch and triggers the "unknown extra" typo
warning.

**Failure scenario:** A user with epoch-second timestamps follows the in-code comment and passes
`TrainingBehaviorConfig(target_temporal_audit_unit="s")`. `BaseConfig` allows the extra, so the value does reach
:120 -- but `_warn_on_unknown_extras` logs a WARNING telling them it looks like a typo, and the knob is invisible
to anyone reading the config class alongside its three declared siblings.

**Suggested fix:** Declare `target_temporal_audit_unit: Optional[str] = None` on `TrainingBehaviorConfig` next to
the other three, or add it to that class's `_known_extras`.

**Evidence:** AST sweep of every `getattr(<x>_config|<x>_cfg, "<name>")` literal in `training/` and
`training/core/` against every class field, function parameter and attribute access in `src/mlframe` -- this is
the only name with no definition anywhere.

**Disposition:** RESOLVED as suggested -- `target_temporal_audit_unit: Optional[str] = None` is declared on `TrainingBehaviorConfig` next to its three siblings, with a docstring naming the accepted units and the auto-detect default. `tests/training/test_dead_slot_reads_and_stale_contracts.py`.

### TRAINING_CORE-8 [P3] contradictory in-block comments

**File:** `src/mlframe/training/core/_phase_helpers_fit_split.py` :707 versus :716-721

**Summary:** Two comments ten lines apart state opposite rules for the Enum domain; the leading one is stale.

**Failure scenario:** :707 says "keyed off the train-only unique set; val/test cast non-strict so OOV becomes
null". The code at :726-736 builds `set(_u_train) | set(_u_val)`, and :785-788 cast train AND val with
`strict=True`. A future edit trusting :707 -- dropping the val union, or switching val to `strict=False` --
would silently reintroduce the exact ES bias :716-721 exists to prevent.

**Suggested fix:** Delete the stale sentence at :707.

**Evidence:** Read :700-795 in full.

**Disposition:** RESOLVED. The stale sentence is deleted; the remaining lead comment now points at the cast site, which is where the domain and the strictness are actually decided, so the two cannot drift apart again. `tests/training/test_dead_slot_reads_and_stale_contracts.py`.

### TRAINING_CORE-9 [P3] docstring promises behaviour the body does not implement

**File:** `src/mlframe/training/_precompute.py` :266-272 versus :284-289

**Summary:** `precompute_all`'s Args section says `target_by_type` / `dummy_baselines_config` / `composite_config`
are "forwarded to the dummy stub" / "forwarded to the composite stub"; the body never calls either stub and the
three parameters are unread.

**Failure scenario:** A caller passes `dummy_baselines_config=cfg` and assumes the dummy-baseline slot will be
populated, or at least attempted. It is silently left at None and the suite recomputes inline.

**Suggested fix:** Rewrite those three Args entries to "accepted for signature stability; not consumed (see the
stubs' NotImplementedError)", matching the honest comment already at :284-287.

**Evidence:** Read :150-290; the AST sweep for never-read parameters flagged all three.

**Disposition:** RESOLVED as suggested -- the Args entries say the parameters are accepted for signature stability and NOT consumed, since both stubs raise `NotImplementedError`. `tests/training/test_dead_slot_reads_and_stale_contracts.py`.

### TRAINING_CORE-10 [P3] unused parameter at a live call site

**File:** `src/mlframe/training/_eval_helpers.py` :571-573, called at :552-567

**Summary:** `_render_split_diagnostics` accepts `split_name` and the caller passes it, but the body never reads
it -- the emitted error-analysis / drift panels carry no split label except whatever `split_plot_file` encodes.

**Failure scenario:** Given the repo's own rule that val, test and OOF mean different things and must be named
after what they are, a reader of the generated worst-K table or residual-vs-time panels has no in-artifact
indication of which split produced them; `metrics_dict["worst_k_table"]` is written under an unqualified key.

**Suggested fix:** Thread `split_name` into the chart titles and the `worst_k_table` key, or drop the parameter
from the signature and the call.

**Evidence:** Read :552-683; `split_name` appears only in the signature.

**Disposition:** RESOLVED by threading rather than dropping. `split_name` now qualifies the artifact key (`worst_k_table_<split>`), with the unqualified key kept as an alias so existing readers are unaffected. The repo's own val/test/OOF naming rule is the reason to keep the parameter rather than delete it: an unqualified table leaves a reader unable to tell an optimistically-biased val figure from an honest test one. `tests/training/test_dead_slot_reads_and_stale_contracts.py`.

### TRAINING_CORE-11 [P3] dead knob exercised as a fuzz axis

**File:** `src/mlframe/training/_model_configs_ensembling.py` :20

**Summary:** `EnsemblingConfig.force_legacy` has no reader anywhere in `src/`, yet the fuzz harness varies it as a
combo axis, so that axis is a guaranteed no-op that silently halves the effective coverage of the combos it
appears in.

**Failure scenario:** `tests/training/_fuzz_combo/axes.py` :953 declares `"ensembling_force_legacy_cfg":
(False, True)` and `combo.py` :1446 feeds it into the config. Because nothing reads the field, both arms produce
identical runs -- the fuzz suite reports coverage of a path it never exercises.

**Suggested fix:** Either wire `force_legacy` to the legacy ensembling path it names, or remove the field and the
fuzz axis together. It is already allowlisted in `tests/test_meta/test_config_field_consumption.py` :73, so the
meta test will not flag the removal.

**Evidence:** Whole-`src` identifier frequency count over 2,445 files: `force_legacy` occurs exactly once, its own
declaration. A follow-up `refs <= 1` sweep found only two other fully-dead config fields
(`CompositeTargetDiscoveryConfigBase.force_inject_diff_on_top_ablation_pct` and
`.structural_fragility_max_amplification_ratio`), both already documented as deliberate in
`docs/composite_config_reference.md` and allowlisted -- tracked dead knobs rather than silent ones, so not
findings.

**Disposition:** RESOLVED by removal, per the finding's own second option -- the field, its fuzz axis, the combo dataclass field, the canonical-key entry, the meta-test allowlist entry and the stale defaults assertion all go together. One correction to the evidence: the finding says `combo.py` :1446 "feeds it into the config", but that line is inside `canonical_key()`'s dedup tuple, not a config construction; the fuzz harness never built an `EnsemblingConfig` with it at all, so the axis was doubly dead. `BaseConfig` allows extras, so a caller still passing `force_legacy` gets a warning rather than a failure. `tests/training/test_dead_slot_reads_and_stale_contracts.py`.

### TRAINING_CORE-12 [P3] docstring versus implementation

**File:** `src/mlframe/training/_overlapping_walk_forward_cv.py` :171-172 versus :202-209

**Summary:** `cv_stability_check`'s docstring describes the jaggedness statistic as "a curve's SECOND-difference
sign-change count, divided by its LENGTH"; the code counts sign changes of the FIRST difference and divides by
the count of NON-ZERO first differences.

**Failure scenario:** A caller tuning `max_sign_change_ratio` against the documented denominator sets a threshold
that is systematically too permissive on any curve with flat segments, where `len(nonzero) < n_points` inflates
the computed ratio. The `stable` verdict at :217 then flips relative to what the caller intended.

**Suggested fix:** Correct the docstring to "sign changes of the first difference, divided by the number of
non-zero first differences".

**Evidence:** Read in full; :202-209 compute `np.diff(curve)` once, filter `signs != 0`, and divide
`sign_changes / len(nonzero)`.

**Disposition:** RESOLVED as suggested, and the docstring now also says WHY the denominator matters -- on a curve with flat segments it is smaller than the length, so a threshold set against the documented form is systematically too permissive. `tests/training/test_dead_slot_reads_and_stale_contracts.py`.

### TRAINING_CORE-13 [P3] unreachable fallback branch

**File:** `src/mlframe/training/core/_phase_finalize.py` :371, :423, :478;
`src/mlframe/training/core/_phase_finalize_calibration.py` :77, :130, :246, :273, :360

**Summary:** Eight `getattr(ctx, "configs", None)` fallbacks read a name that is not a `TrainingContext` slot, so
each `_root`/`_configs_root` is always None and the fallback can never supply a config.

**Failure scenario:** Harmless today -- each site is guarded by `if _cfg is None:` and the real slots are always
populated. But the code reads as a working two-source resolution, so a future refactor that legitimately leaves
`ctx.reporting_config` unset would silently skip the whole comparison-chart / isotonic-risk /
threshold-optimisation block instead of falling back.

**Suggested fix:** Delete the eight fallback branches, or add a real `configs` aggregate slot if one is intended.

**Evidence:** Slot enumeration plus reading each of the eight sites.

**Disposition:** RESOLVED by deleting all eight fallback branches rather than adding a `configs` aggregate slot: nothing else in the tree wants one, and the branches' only effect was to make an unreachable path read as a working two-source resolution. `tests/training/test_dead_slot_reads_and_stale_contracts.py`.

### TRAINING_CORE-14 [P3] purge double-charged on an empty carve

**File:** `src/mlframe/training/_conformal_split.py` :84-95

**Summary:** `carve_calib_conformal_temporal` subtracts `purge` twice even when both `calib_frac` and
`conformal_frac` resolve to zero, silently discarding `2 * purge` train rows for no boundary.

**Failure scenario:** `carve_calib_conformal_temporal(train_idx, 0.0, 0.0, purge=100)` returns a fit slice
missing the 200 most recent rows with empty calib and conformal, and no error. Latent: currently reachable only
from tests (see TRAINING_CORE-2), but it activates the moment the module is wired.

**Suggested fix:** Apply each purge conditionally: `calib_stop = conf_start - (purge if n_conf else 0)`;
`fit_stop = calib_start - (purge if n_calib else 0)`.

**Evidence:** Read in full; :86-89 apply `purge` unconditionally after `_resolve_counts` (:21-25) has already
collapsed non-positive fractions to 0.

**Disposition:** RESOLVED as suggested -- each purge is charged only when the slice it separates is non-empty. Verified across all four combinations: an empty carve now returns the full 1000-row fit slice where it previously returned 800, a calib-only carve charges one purge, a full carve charges both, and `purge=0` is unaffected. `tests/training/test_dead_slot_reads_and_stale_contracts.py`.

## Coverage

Fully or substantially read (39 files): `core/_training_context.py`, `core/_process_flag_scope.py`,
`core/_setup_helpers_pipeline_cache.py`, `core/_main_train_suite.py` (signature, orchestration :270-640, the
`finally` at :779-783), `core/_phase_config_setup.py` (:200-330, :480-546), `core/_main_train_suite_polars_gate.py`,
`core/_phase_helpers_fit_split.py` (:615-800), `core/_ensemble_chooser.py`, `core/_ar1_failsafe_veto.py`,
`core/_phase_train_one_target_pre_screen.py`, `core/_phase_train_one_target_post.py` (:235-267),
`core/_phase_train_one_target_dataset_cache.py` (:1-120), `core/_phase_composite_wrapping.py` (:78-260),
`core/_phase_temporal_audit.py` (:100-159), `core/_phase_recurrent.py` (:405-475),
`core/_phase_finalize.py` (:365-480), `core/_phase_finalize_calibration.py` (:70-260),
`core/_phase_train_one_target_schema.py` (:28-99), `core/_phase_train_one_target_weight_iteration.py`,
`core/_phase_train_one_target_body.py` (:194, :534-548), `_gpu_probe.py`, `_honest_decision_threshold.py`,
`_conformal_split.py`, `_calib_oof_outputs.py`, `_dataset_cache_fingerprint.py`, `_split_helpers.py` (:235-306),
`_overlapping_walk_forward_cv.py`, `_calibration_models.py` (:355-474), `_eval_helpers.py` (:540-686),
`_precompute.py` (:150-290), `mlp_runtime_defaults.py` (:55-175), `_helpers_training_configs.py` (:100-140,
:240-250, :375-385), `_lgb_shim_helpers.py` (:60-160), `lgb_shim.py` (:340-510), `xgb_shim.py` (:155-235,
:440-595), `_preprocessing_configs.py` (:119-215), `_feature_selection_config.py` (:140-190),
`_configs_base.py` (:228-260), `_model_configs_behavior.py` (:275-295),
`_composite_target_discovery_config_base.py` (:85-95, :615-625, :735-745), `_model_configs_ensembling.py`,
`evaluation.py` (:55-70), plus `cb/_cb_pool_build.py` (:130-329) and `cb/_cb_pool.py` (:640-720) for
TRAINING_CORE-3.

Sweeps across the full cluster (all 66 `core/*.py` plus all 47 loose `training/*.py`): every `except Exception`
in `core/` (268 occurrences across 54 files) triaged for debug-only silent fallbacks; `@njit`/`parallel=True`/
`prange`/`cuda.jit`/`cupy`/`kernel_tuning_cache` presence (the cluster contains no first-party numeric kernels --
all numba references are import warm-up or comments, so no perf finding is asserted); `threading.local`
set-without-restore; `min()==max()` / `n_unique()==1` polars constant-column checks; `pl.Categorical` versus
`pl.Enum`; JSON feeding a hash or cache key without sorted keys; `x or <falsy-legitimate-default>` traps;
`.copy()`/`.clone()`/`to_pandas()` on 100 GB-capable paths. AST sweeps: every function parameter never read in
its own body (64 hits triaged); every `getattr(ctx, "<name>", ...)` literal versus `TrainingContext`'s 124
declared slots (13 hits -> findings 4, 5, 13); every `getattr(<x>_config|_cfg|_conf, "<name>", ...)` literal
versus every field/param/attribute name in `src/mlframe` (2 hits -> finding 7); every config-class field with
<= 1 reference across all 2,445 src files (3 hits -> finding 11 plus two already-allowlisted).
