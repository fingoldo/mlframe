# Feature-selector suite-integration audit (2026-06-22)

Scope: how feature SELECTORS are wired into `train_mlframe_models_suite` — the integration/config/registry layer only. Selector internals (wrappers/boruta_shap/shap_proxied_fs/filters/mrmr) are owned by other agents and were read-only here.

Files in scope:
- `src/mlframe/training/core/_setup_helpers_pre_pipelines.py` (`_build_pre_pipelines`)
- `src/mlframe/training/_feature_selection_config.py` (`FeatureSelectionConfig`)
- `src/mlframe/feature_selection/registry.py` (selector dispatch)
- `src/mlframe/training/core/_main_train_suite.py`, `_phase_train_one_target*.py`, `_predict_pre_pipeline.py`, plus `pipeline/_pipeline_helpers*.py` (the fit/transform driver) read to follow the lifecycle.

---

## 1. Findings table (report FIRST, per project rules)

| # | file:line | severity | issue | disposition |
|---|-----------|----------|-------|-------------|
| F1 | `core/_setup_helpers_pre_pipelines.py:141` (stamp) + `filters/mrmr/_mrmr_class.py:3929` (read) | **P0** | MRMR cross-target identity cache (`_mlframe_identity_cache_override_`) is a ctx-scoped shared runtime dict stamped as a plain attribute -> deep-walked by `dill.dumps` into EVERY saved model bundle (size bloat + stale cross-suite replay on reload). Classic "runtime caches break pickle". | **RESOLVED** — stamp a `_NonPicklingCacheView` that delegates in-process but `__reduce__`s to `{}`. |
| F2 | `_feature_selection_config.py:168` | **P1** | `rfecv_kwargs` validator whitelisted `cluster_reduce`/`cluster_corr_threshold`/`cluster_min_reduction`/`cluster_corr_method`. The suite builds RFECV directly in `configure_training_params` (NOT via `registry._instantiate_rfecv`), so these keys are forwarded verbatim to `RFECV(**kwargs)` -> `TypeError` at construction. Config-time-green / fit-time-crash trap. | **RESOLVED** — dropped the cluster-key allowance from rfecv_kwargs validation; BorutaShap (which DOES route through the registry wrap) keeps it. |
| F3 | `feature_selection/registry.py:75-105` vs `targets/_train_eval_select_target.py:290` + `_trainer_configure.py:739` | **P1** | Registry/config/comments claim cluster-medoid pre-reduction is "DEFAULT-ON for the suite's RFECV", but the suite never calls `registry._instantiate_rfecv` — it uses pre-built RFECV from `configure_training_params`. So the documented default-ON RFECV cluster wrap is DEAD for the suite (only MRMR/BorutaShap, which go through the registry, actually get it). | **DOC/FUTURE** — corrected the config comment (F2). Wiring suite RFECV through the registry wrap is a selection-altering change requiring a multi-seed bench; out of scope for a safe wiring pass. Documented as FUTURE. |
| F4 | `feature_selection/registry.py:135-152` | **P1** | `ShapProxiedFS` is registered but has NO `FeatureSelectionConfig` flag and NO branch in `_build_pre_pipelines` -> unreachable from the suite. The registry's "adding a fourth selector = a single registration" claim is false: a new selector also needs a config flag, a `_build_pre_pipelines` branch, sticky-attr stamping, and `_SELECTOR_STICKY_ATTRS`/`_selector_kind` entries. | **DOC** — registry is a dispatch table, not the single wire-in point. Architecture verdict below proposes the unification. |
| F5 | `core/_phase_train_one_target_body.py:73` + `core/_setup_helpers_pre_pipelines.py:109,135` | Low | `_mlframe_selector_kind_` / `_mlframe_use_sample_weights_in_fs_` are also non-constructor attributes that pickle into the saved model (harmless small strings/bools, but inconsistent with the F1 cleanup intent). | **DOC/FUTURE** — string/bool markers are cheap and not stale-state hazards; left as-is. Noted for a future "scrub all `_mlframe_*` runtime markers before save" pass at the bundle boundary (trainer.py, out of scope). |
| F6 | `_feature_selection_config.py` (whole) | Low | The "first-class FS levers" surface (`rfecv_*`, `mrmr_*`) only feeds MRMR/RFECV; there is no lever surface for BorutaShap or ShapProxiedFS, so per-selector config is asymmetric. | **DOC** — consistent with F4; addressed conceptually in the architecture verdict. |

No silent-error-swallowing P0s found in the report-builder path: `_build_feature_selection_report` wraps every selector-attribute read in try/except by design (a failed FS report must never abort a training run) and falls back to a minimal report — acceptable. The predict-time pre-pipeline reuse path (`_predict_pre_pipeline.py`, `_prepare_test_split`) is robust: pure-selector vs value-transform discrimination (`_pre_pipeline_is_pure_selector`) correctly re-raises rather than serving un-transformed columns.

---

## 2. Correctness review (no leakage / split discipline)

- **Train-only fit contract: PASS.** `_apply_pre_pipeline_transforms` (`pipeline/_pipeline_helpers_apply.py`) only ever calls `fit`/`fit_transform` on the train frame and `transform` on val/test. The unsupervised pre-screen (`_phase_train_one_target_pre_screen.py`) computes drops on the train split and mirrors them to val/test (`apply_drops`), never reading held-out distribution. Group/timestamp protection + the `use_groups`+empty-protected SKIP guard are sound.
- **Row-preserving contract: PASS.** `_raise_pre_pipeline_rowcount_change` enforces that a pre-pipeline never changes the row count (resampler-in-FS-slot guard).
- **0-feature edge: PASS.** Empty selection short-circuits both train and val to `(N, 0)` symmetrically so trainer's 0-feature guard fires cleanly.
- **selector_kind stamping: PASS but fragile.** `_selector_kind` prefers the `_mlframe_selector_kind_` marker (stamped on the bare selector / GroupAwareMRMR wrapper) and falls back to class-name suffix. `_forward_selector_sticky_attrs` re-asserts the markers onto per-strategy clones (sklearn.clone strips non-ctor attrs) for both bare selectors and the inner `'pre'` step — correct.
- **Multioutput/group paths: PASS.** MRMR gets `strict_groups=True` defaulted under a group-aware split (raises rather than computing group-naive MI); RFECV gets `groups` threaded into fit via `_passthrough_cols_fit_transform`. The ranker FS path forwards `groups` to RFECV only (BorutaShap ignores it) — correct.
- **Cold-import gating: PASS.** MRMR (~25s), BorutaShap (shap/matplotlib/seaborn ~2s), ShapProxiedFS imports stay lazy inside `registry.instantiate` and inside the `use_*` branches of `_build_pre_pipelines`; the pre-screen uses the shorter `feature_selection.pre_screen` path to avoid the `filters/__init__` njit cascade.
- **must_include / NaN: PASS.** must_include/exclude flow via the lever merge into `rfecv_kwargs`; MRMR handles NaN natively (`nan_strategy`), not wrapped in an imputer that would destroy the signal.

---

## 3. Optimization review

cProfile harness committed at `src/mlframe/training/_profile_fs_suite_integration.py` (full pipeline: train + predict + save/load of an MRMR suite). See section 5 for numbers.

Observations on the wiring (independent of the profile):
- **Cross-target FS reuse already present:** the pipeline cache is hoisted to `ctx._pipeline_cache` (shared across targets) and the MRMR identity cache is ctx-scoped — both correct. The F1 fix preserves this sharing (the view delegates to the same backing dict) while removing the pickle bloat.
- **No redundant selector refit found** in the per-target loop: the structural-identity `_PRE_PIPELINE_CACHE` short-circuits identical pre_pipelines across models within a target, and the fitted pipeline state is transferred (shallow per-attr copy) to cloned instances for predict-time transform.
- **No unnecessary dense copy / dtype coercion** introduced by the wiring layer; the predict-side passthrough stash already batches the polars->pandas conversion once (not per-column).

---

## 4. Architecture verdict (registry extensibility)

**Is adding a new selector a single registration? No.** The registry (`registry.py`) is a clean *instantiation+report* dispatch table, but it is NOT the single wire-in point the docstring claims. To actually surface a new selector in the suite you must touch FIVE places:

1. `FeatureSelectionConfig` — add a `use_<sel>` flag + `<sel>_kwargs` + a validator.
2. `_build_pre_pipelines` — add a `if use_<sel>:` branch that instantiates via the registry and stamps markers.
3. `_SELECTOR_STICKY_ATTRS` / `_forward_selector_sticky_attrs` — already generic (no change needed if you reuse the standard markers).
4. `_selector_kind` — add the kind string to the allow-list.
5. `_build_feature_selection_report` — add a per-kind extraction branch (or rely on the registry `report_extract`, which is defined in the Protocol but NOT yet consumed by `_build_feature_selection_report`).

`ShapProxiedFS` (F4) is the proof: registered, but invisible to the suite.

**Concrete improvement ideas (FUTURE, not applied — they restructure the integration and need their own validation pass):**
- Make `_build_pre_pipelines` iterate `registry.available()` data-driven: each spec carries `config_enable_field` ("use_mrmr_fs"), `config_kwargs_field` ("mrmr_kwargs"), and `selector_kind`. The branch-per-selector vanishes; adding a selector becomes one `register(...)` call + one config field.
- Actually consume `FeatureSelectorSpec.report_extract` in `_build_feature_selection_report` so per-selector report logic lives next to the spec (the Protocol already declares it; the central builder still hard-codes MRMR/RFECV/BorutaShap branches).
- Unify the cluster-medoid wrapping defaults: today MRMR/BorutaShap wrap (default-ON) but suite RFECV does NOT (F3) — drive the wrap decision from a single spec attribute (`cluster_wrap_default: bool`) and route ALL suite selectors (incl. RFECV) through the registry instantiate, so the documented default actually holds. This is selection-altering for RFECV -> gate behind a multi-seed bench before flipping.
- Move `_mlframe_*` runtime-marker scrubbing to a single `_scrub_runtime_markers(pre_pipeline)` called at the bundle-save boundary, generalising the F1 fix to F5.

---

## 5. cProfile numbers

Full pipeline (train + predict + save/load), MRMR suite, n=1500 x 20, lgb + ordinary, single run, CPU-only. Total 656.2s / 145M calls.

Top cumulative / tottime are ENTIRELY reporting + model-fit, NOT the FS wiring:

| rank (tottime) | tottime s | frame |
|---|---|---|
| 1 | 329.7 | `_tkinter.tkapp.call` (matplotlib/tkinter diagnostic plots) |
| 2 | 34.9 | `llvmlite ffi.__call__` (numba JIT of MRMR/perm kernels) |
| 3 | 21.7 | `_thread.lock.acquire` (joblib/thread wait) |
| 4 | 6.8 | `matplotlib ft2font.set_text` |
| 5 | 6.1 | `numpy.ufunc.reduce` (MRMR MI) |

Top cumulative: `reporting/renderers/save.py` 481s, `matplotlib figure.savefig` 292.8s, `diagnostics_dispatch` 432s.

**FS-wiring verdict: no actionable speedup.** `_build_pre_pipelines`, `_build_feature_selection_report`, `_apply_pre_pipeline_transforms`, `_selector_kind`, and the selector pickle path do NOT appear in the top 30 cumulative or top 20 tottime — they are below the profiler's noise floor at this shape. The wall is dominated by matplotlib/tkinter diagnostic rendering (outside this audit's scope) and the model fit. The F1 `_NonPicklingCacheView` adds three delegating method calls per cache hit (negligible) and REMOVES bytes from every saved model. No optimization is applied to the wiring because none is measurable; the harness is committed so a future maintainer can re-profile a no-diagnostics shape if the wiring grows.

Harness: `src/mlframe/training/_profile_fs_suite_integration.py`.
