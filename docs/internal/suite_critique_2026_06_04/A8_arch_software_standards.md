# A8 — Architecture & Software-Engineering Standards Audit

`train_mlframe_models_suite` and its estimator/wrapper/config surface, viewed strictly
through a software-engineering lens (sklearn API compliance, config design, cohesion/coupling,
error handling, public API & docstring contracts, typing, logging/observability, dependency hygiene).
ML methodology / leakage is explicitly out of scope (covered by a sibling agent).

**Scope read:** `core/_main_train_suite.py`, `core/main.py`, `core/_phase_config_setup.py`,
`core/_main_train_suite_phases.py`, `core/_setup_helpers.py`, `core/_misc_helpers.py`,
`core/_training_context.py`, `training/trainer.py`, `training/_composite_target_estimator*.py`,
`training/composite_estimator.py`, `training/lgb_shim.py`, `training/_model_factories.py`,
`training/configs.py`, `training/_configs_base.py`, `estimators/{base,custom}.py`, the public
`__init__.py` surfaces, and `tests/test_sklearn_compliance_composite.py`.

**Verified facts used below:**
- `train_mlframe_models_suite` has **34 parameters**, all `POSITIONAL_OR_KEYWORD`, **0 mutable defaults** (verified via `inspect.signature`).
- A sklearn-compliance test file exists (`tests/test_sklearn_compliance_composite.py`) but does **not** run `check_estimator` / `parametrize_with_checks` on `CompositeTargetEstimator` (only get_params/clone/fit-predict happy-path).

---

## Findings

### A8-01 — OK-positive — `CompositeTargetEstimator` follows sklearn fitted-attr + clone conventions well
**File:** `src/mlframe/training/_composite_target_estimator.py:113-128`, `:605-628`, `:322-353`
**What's right:** `__init__` is stateless — it only stores constructor args verbatim onto same-named attributes, no derived computation, no validation (the sklearn contract). Fitted state uses the trailing-underscore convention (`estimator_`, `fitted_params_`, `feature_names_in_`, `runtime_stats_`). `fit` clones `base_estimator` so the prototype stays clean (`:547`). The `__sklearn_clone__` override is a thoughtful, correct guard: a `from_fitted_inner`-built instance carries fitted state outside the `__init__` signature, so cloning it would yield a silent unfitted shell; the override raises `NotImplementedError` with a clear message and falls back to the default clone for `fit()`-built instances (`:337-353`).
**Why it matters:** This is the hardest part of the sklearn contract to get right and it is right; worth preserving under refactors.
**Recommendation:** Keep. Add a regression test pinning the `__sklearn_clone__` raise on a `from_fitted_inner` instance if not already present.
**Confidence:** High

### A8-02 — P2 — `check_estimator` is never run against any mlframe wrapper; "compliance" test only covers happy path
**File:** `tests/test_sklearn_compliance_composite.py:24-31`, `:93-137`
**What's wrong:** The file's docstring explicitly states it is "intentionally narrower than `check_estimator`" and the `CompositeTargetEstimator` block only checks get_params round-trip, clone-of-unfitted, and one fit→predict. `estimators/base.py` *imports* `check_estimator`/`check_transformer_general` (`base.py:10`) but nothing calls them. So the estimator-protocol invariants `check_estimator` enforces (e.g. `fit` returns `self`, `n_features_in_` set, no attribute set in `__init__`, idempotent re-fit, NotFittedError on pre-fit predict, `get_params`/`set_params` exhaustiveness against the signature) are not machine-verified. Several wrappers are knowingly non-compliant (`base_column` X-contract, RFECV's rich init), which is a legitimate reason to skip the *full* suite — but the individual checks that DO apply are not run either.
**Why it matters:** Wrapper regressions on sklearn minor bumps (the exact class of bug the file's docstring says it exists to catch) will slip if only the happy path is asserted. The reasoning quoted ("XFAILs would be high-noise") conflates "the whole suite is incompatible" with "no individual check is worth running".
**Recommendation:** Use `sklearn.utils.estimator_checks.check_estimator(..., expected_failed_checks={...})` (sklearn ≥1.6 supports a per-check skip dict) or `parametrize_with_checks([...])` with an xfail-marker callback to run the applicable subset on `CompositeTargetEstimator`, the early-stop wrappers, and the `Pd*` transformers. Pin the known-incompatible checks explicitly with reasons so the rest are enforced.
**Confidence:** High

### A8-03 — P2 — `CompositeTargetEstimator` predict-family methods are bound by runtime class-attribute assignment, defeating static analysis
**File:** `src/mlframe/training/_composite_target_estimator.py:672-692`
**What's wrong:** `predict`, `predict_quantile`, `predict_pre_clip`, `_predict_unclipped`, `update`, `get_booster`, and the `feature_importances_`/`coef_`/`intercept_`/`booster_`/`n_features_in_` properties are NOT defined in the class body — they are imported from sibling modules and stapled onto the class at module bottom (`CompositeTargetEstimator.predict = _pred.predict`). The class therefore has no `predict` in its own body, so IDEs, mypy, and `help(CompositeTargetEstimator)` see an incomplete API; readers grepping the class definition won't find `predict`. This is the documented "monolith split via re-export" pattern, but for a *public sklearn estimator* the bound methods are the public contract.
**Why it matters:** Discoverability and tooling. A new contributor reading `_composite_target_estimator.py` cannot tell the class is a regressor with `predict`/`predict_quantile` without finding the bottom-of-file rebinding. `RegressorMixin` provides a `score` that calls `self.predict` — works at runtime, invisible statically.
**Recommendation:** Either (a) define thin stub methods in the class body that delegate to the sibling implementations (`def predict(self, X): return _pred_predict(self, X)`), giving static tools + docstrings a home; or (b) at minimum add `if TYPE_CHECKING:` method signatures to the class body. The current carve was driven by the <1k-LOC rule; a stub-delegation keeps both the line budget and the discoverable surface.
**Confidence:** Med

### A8-04 — P1 — `_ensure_config` dict-vs-None branch is asymmetric: dict path accepts arbitrary keys, None path filters to declared fields
**File:** `src/mlframe/training/core/_setup_helpers.py:79-89`
**What's wrong:**
```python
if isinstance(config, dict):
    return config_class(**config)            # ALL keys flow in (extra='allow' absorbs typos)
elif config is None:
    return config_class(**{k: v for k, v in kwargs.items() if k in config_class.model_fields})
return config
```
When a caller passes a `dict`, every key is splatted into the Pydantic constructor. Because `BaseConfig.model_config` sets `extra="allow"` (`_configs_base.py:236`), a typo like `{"iteratoins": 100}` is silently absorbed as an extra field and has no effect — caught only by a WARNING log from `_warn_on_unknown_extras` (`_configs_base.py:250-271`), not an error. The `None` path, by contrast, filters `kwargs` to `model_fields`. So the two construction paths have different strictness, and the dict path's only typo defense is a log line a batch/CI run will not see.
**Why it matters:** The dual-accept `Union[Config, Dict]` pattern is offered across ~15 suite config params explicitly for ergonomics; the dict path is the *more* error-prone one (no type checker, no field name autocomplete) yet has the *weaker* validation. Silent dict-key typos in a 4-hour training run are an expensive failure mode (the config field stays at its default and the run produces wrong-but-plausible output).
**Recommendation:** Provide a `strict=True` mode that, for the dict path, raises (or at least logs at ERROR) on unknown keys not in `_known_extras`. Alternatively flip `extra="forbid"` on the leaf configs that have no legitimate pass-through extras (most of them — only `ModelHyperparamsConfig`-style configs need `extra="allow"`), keeping `extra="allow"` + `_known_extras` only where pass-through is real. This is the single highest-leverage config-design fix.
**Confidence:** High

### A8-05 — OK-positive — Suite kwargs surface is large but well-organized; no mutable defaults
**File:** `src/mlframe/training/core/_main_train_suite.py:95-141`
**What's right:** 34 params, but only ~10 are "what does this do" (df/target/model_name/extractor/model lists/target_type) and the remaining ~22 are typed `Optional[Union[SomeConfig, Dict]]` config-bundle handles — i.e. the wide surface is a *facade over grouped dataclasses*, not 34 loose scalars. Zero mutable defaults (verified). `Optional[...] = None` is used correctly throughout; configs are resolved centrally in one `setup_configuration` call. This is a defensible answer to the "god-function signature" smell.
**Why it matters:** Confirms the config-bundle design is doing its job; the param count is not itself a defect.
**Recommendation:** Keep. (See A8-04 for the validation gap, A8-12 for the docstring gap.)
**Confidence:** High

### A8-06 — P1 — `train_mlframe_models_suite` is a god-orchestrator with deep, brittle phase coupling via a 70+-slot mutable `ctx`
**File:** `src/mlframe/training/core/_main_train_suite.py:95-723`; `core/_training_context.py:25+`
**What's wrong:** The function body is a ~600-line sequence of phase calls where each phase returns a large positional tuple that is immediately fanned out into locals AND mirrored onto `ctx` via `_bulk_setattr_to_ctx(ctx, ("train_df","val_df",...), locals())` (e.g. `:368-373`, `:422-429`, `:465-469`). The codebase comments admit this is a half-finished migration ("in-progress migration from the legacy phase-returns-big-tuple form to a pure ctx-form"; `:363-367`). The result: state lives in two places at once (locals AND ctx), each phase has an enormous implicit input/output contract, and `_bulk_setattr_to_ctx` reads `locals()` reflectively to copy names — coupling that no static tool can verify. `TrainingContext` has 70+ fields, almost all typed `Any` (`_training_context.py:38-55`).
**Why it matters:** This is the central maintainability risk. The dual local/ctx bookkeeping is exactly the kind of state that drifts: a phase that writes `ctx.train_df` but the orchestrator that re-reads a stale local `train_df` (or vice versa) is a whole bug class, and the `del df; ctx.df = None` dance (`:358-362`) shows the team already hit memory-aliasing consequences. The `locals()`-reflection copy is fragile (rename a local and the slot silently goes missing — the helper's own docstring at `_misc_helpers.py:966-977` documents a prior `train_df_pandas_pre` slot-miss bug).
**Recommendation:** Finish the ctx migration in one direction. Pick ctx-as-single-source-of-truth: phases read/write `ctx.*` only, the orchestrator stops binding locals and stops calling `_bulk_setattr_to_ctx(..., locals())`. This removes the dual-state hazard and makes phase contracts explicit (each phase declares which ctx slots it reads/writes). Strongly type the ctx config slots (they're all known Pydantic types) instead of `Any` so the slots catch typos.
**Confidence:** High

### A8-07 — P2 — Module-global monkeypatching of third-party libs (LightGBM, CatBoost, XGBoost, joblib/loky) is process-wide and irreversible
**File:** `src/mlframe/training/_model_factories.py:26-78` (`_patch_lgb_feature_names_in_setter`), `:158-272` (`_patch_dataset_constructors_with_logging`), `:89-110` (`apply_third_party_patches_once`); `training/__init__.py:99-116` (loky); `core/_main_train_suite.py:215` + `core/_main_train_suite_phases.py:19-33`
**What's wrong:** Calling the suite mutates global class state of installed third-party libraries: it replaces `lightgbm.sklearn.LGBMModel.feature_names_in_` with a patched property, wraps `catboost.Pool.__init__` / `xgboost.DMatrix.__init__` / `lightgbm.Dataset.__init__` with logging wrappers, and rebinds `joblib.externals.loky.backend.context._count_physical_cores`. These patches are install-once with marker flags (idempotent) and the import-time→lazy refactor (audit 2026-05-17) was a genuine improvement — bare `import mlframe.training` no longer mutates anything. But once the suite runs, the patches persist for the process lifetime and affect *every other consumer of those libraries in the same process*, including non-mlframe code. There is no un-patch path.
**Why it matters:** A user who runs the suite and then trains a plain `LGBMClassifier` in the same notebook gets mlframe's patched `feature_names_in_` setter and mlframe's dataset-build logging on their unrelated model. The loky override (logical=physical approximation) silently changes joblib parallelism heuristics process-wide. This is acceptable as a pragmatic workaround for upstream bugs, but it is hidden global state that violates the principle of least surprise.
**Recommendation:** Document the process-wide side effects prominently in the suite docstring (currently undocumented). The `feature_names_in_` setter is described as "belt-and-braces" defense for a path the primary fix already covers — consider gating it behind a flag now that the primary fix (pandas-path conversion) is in place. For the dataset-build logging wrapper, prefer the explicit `make_pool`/`make_dmatrix`/`make_lgb_dataset` factories (already present at `:125-155`) over global `__init__` wrapping where mlframe controls the call site.
**Confidence:** Med

### A8-08 — Low — `print()` in library training paths (eval-metric callback + extractor + cat-diag)
**File:** `src/mlframe/training/_helpers_training_configs.py:350`; `src/mlframe/training/extractors.py:140`; `src/mlframe/training/_eval_helpers.py:79,126,128`; `src/mlframe/estimators/custom.py:480`
**What's wrong:** Five production (non-benchmark) sites use `print()` instead of the module logger:
- `_helpers_training_configs.py:350` — `print(len(y_true), "integral_calibration_error=", err)` inside a CatBoost eval-metric closure, gated on `verbose`. This fires *per eval iteration during training* and bypasses the logging system (no level, no logger name, not capturable/filterable).
- `_eval_helpers.py:79,126,128` — env-gated (`MLFRAME_CAT_DIAG`) diagnostic prints; lower severity since opt-in.
- `extractors.py:140` — `print("Processed data:")` in `showcase`-style code.
- `custom.py:480` — `print(...)` inside `soft_winsorize`'s linear branch (a stray debug print left in a numeric utility).
**Why it matters:** Library code emitting to stdout cannot be redirected to a log file, has no severity, and on Windows can crash on non-ASCII (per the project's cp1251 rule). The per-iteration calibration print is the worst because it scales with training rounds.
**Recommendation:** Replace `_helpers_training_configs.py:350` and `custom.py:480` with `logger.debug(...)`. Leave the `MLFRAME_CAT_DIAG`-gated ones as `logger.debug` too (they're opt-in but should still route through logging).
**Confidence:** High

### A8-09 — P2 — Public API surface is well-delineated but `training/__init__.py` re-exports private `_*` symbols without underscore
**File:** `src/mlframe/training/__init__.py:158-159`, `:263-265`; `mlframe/__init__.py:1-31` (docstring contract)
**What's wrong:** The top-level `mlframe/__init__.py` documents a clean contract: only `__version__` + config enums are public at top level; everything else is deep-imported; `_*` modules are internal. `training/__init__.py` mostly honors this via `_LAZY_IMPORTS`. However a few entries re-export a `_private` implementation under a public name (`'canonical_predict_proba_shape': ('.helpers', '_canonical_predict_proba_shape')`, `'predict_from_probs': ('.helpers', '_predict_from_probs')`, `'short_model_tag': ('._format', ...)`). This is intentional ("public re-exports of underscore source") and fine — but the *source* symbols keep their underscore, so two names (`_canonical_predict_proba_shape` and `canonical_predict_proba_shape`) refer to the same object, and which one is "the" public API is ambiguous.
**Why it matters:** Minor. The delineation is otherwise good and enforced by meta-tests (`test_no_underscore_imports_cross_package.py`, `test_no_production_underscore_imports.py`). The double-name just means a future reader can't tell stability guarantees from the name alone.
**Recommendation:** Keep the public alias; add a one-line docstring note on the source `_` functions stating "public alias: `mlframe.training.canonical_predict_proba_shape`" so the stability contract is discoverable from either name.
**Confidence:** Med

### A8-10 — P2 — Return-type contract `Tuple[Dict, Dict]` is under-specified and the actual return is `dict(models), metadata`
**File:** `src/mlframe/training/core/_main_train_suite.py:141`, `:723`; docstring `:189-190`
**What's wrong:** The signature annotates `-> Tuple[Dict, Dict]` and the docstring says "Returns: Tuple of (models_dict, metadata_dict)". `Dict` with no parameters is `Dict[Any, Any]` — it tells a caller nothing about the nested structure (models is `{target_type: {target_name: {model_name: entry}}}`; metadata is a deeply-nested dict with documented keys like `"composite_target_specs"`, `"dummy_baselines"`, `"text_features"`, `"schema_version"`). The LTR early-dispatch and auto-route paths (`:269-270`, `:321-322`) return `_ltr_result` directly, which comes from a *different* function (`_maybe_dispatch_to_ltr_ranker_suite`) — its return shape compatibility with `Tuple[Dict, Dict]` is asserted only implicitly.
**Why it matters:** The two return dicts ARE the entire output contract of the suite; an un-parameterized `Dict` provides no machine-checkable guarantee and forces every consumer to reverse-engineer the structure. The LTR-result passthrough is a second, unverified return shape behind the same annotation.
**Recommendation:** Introduce `TypedDict`s (or at least a documented module-level type alias `SuiteModels = Dict[str, Dict[str, Dict[str, Any]]]` and `SuiteMetadata = Dict[str, Any]`) and annotate `-> Tuple[SuiteModels, SuiteMetadata]`. Assert in `_maybe_dispatch_to_ltr_ranker_suite` (or at the two return sites) that the LTR path returns the same 2-tuple shape.
**Confidence:** Med

### A8-11 — Low — Typing on the suite signature is good; typing inside the orchestrator + helpers leans heavily on `Any`
**File:** `src/mlframe/training/core/_main_train_suite_phases.py` (every helper is `(...: Any) -> Any` / `-> tuple`), `core/_training_context.py:38-55`, `trainer.py` config-builder uses bare params
**What's wrong:** The public suite signature is well-typed (`Optional[Union[Config, Dict]]` per param). But the layer beneath is almost entirely `Any`: `_main_train_suite_phases.py` helpers take `ctx: Any`, `precomputed: Any`, `metadata: dict` and return `tuple`/`Any`; `TrainingContext`'s 18 config slots are all `Any` despite each having a known Pydantic type; `_build_configs_from_params` in `trainer.py:278-378` is ~60 untyped params. The `forward-ref strings` for config types in the suite signature (`Optional["TargetTypes"]`, `Optional["LearningToRankConfig"]`) are inconsistent — some configs are imported and used bare, others are string forward-refs even though they're imported at module top (`:21-42`).
**Why it matters:** The well-typed boundary gives a false sense of coverage; once you cross into the phase layer, no type checker can help. The `ctx: Any` everywhere is the typing analog of the A8-06 god-context problem.
**Recommendation:** Type the `TrainingContext` config slots with their real Pydantic types (`preprocessing_config: PreprocessingConfig | None`) — this is a low-risk, high-value change that turns the slots into typo-catchers. Type the phase helper `ctx: TrainingContext`. Make the forward-ref usage consistent (all imported configs should be used bare, not quoted, since `from __future__ import annotations` is active and they're all imported).
**Confidence:** Med

### A8-12 — P2 — Suite docstring documents ~13 of 34 parameters; ~21 config params undocumented in the Args block
**File:** `src/mlframe/training/core/_main_train_suite.py:142-209`
**What's wrong:** The Args section documents `df`, `target_name`, `model_name`, `features_and_targets_extractor`, the model-list params, `preprocessing_config`, `split_config`, `pipeline_config`, `feature_selection_config`, `hyperparams_config`, `behavior_config`, `reporting_config`, `output_config`, `outlier_detection_config`, `verbose`, `precomputed`. It does NOT document: `target_type`, `ranking_config`, `preprocessing_extensions`, `feature_types_config`, `linear_model_config`, `multilabel_dispatch_config`, `confidence_analysis_config`, `baseline_diagnostics_config`, `dummy_baselines_config`, `quantile_regression_config`, `composite_target_discovery_config`, `feature_handling_config`, `enable_target_distribution_analyzer`. Some of these have inline `#` comments at the parameter site (good) but those don't surface in `help()` / Sphinx.
**Why it matters:** For a 34-param public entry point, a docstring covering <40% of params is a real contract gap — callers can't discover `composite_target_discovery_config` or `feature_handling_config` from `help(train_mlframe_models_suite)`. The project's own convention ("Content richness > parsimony", "Always document refactors/new features") makes this in-scope.
**Recommendation:** Add a one-line entry per undocumented config param pointing at the config class (`composite_target_discovery_config: see CompositeTargetDiscoveryConfig`). The inline comments already contain the prose; promote a one-liner into the Args block.
**Confidence:** High

### A8-13 — OK-positive — Error handling in the suite/config layer is disciplined: narrow excepts with surfaced context
**File:** `core/_main_train_suite_phases.py:148-150`, `:247-269`; `core/_phase_config_setup.py:324-337`; `_composite_target_estimator_utils.py:13-23`
**What's right:** The codebase has clearly internalized the "no silent swallow" rule. Broad-except sites consistently log with `type(err).__name__` + message and either re-raise or document why a fallback is safe (e.g. the CB-pool-clear except at `_phase_config_setup.py:329-337` is narrowed to `(ImportError, AttributeError)` and WARNs with the stale-Pool risk spelled out; the dummy-baselines precompute path at `_main_train_suite_phases.py:255-269` *raises* `RuntimeError` rather than swallow a frozen-pydantic failure). `CompositeTargetEstimator` pre-fit attribute access raises `NotFittedError` via `_require_fitted` instead of returning `None` (`_composite_target_estimator_utils.py:13-23`), the documented sklearn convention. Fit-time invalid inputs raise `DomainViolationError`/`ValueError` with actionable messages.
**Why it matters:** Confirms the error-surfacing discipline the project's memory rules demand is actually present in this layer; this is a positive baseline.
**Recommendation:** Keep. (Two narrow exceptions noted below.)
**Confidence:** High

### A8-14 — P2 — Two broad `except Exception` swallow paths in the suite orchestrator hide failures
**File:** `core/_main_train_suite.py:592-596` (discovery-cache-dir build); `core/_main_train_suite_phases.py:431-473` (votenrank leaderboard export, double broad-except)
**What's wrong:**
- `_main_train_suite.py:592-596`: `try: if data_dir: _discovery_cache_dir = str(_P(data_dir)/".discovery_cache") except Exception: _discovery_cache_dir = None`. A bare `except Exception` around pure string/Path construction will swallow anything (including a programming error) and silently disable the discovery cache with no log.
- `_main_train_suite_phases.py:431-473`: `export_votenrank_leaderboards` wraps the whole body in `try/except Exception` (WARN), and the inner CSV write in a *second* `try/except Exception` (WARN). The outer broad-except means any bug in the leaderboard-collection loop (e.g. an attribute rename on `ctx.ensembles`) is downgraded to a WARN and the feature silently produces nothing.
**Why it matters:** These are the two remaining "silent fallback" sites in the orchestrator that contradict the otherwise-good discipline in A8-13. The discovery-cache one has no log at all.
**Recommendation:** Narrow the cache-dir except to the only realistic failure (`TypeError`/`OSError`) and add a `logger.debug`. For the leaderboard export, keep the inner CSV-write guard but narrow the outer except (or at least log at WARNING with the exception type, which the inner one does but the structure invites masking real bugs in collection).
**Confidence:** Med

### A8-15 — Low — `from typing import *` and stray test-only import in production estimator module
**File:** `src/mlframe/estimators/custom.py:18` (`from typing import *`); `src/mlframe/estimators/base.py:10` (`from sklearn.utils.estimator_checks import check_estimator, check_transformer_general` — imported, never used)
**What's wrong:** `custom.py` does `from typing import *`, a wildcard import that pollutes the namespace and is flagged by every linter; it also makes the module's actual type dependencies opaque. `base.py` imports `check_estimator`/`check_transformer_general` from sklearn at module top but never calls them (dead import; presumably a vestige of an intent to run compliance checks — see A8-02).
**Why it matters:** Hygiene. The wildcard import is a latent shadowing hazard (any name added to `typing` could clobber a local). The dead `check_estimator` import adds an unnecessary import-time dependency on `sklearn.utils.estimator_checks` (which pulls a non-trivial chunk of sklearn) into a module that just defines estimators.
**Recommendation:** Replace `from typing import *` with the explicit names used. Remove the unused `check_estimator`/`check_transformer_general` import from `base.py` (or wire it into a real test per A8-02).
**Confidence:** High

### A8-16 — OK-positive — Optional-dependency import hygiene is consistently correct
**File:** `trainer.py:36-69,158-171`; `lgb_shim.py:77-98`; `composite_estimator.py:18-23`; `mlframe/__init__.py:36-109`
**What's right:** Every heavy/optional backend (matplotlib, catboost, lightgbm, xgboost, torch, ngboost, polars, cupy) is imported in a `try/except ImportError` that binds the name to `None` (or `object` for base-class fallbacks) so `import mlframe.training` stays cheap and doesn't crash on a missing backend; failures defer to first use. The torch import also catches `OSError` for Windows DLL-load failures (`trainer.py:162`). The cupy guard (`mlframe/__init__.py:36-109`) is a sophisticated, well-documented poison-`sys.modules["cupy"]=None` workaround for a broken-CUDA RecursionError, with an opt-out env var. `lgb_shim` binds `LGBMClassifier = object` on missing lightgbm so the subclass definitions don't NameError. Tests use `pytest.importorskip` (e.g. `test_sklearn_compliance_composite.py:38`).
**Why it matters:** This is textbook optional-dep handling and directly supports the "fast import / optional-dep isolation" goal in `training/__init__.py`. No import-time hard dependency on any model backend.
**Recommendation:** Keep. The only nuance: the `except Exception` (not `ImportError`) in `composite_estimator.py:21` for the polars import is slightly broader than its siblings — harmless here but inconsistent.
**Confidence:** High

### A8-17 — P2 — `lgb_shim` eval-set normalization is a fragile heuristic stack reimplementing LightGBM's own input parsing
**File:** `src/mlframe/training/lgb_shim.py:452-500`
**What's wrong:** To detect whether `eval_set` is a bare `(X_val, y_val)` tuple vs a list-of-pairs, the shim runs a multi-arm heuristic: check tuple-of-len-2/3 with non-list first element (`:464-468`), then check list-of-len-2/3 with array-like first element via `hasattr(_first, "shape")/"columns"/"iloc"/"dtypes"` (`:473-499`), then a "shape-shape disagreement" arm comparing `_second.shape[1] == _first.shape[1]` to distinguish `(X,y)` from a list of feature matrices (`:489-499`). This duck-typed shape archaeology is reimplementing the input-shape disambiguation that `LGBMModel.fit` does natively — and it can misclassify edge inputs (e.g. a list of two equal-width 2-D arrays that genuinely IS two eval sets, or a multilabel `y` that's 2-D matching X's width).
**Why it matters:** The shim is a *drop-in subclass* whose entire value proposition is behaving identically to `LGBMClassifier.fit`. The eval-set parsing is the riskiest divergence point; a misclassification silently feeds the wrong arrays to `lgb.train` and either crashes opaquely or trains against a wrong eval set (degrading early-stopping). The module docstring already frames the shim as temporary pending an upstream PR.
**Why it matters (eng):** This is complex, behavior-critical code with many comment-documented past bugs (label-name aliasing, `valid_0` vs `validation_0`) — a sign the heuristic surface keeps producing regressions.
**Recommendation:** Where possible, require callers to pass the canonical list-of-tuples form to the shim (normalize ONCE at the suite boundary in `_setup_eval_set`, not inside the shim), shrinking the shim's heuristic to a single assert. If the bare-tuple support must stay, add explicit unit tests pinning each arm (bare 2-tuple, bare 3-tuple, list-of-one-pair, list-of-two-pairs, 2-D multilabel y) so future edits don't silently break a branch.
**Confidence:** Med

### A8-18 — Low — `lgb_shim`/composite estimators correctly exclude unpicklable runtime caches from `__getstate__`, matching the project pickle rule
**File:** `src/mlframe/training/lgb_shim.py:256-310`
**What's right:** `_DatasetReuseMixin.__getstate__` strips the ctypes-backed cached `Dataset` pointers and their key siblings before pickling (`:264-265`) and stamps the lightgbm version for drift detection on load; `__setstate__` re-initializes the cache attrs so legacy saves load cleanly and WARNs on version skew (`:279-310`). This directly satisfies the project's "runtime caches break pickle — needs `__getstate__` exclusion in the SAME change" memory rule. `__sklearn_clone__` on the shim subclasses produces a fresh empty-cache instance (`:692-699`, `:742-743`).
**Why it matters:** Confirms the pickle-safety discipline is correctly applied on the live-cache estimators in scope.
**Recommendation:** Keep.
**Confidence:** High

### A8-19 — P2 — `_build_configs_from_params` (trainer.py) silently drops unknown reporting kwargs via `**_unused_reporting_kwargs`
**File:** `src/mlframe/training/trainer.py:362-377`
**What's wrong:** The function ends its signature with `**_unused_reporting_kwargs` and the comment explains it exists so that new `ReportingConfig` fields flowing through `**all_params` (a splat of the caller's reporting_config) don't break the call with `TypeError`. "everything else is silently dropped so the splat survives future additions." So a new ReportingConfig field is silently NOT plumbed into the rebuilt config unless someone also adds an explicit param + routing line here (as was done for `honest_estimator_diagnostics` and `mase_seasonality`). This is a maintenance trap: the config field exists, the caller sets it, and it silently has no effect on this path.
**Why it matters:** Same silent-typo/silent-drop class as A8-04 but on the train-eval path. The comment frames it as intentional robustness, but it trades a loud `TypeError` (which would point at the exact missing field) for a silent no-op (which surfaces as "why is my reporting setting ignored?" debugging hours later).
**Recommendation:** Instead of swallowing, rebuild the `ReportingConfig` by copying ALL fields from the incoming config object generically (`ReportingConfig(**{f: getattr(rep,f) for f in ReportingConfig.model_fields})`) rather than enumerating each field by hand, so new fields auto-propagate. If a catch-all must remain, log unknown keys at DEBUG so they're discoverable.
**Confidence:** Med

### A8-20 — Low — `verbose: int` is documented as 0/1/2 levels but used inconsistently (truthiness vs level comparison)
**File:** `core/_main_train_suite.py:140,178,211`; `_configure_mlp_params` & many phases use `if verbose:`; `_phase_config_setup.py:120` uses level-aware `_step_done`
**What's wrong:** The suite documents `verbose: Verbosity level (0=silent, 1=info, 2=debug)`. But most consumers treat it as a boolean (`if verbose:` — so 1 and 2 are indistinguishable), while a few sites do level-aware work. There's no single mapping from `verbose` to a logging level; instead each phase re-interprets it, and several phases pass `verbose=bool(verbose)` downward (`_main_train_suite.py:668`, `_main_train_suite_phases.py:312`), discarding the 0/1/2 distinction entirely.
**Why it matters:** The documented 3-level contract isn't honored — `verbose=2` rarely produces more than `verbose=1`. Minor, but it's a public-API contract that the implementation doesn't keep.
**Recommendation:** Either (a) map `verbose` to a logging level once at entry (`logger.setLevel`) and let levels do the work, dropping the per-site `if verbose:` re-interpretation; or (b) downgrade the docstring to "0=silent, >0=verbose" to match reality.
**Confidence:** Med

---

## Summary of dispositions (for the integrating agent)

| ID | Severity | One-line |
|----|----------|----------|
| A8-01 | OK-positive | Composite estimator nails sklearn fitted-attr + clone conventions |
| A8-02 | P2 | `check_estimator` never run on any wrapper; only happy-path asserted |
| A8-03 | P2 | Public predict-family bound by runtime class-attr assignment (invisible to tooling) |
| A8-04 | P1 | `_ensure_config` dict path accepts arbitrary keys (silent typo absorption) vs filtered None path |
| A8-05 | OK-positive | 34-param facade is well-grouped; 0 mutable defaults |
| A8-06 | P1 | God-orchestrator + dual local/ctx state + `locals()`-reflection copy (half-finished migration) |
| A8-07 | P2 | Process-wide irreversible monkeypatching of LGB/CB/XGB/loky, undocumented in suite |
| A8-08 | Low | `print()` in library training paths (per-iter calibration print is worst) |
| A8-09 | P2 | Public/`_private` mostly clean; a few double-named public aliases |
| A8-10 | P2 | Return contract is bare `Tuple[Dict,Dict]`; LTR passthrough is a 2nd unverified shape |
| A8-11 | Low | Suite signature typed well; phase layer + ctx slots are all `Any` |
| A8-12 | P2 | Docstring documents <40% of the 34 params |
| A8-13 | OK-positive | Error handling discipline is strong (narrow excepts, surfaced context, raises not swallows) |
| A8-14 | P2 | Two remaining broad-except silent-fallback sites in the orchestrator |
| A8-15 | Low | `from typing import *` + dead `check_estimator` import in estimators/ |
| A8-16 | OK-positive | Optional-dep import guards are textbook-correct everywhere |
| A8-17 | P2 | `lgb_shim` eval-set normalization is a fragile heuristic stack reimplementing LGB input parsing |
| A8-18 | OK-positive | Live-cache estimators correctly exclude caches from pickle (`__getstate__`) |
| A8-19 | P2 | `_build_configs_from_params` silently drops unknown reporting kwargs |
| A8-20 | Low | `verbose` 3-level contract not honored (mostly used as boolean) |

**Highest-leverage two:** A8-04 (config dict-key silent typos) and A8-06 (finish the ctx migration / kill dual state). Both are root causes that the codebase's own comments and prior-bug archaeology already point at.
