## MRMR Module Code-Quality / Design Audit

Read in full: `__init__.py` (253 LOC), `_mrmr_class.py` (3706 LOC), `_mrmr_class_config.py` (386), `_mrmr_class_fit_helpers.py` (362), `_mrmr_class_shared.py` (25), `_mrmr_class_transform.py` (229), `_mrmr_param_constants.py` (99), `_mrmr_setstate_defaults.py` (394). Total 5454 LOC.

### File-size / module-boundary

1. **`_mrmr_class.py:194-2802` — `__init__` signature alone is ~2600 lines.** The constructor's parameter list (with inline comments/rationale) runs from line 194 to the closing `):` at line 2802 — over 250 keyword parameters, each with a multi-line justification comment embedded directly in the signature. The actual constructor *body* (lines 2803-2851) is only ~48 lines. This is by far the module's biggest violation of the "split before ~800-900 LOC" convention, and it can't be fixed by carving the surrounding module (the split into `_mrmr_class_config.py` / `_mrmr_class_fit_helpers.py` / `_mrmr_class_transform.py` mixins was a good move, but it left the mother-of-all-constructors untouched). Splitting the parameter block into cohesive dataclasses/config objects (there is already a precedent — `cat_fe_config` bundles ~22 knobs into one `CatFEConfig`) is the obvious fix; instead the module has grown ~250 more scalar knobs directly on `__init__`, all validated ad hoc.

2. **`_mrmr_class.py:2955-3688` — `fit()` is ~730 lines**, a single method mixing: GPU circuit-breaker re-arm, polars validation, polars→pandas FE bridge, accuracy-caveat warnings, stability-selection dispatch, y NaN/Inf validation, multi-output dispatch, degenerate-column diagnostics, cross-target identity-cache lookup/store, sample-weight resampling, CoW-aware frame copy, SIS front-gate dispatch, `fe_auto` flag resolution, 8 different MI-related thread-local toggles (SU/JMIM/BUR/Miller-Madow/RelaxMRMR/PID/CMI-perm/CPT), group-aware MI setup, DCD activation, fast-search profile application, default-screen-subsample application, seed/skip_content alias reconciliation, the actual `_fit_impl` call, provenance recording, FE provenance/rejection-ledger population, and a `finally` block that restores ~15 pieces of mutated global/thread-local/instance state. This is a textbook case for decomposition into named helper functions (many already exist as one-liners elsewhere in the module — e.g. `_apply_fast_search_profile`, `_apply_default_screen_subsample` — but the majority of this orchestration is inlined directly in `fit`). Why it matters: at this size, `fit` is unreviewable as one unit and each `finally`-block restore is a silent correctness dependency nobody can audit at a glance.

3. **`_mrmr_class.py:3567-3661` — 11 near-identical `try/except Exception as e: logger.debug("suppressed in _mrmr_class.py:<N>: %s", e); pass` blocks** in `fit`'s `finally` clause (lines ~3573, 3578, 3583, 3588, 3609, 3614, 3630, 3651, 3659, plus two more). Two problems:
   - Duplicated logic that should be one helper, e.g. `_safe_restore(callable)`, called 11 times.
   - The log messages hardcode source line numbers (`"suppressed in _mrmr_class.py:3574"`, `:3578`, `:3582`...) that are **already off by 1** from their actual current lines (e.g. the string says `:3574` but the `try` is on line 3571 / the `except` on 3573) and will silently drift further out of sync on every future edit — a maintenance trap that gives false diagnostic information forever once it first goes stale.

### Type-annotation gaps (would mypy --strict flag)

4. **`_mrmr_class.py:2955` — `fit(self, X, y, groups: pd.Series | np.ndarray = None, ...)`.** `groups` is typed `pd.Series | np.ndarray` but defaults to `None` — a bare implicit-Optional, exactly the pattern the project's own CLAUDE.md calls out as CRITICAL ("never `param: T = None`; always `Optional[T]`"). Should be `pd.Series | np.ndarray | None = None`.

5. **`_mrmr_class.py:1029` — `cv: object | int | None = 3`.** `object` used as part of a union for a parameter that (per its own docstring: "int, cross-validation generator or an iterable") has a real concrete type (`int | BaseCrossValidator | Iterable | None`, matching sklearn's own `cv` typing). This is precisely the "no bare `object` type for params that have a concrete type" violation called out in the project conventions.

6. **`_mrmr_class_config.py:30-37` — `_MRMRConfigMixin` declares `_FIT_CACHE: Any`, `_FAST_SEARCH_OVERRIDES: Any`, `random_seed: Any`, `random_state: Any`, `verbose: Any`, `cv: Any`, `cv_shuffle: Any`.** These are all typed `Any` purely so mypy resolves the attribute at all via the mixin's MRO; the concrete types are documented in prose in the class docstring ("`_FIT_CACHE` ... class attribute ... on MRMR", etc.) but never expressed as types. `_FIT_CACHE` is concretely `OrderedDict[tuple, MRMR]` (as declared on `MRMR` itself, `_mrmr_class.py:164`), `random_seed`/`random_state` are `int | None`, `verbose` is `bool | int`, `cv_shuffle` is `bool` — all knowable from `_mrmr_class.py`'s own `__init__` signature. Using `Any` here defeats the purpose of the mixin's type declarations and would not catch a caller passing the wrong type into any of these.

7. **`_mrmr_class_fit_helpers.py:33-41` — the `TYPE_CHECKING` stub block declares `get_params: Any`, `fit: Any`, `random_seed: Any`, `get_feature_names_out: Any`.** Same pattern — concrete sklearn/`MRMR` signatures exist (`get_params(self, deep=True) -> dict`, `fit(...) -> MRMR`, `get_feature_names_out(self, input_features=None) -> np.ndarray`) but are collapsed to `Any`, silently defeating static checking of every call site that uses these through the mixin.

8. **`_mrmr_class_fit_helpers.py:124` — `_maybe_resample_for_sample_weight(self, X, y, sample_weight: np.ndarray | None)`** has no return type annotation despite always returning a `tuple` (`return X, y` / `return X_rs, y_rs`). Minor but inconsistent with sibling methods in the same file that do annotate returns (`export_artifacts(self) -> dict`, `_fit_identity_shortcut(self, X) -> None`).

9. **`_mrmr_class_fit_helpers.py:321` — `_fit_multioutput(self, X, y, groups, sample_weight, strategy: str, fit_params)`** — `X`, `y`, `groups`, `sample_weight`, `fit_params` are all unannotated even though the surrounding module (and `fit()` itself) has concrete unions available for every one of them (`pd.DataFrame | np.ndarray | Any`, `pd.Series | np.ndarray | None`, etc.). No return type either, despite always `return self`.

10. **`_mrmr_class_shared.py:13` — `def _mrmr_y_columns(y):`** — no parameter or return type annotation at all (it's a generator; should be `Iterator[tuple[str, np.ndarray]]`). Small function, but it's imported and reused from three different files (`_mrmr_class.py`, `_mrmr_class_fit_helpers.py`, and re-exported from `__init__.py`), so an incorrect call site anywhere would not be caught.

11. **`_mrmr_class.py:194` `quantization_dtype: type = np.int32`** and **`dtype=np.int32`** (line 320, no annotation at all on `dtype`) — inconsistent: one dtype-like parameter is typed `type` (itself a very loose mypy type — `type[Any]`), the sibling `dtype` parameter two lines later has no annotation whatsoever.

12. **`_mrmr_class.py:2852` — `def __repr__(self, N_CHAR_MAX: int = 700) -> str:`** ends with `return str(r)` where `r` is already the `str` returned by `super().__repr__(...)`; the wrapping `str(...)` is a no-op that masks the fact that `r` is reassigned via string slicing (`r = f"{_inner}{_sep}n_workers=...)"`) inside the `if`, so the final `str()` call obscures rather than fixes any type ambiguity — harmless but a "pointless extra wrapping" pattern the project's own conventions warn against ("never silence an error with pointless extra wrapping").

### Dead / misleading code

13. **`_mrmr_class_config.py:175-202` — `recommend_default_scorer()` docstring is broken prose**: "``'s 7-dataset x 10-mechanism showdown placed CMIM..." — the subject of the sentence (presumably a benchmark name or author) was dropped, leaving a docstring that opens mid-sentence with `'s`. Same paragraph later: "accelerated JMIM (~2.3x) and TC (~5.0x) via batched quantile binning..." also opens with a dangling verb with no subject. This reads as an editing artifact (text deleted mid-refactor) rather than an intentional style choice, and actively confuses a doc reader about what "showdown" or "L86" refers to.

14. **`_mrmr_class_transform.py:21-23` — `transform(self, X): raise NotImplementedError`** on the mixin, with a comment saying the concrete class overrides it "so calling it on the mixin directly is a programming error." This is fine as a documented placeholder, but note it duplicates the exact same disclaimer already stated 3 times across `_mrmr_class.py:3679-3701` (`_fit_impl`/`_run_fe_step`/`transform` bottom-of-module comments) and the module docstring at the top of `_mrmr_class_transform.py` itself — the "why is this here and not there" explanation is repeated near-verbatim in at least 4 places across the module rather than stated once and referenced.

### Naming / API-design smells

15. **Inconsistent parameter naming for "the same knob" across the package**: `skip_retraining_on_same_content` (current) vs. the deprecated alias `skip_retraining_on_same_shape` — both accepted simultaneously as `__init__` params, with the reconciliation logic split across three places: a `DeprecationWarning` in `__init__` (`_mrmr_class.py:2840-2846`), a lazy resolution inside `fit()` (`_mrmr_class.py:3457-3463`), and a legacy-pickle override entry in `_mrmr_setstate_defaults.py`. A caller reading only `__init__`'s docstring would not learn that the "effective" value is actually resolved later, in `fit`, not at construction.

16. **`n_workers` vs `n_jobs`** (`_mrmr_class.py:334`, `1039`) are two independently-meaning parallelism knobs on the same estimator with overlapping names and no shared prefix or naming convention distinguishing "this drives the candidate-MI screen loop" (`n_workers`) from "this drives CPU sub-helpers" (`n_jobs`). The class's own `__repr__` override (`_mrmr_class.py:2852-2862`) exists *specifically* because users misread `n_jobs` as controlling the whole fit and never notice `n_workers` (sklearn's default repr hides params at their default value) — i.e. the module itself documents that this naming is confusing enough to need a workaround, rather than renaming one of the two knobs.

17. **`random_seed` vs `random_state`** — both are accepted, independently stored (sklearn `get_params` contract), and reconciled at fit time via `_effective_random_seed()`. This dual-name pattern (also seen in #15/#16) recurs three times in this module for different features, suggesting a systemic naming-drift problem across the parameter surface rather than three independent incidents.

18. **`_mrmr_class_transform.py:206` — `transform_usability(self, X, which: str = "linear")`** takes a bare `str` for `which` where only 3 literal values are valid (`'linear'|'universal'|'nonlinear'`) and are validated with an ad hoc `dict.get(...)` + `raise ValueError` at runtime rather than `Literal["linear", "universal", "nonlinear"]`, which would let mypy catch a typo'd caller at type-check time instead of only at runtime. This is inconsistent with the module's own `_mrmr_param_constants.py`, which documents *why* it avoided `Literal` for the ctor-level string params (wanting a richer runtime error listing) — but `transform_usability`'s error message is not richer than a `Literal` type error would be, so the same rationale doesn't actually apply here.

### API-design / sklearn-estimator-API concerns

19. **`MRMR` inherits `TransformerMixin` but not `SelectorMixin`**, with an explicit comment (`_mrmr_class.py:99-100`) explaining the choice ("transform can add engineered features... not a pure mask-based selector"). That's a reasonable design call, but it means `get_support`/`get_feature_names_out` are hand-rolled reimplementations of `SelectorMixin`'s contract (`_mrmr_class_transform.py:117-173`) rather than inherited — any future sklearn `SelectorMixin` contract change (e.g. an additional required method) has to be manually mirrored here and won't be caught by an `isinstance(_, SelectorMixin)` check elsewhere in the codebase or by sklearn's own selector test suite.

20. **`groups` parameter on `fit()` is accepted but silently ignored** unless `group_aware_mi=True` (a warning is emitted, or a `NotImplementedError` if `strict_groups=True`). This is documented at length in the docstring, but it is an sklearn API violation in spirit — the `fit(X, y, groups=...)` signature exists specifically for `GroupKFold`-style pipelines to route this data through, and a mask-only warning (rather than raising by default) means a Pipeline silently discards the caller's group information unless the caller separately knows to flip `strict_groups=True`. The module's own docstring acknowledges the asymmetry with `sample_weight` (which *is* consumed) at `_mrmr_class.py:2975`, i.e. the authors are aware this is inconsistent but have not resolved it.

### Duplicated logic within the module

21. **Ctor-default resolution duplicated 3 ways** across `_mrmr_class_config.py:_ctor_defaults()`, `_mrmr_setstate_defaults.py`'s hand-maintained `_SETSTATE_LEGACY_DEFAULTS` dict (with a "D5 no-drift overlay" mechanism specifically built to reconcile the two), and a third fallback in `__setstate__` (`_mrmr_class.py:2942-2951`) that constructs `type(self)()` fresh and reads every remaining ctor default off the live instance. Three separate mechanisms exist to solve "don't let a hardcoded default drift from the real ctor default" — a sign the underlying problem (defaults living in 3 different places: signature, legacy dict, and instance) should be solved once (e.g. defaults sourced from the signature everywhere, with an explicit override table for the documented legacy exceptions only) rather than patched three times.

22. **The `_apply_fast_search_profile` / `_apply_default_screen_subsample` pattern** (`_mrmr_class_config.py:93-173`) — "read ctor defaults via `inspect.signature`, only override an attr if it still equals its package default, save old value for restore" — is implemented twice, nearly identically, for two different knob sets (fast-search overrides and screen-subsample). Both duplicate the same `inspect.signature(type(self).__init__)` introspection and the same "only touch untouched knobs" guard logic; this could be one parametrized helper taking `(attrs, values)` instead of two independent ~40-line methods.

### Summary table

| # | File:Line | Issue | Severity |
|---|---|---|---|
|1|`_mrmr_class.py:194-2802`|`__init__` signature is ~2600 LOC|High — module-boundary|
|2|`_mrmr_class.py:2955-3688`|`fit()` is ~730 LOC, does 20+ concerns|High — decomposition|
|3|`_mrmr_class.py:3567-3661`|11 duplicated try/except/log/pass blocks with stale hardcoded line numbers|Medium|
|4|`_mrmr_class.py:2955`|`groups` implicit-Optional|Medium — mypy|
|5|`_mrmr_class.py:1029`|`cv: object \| int \| None`|Low-Medium — mypy|
|6|`_mrmr_class_config.py:30-37`|mixin attrs typed `Any` instead of concrete|Medium — mypy|
|7|`_mrmr_class_fit_helpers.py:33-41`|`TYPE_CHECKING` stubs typed `Any`|Medium — mypy|
|8|`_mrmr_class_fit_helpers.py:124`|missing return-type annotation|Low|
|9|`_mrmr_class_fit_helpers.py:321`|multiple missing annotations|Low-Medium|
|10|`_mrmr_class_shared.py:13`|no annotations on a 3x-reused shared helper|Low-Medium|
|11|`_mrmr_class.py:194,320`|inconsistent dtype-param typing|Low|
|12|`_mrmr_class.py:2852-2862`|pointless `str(r)` wrap|Low|
|13|`_mrmr_class_config.py:175-202`|broken/dangling docstring prose|Low|
|14|`_mrmr_class_transform.py:21-23`|same disclaimer repeated 4x across module|Low|
|15|`_mrmr_class.py` (multiple)|`skip_retraining_on_same_content`/`_shape` split logic across 3 sites|Medium|
|16|`_mrmr_class.py:334,1039`|`n_workers` vs `n_jobs` naming confusion (self-acknowledged)|Medium|
|17|`_mrmr_class.py` (multiple)|`random_seed`/`random_state` dual-name pattern|Low-Medium|
|18|`_mrmr_class_transform.py:206`|`which: str` not `Literal`|Low|
|19|`_mrmr_class.py:99-100`|hand-rolled `SelectorMixin`-like API instead of inheriting it|Low-Medium|
|20|`_mrmr_class.py:2968-2975`|`groups` silently dropped by default (self-acknowledged asymmetry)|Medium|
|21|3 files|ctor-default resolution duplicated 3 ways|Medium|
|22|`_mrmr_class_config.py:93-173`|near-duplicate override/restore helpers|Low-Medium|

No correctness bugs or performance issues are included above per the task scope — those are left to other agents.
