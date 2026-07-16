# MRMR Module Performance Audit — Findings

Covered all 8 files. `_mrmr_class.py` (270KB) was read via a background sub-agent in chunks; the rest read directly. Findings below are grouped by file, each with file:line, description, and scale. Prior findings already tracked in `filters/_benchmarks/mrmr_critique_2026_07/mrmr_crit_perf.md` (screen-copy, cpu_fe_batch_mi upcast, GPU kernel findings) are NOT re-listed.

## `_mrmr_class.py`

1. **Line ~3170** — `_x_fp = _mrmr_compute_x_fingerprint(X)` computed unconditionally on every `fit()` because `mrmr_skip_when_prior_was_identity` defaults to `True`. Full-content hash over X, O(n·p). Once per fit (not per iteration), but unconditional on the default path.

2. **Lines 3258-3269** — `X = X.copy(deep=False) if _cow_on else X.copy(deep=True)`. On pandas builds without Copy-on-Write enabled, this is a real deep copy of the whole (possibly SIS-reduced) frame purely for input-mutation isolation. O(n·p) alloc+memcpy, once per fit. In-file comment cites ~11ms at n=20k/p=299 as negligible, but that measurement doesn't cover the 100GB-class frames the project's memory-discipline convention targets.

3. **Lines 3064-3066** — `_fe_will_run` does `any(k.startswith("fe_") and k.endswith("_enable") and v for k, v in vars(self).items())`, scanning the FULL instance `__dict__` (hundreds of ctor params) with two string ops per entry, every `fit()` call, just to detect whether any FE flag is on. A class-level precomputed tuple of `fe_*_enable` attribute names + a fixed-list `getattr` loop would cut this from O(all attrs) to O(FE-flag count). Once per fit, currently O(ctor-param count).

4. **Lines 2942-2951** — `__setstate__` does `_fresh = type(self)()` — constructs an entire throwaway `MRMR()` instance (running the full ~300-param constructor, including the frame-reflection `store_params_in_object`) just to source ctor defaults for any pickled state missing keys. Once per unpickle event; matters when unpickling recurs (multi-worker CV, service reload). A cached ctor-defaults dict (see `_ctor_defaults()` in `_mrmr_class_config.py`, which does the same job via `inspect.signature` without instantiating) would avoid the full re-construction.

5. **Line 2849** — `store_params_in_object(obj=self, params=get_parent_func_args())` runs on every `MRMR()` construction, including every `sklearn.clone()` call. This recurs per stability-selection bootstrap replicate (`_stability_outer_fit`, default `stability_n_bootstrap=50`) and per `GridSearchCV`/`RFECV` clone. Reflection cost lives in `pyutilz.pythonlib` (out of this module), but the multiplier is real under heavy cloning/bootstrapping.

6. Scattered inline `from ... import ...` statements inside `fit()` body (15+ call sites, e.g. lines 2986, 2992, 2998, 3070, 3151, 3280, 3309-3314, 3344, 3376, 3408, 3482, 3494, 3554, 3560, 3562, 3638) — negligible in practice (cached in `sys.modules`, sub-microsecond each), listed for completeness only.

## `_mrmr_class_config.py`

7. **Lines 100-107 (`_apply_default_screen_subsample`) and 131-138 (`_apply_fast_search_profile`)** — each independently calls `inspect.signature(type(self).__init__).parameters` to rebuild the ctor-defaults dict, both from scratch, both run on every single `fit()` call (this is separate from `_ctor_defaults()` at line 268-280, a classmethod that does the *same* reflection again on the unpickle path). Introspecting a ~300-parameter `__init__` signature 2-3x per fit via `inspect.signature` is real, avoidable repeated work — caching the result once (e.g. as a `functools.lru_cache`'d classmethod, or the `_ctor_defaults()` result reused directly instead of two independent re-derivations) would remove duplicate reflection cost per fit call.

8. **Lines 50-71, 73-91** (`_fast_search_default_subsample_n`, `_default_screen_subsample_n`) — each does a `get_kernel_tuning_cache()` + `.lookup(...)` round-trip on every fit call rather than caching the resolved value on `self`/class for the process lifetime. Minor — depends on how cheap the KTC lookup itself is (not audited here), but it's an uncached per-fit call to an external cache lookup for a value that's host-constant.

## `_mrmr_class_fit_helpers.py`

9. **Lines 76-90** (`_inner_selector` inside `_stability_outer_fit`) — for each bootstrap/complementary-pair replicate (default `stability_n_bootstrap=50`), builds `self.get_params()` (a full dict of ~300 ctor params) and a dict comprehension filtering 5 keys out, then constructs a fresh sibling `MRMR` instance. This is inherent to the stability-selection algorithm (each replicate needs its own sub-fit) so not fixable without changing semantics, but the `get_params()`-then-filter dict-copy work is duplicated per replicate rather than computed once outside the loop and only overridden per-iteration.

10. **Lines 151-153** (`_maybe_resample_for_sample_weight`) — `rng.choice(n_rows, size=n_rows, replace=True, p=probs)` allocates a full-size index array plus the subsequent row-selection copy (`X.iloc[idx]` / `X[idx.tolist()]` / `np.asarray(X)[idx]`). Only triggers when a non-uniform `sample_weight` is passed (uniform short-circuits at line 145-146), so O(n) extra work once per such fit — flagged only for completeness, not avoidable given the resampling semantics.

11. **Lines 197, 205-208** (`_print_fit_summary`) — `prov[disp_cols].copy()` and a `.sort_values(...)` + `.map(...)` pass over the provenance frame, gated behind `verbose>=1` only. O(n_features), once per fit, negligible in absolute terms but worth noting it's an unconditional-when-verbose full copy rather than an in-place view.

## `_mrmr_class_shared.py`

No performance issues — `_mrmr_y_columns` is a thin O(n_output_columns) generator with no redundant work.

## `_mrmr_class_transform.py`

12. **Lines 96-102** (`get_feature_names_out`) — rebuilds `_adv_recipes` (a filtered list comprehension over `_engineered_recipes_`) and calls `simplified_recipe_names(...)` fresh on every call, including calls made purely for introspection (e.g. from `_append_usability_union` at line 186, and from any external caller checking column names repeatedly in a loop). Not cached across repeated calls on the same fitted instance. O(n_engineered_features) per call; matters only if a caller invokes `get_feature_names_out()` in a hot loop (e.g. per-batch inference wrapper) rather than once and reusing the result.

13. **Line 169** (`_usability_union_extra`) — `seen = set(map(str, base_names))` rebuilt from scratch on every call to both `get_feature_names_out` and `_append_usability_union`/`transform_usability`, duplicating the same set-construction work across sibling call sites within one `transform()` invocation. O(n_features), minor.

## `_mrmr_param_constants.py`, `_mrmr_setstate_defaults.py`, `__init__.py`

No performance issues found — these are pure literal-data modules (allow-list tuples, a legacy-pickle default dict deep-copied once per unpickle via `build_setstate_defaults()`, which is already the minimal correct approach) and a binding/re-export facade with no hot-path logic.

## Summary of the most actionable items
The two clearest "real, cheaply-fixable, repeated-per-fit" items are #7 (duplicate `inspect.signature` reflection on the ~300-param `__init__`, done 2-3x per fit across `_mrmr_class_config.py`) and #4 (`__setstate__` building a whole throwaway `MRMR()` instance for defaults instead of reusing `_ctor_defaults()`'s signature-only approach) — both are pure introspection/reflection overhead unrelated to the numeric MI/redundancy hot loops, so they wouldn't need njit/GPU work, just caching the reflected defaults dict once (class-level or `lru_cache`) instead of re-deriving it per call.

No O(n²)-over-features/samples loops, no missing MI/redundancy-matrix memoization, and no un-vectorized numeric hot loops were found in these 8 files themselves — the numeric MRMR kernels (screen/MI/redundancy computation) live in sibling modules (`_mrmr_fit_impl`, `_screen_predictors`, `_fe_cpu_batch`, GPU modules) already covered by the existing `mrmr_critique_2026_07` bench material, which this audit deliberately did not re-litigate.
