# Dynamic code execution audit (mlframe, src/mlframe)

Scope: `eval(`, `exec(`, `compile(`, `__import__(`, `importlib.import_module(`, `getattr` used to call code, and `ast.literal_eval`.
Read-only audit. No code or git state was changed.

## Summary

- No `eval()` on any runtime string in production code. The only `exec(compile(...))` calls are in `_benchmarks` modules and load source from `git show HEAD:`.
- Production dynamic imports (`__import__`, `importlib.import_module`) are gated by static dispatch dicts, literal call sites, or an `mlframe.` prefix check. None takes a value that a remote or unauthenticated caller controls.
- The one real hardening gap is `instantiate_recommended_estimator` (EXE-10): the prefix check is present, but the class name is not allow-listed.
- Deserialization sinks (`pickle` / `joblib` / `dill` loads, about 25 files) execute code by design and belong to a separate deserialization audit. EXE-18 is one such sink that also performs a dynamic import.
- Counts: High 0, Medium 0, Low 10, Info 19. Total 29.

## Findings table

| ID | File:line | Severity | Proposed status |
|---|---|---|---|
| EXE-01 | src/mlframe/calibration/post.py:79-80 | Info | RESOLVED |
| EXE-02 | src/mlframe/calibration/post.py:191, 233, 235, 239, 281 | Info | REJECTED |
| EXE-03 | src/mlframe/data/datasets/__init__.py:80-81 | Info | RESOLVED |
| EXE-04 | src/mlframe/estimators/pipelines.py:18 | Info | DOC |
| EXE-05 | src/mlframe/feature_selection/filters/engineered_recipes/_recipe_dispatch.py:26 | Info | RESOLVED |
| EXE-06 | src/mlframe/feature_selection/filters/_orthogonal_meta_scorer_fe.py:16-17 | Info | RESOLVED |
| EXE-07 | src/mlframe/feature_selection/filters/_permutation_null_resident_ktc.py:113 | Info | RESOLVED |
| EXE-08 | src/mlframe/feature_selection/__init__.py:128 | Info | RESOLVED |
| EXE-09 | src/mlframe/training/_param_model_spec.py:48 | Info | RESOLVED |
| EXE-10 | src/mlframe/training/composite/_estimator_dispatch.py:70-72 | Low | FUTURE |
| EXE-11 | src/mlframe/training/__init__.py:324-325 | Info | RESOLVED |
| EXE-12 | src/mlframe/training/_polars_native_support.py:109 | Info | RESOLVED |
| EXE-13 | src/mlframe/training/core/_setup_helpers_pipeline_cache.py:155-156 | Info | RESOLVED |
| EXE-14 | src/mlframe/system/kernel_tuning_cache/_discovery.py:81, 91 | Info | REJECTED |
| EXE-15 | src/mlframe/preprocessing/opt_in_slow.py:58-59 | Info | RESOLVED |
| EXE-16 | src/mlframe/_optional_imports.py:12 | Info | RESOLVED |
| EXE-17 | src/mlframe/__init__.py:482 | Info | RESOLVED |
| EXE-18 | src/mlframe/training/neural/_flat_torch_module/__init__.py:216-217 | Low | FUTURE |
| EXE-19 | src/mlframe/data/datasets/_sapp.py:17 | Low | FUTURE |
| EXE-20 | src/mlframe/data/datasets/_sapp.py:21 | Info | RESOLVED |
| EXE-21 | src/mlframe/feature_engineering/_benchmarks/bench_anchor_slope_cache.py:36 | Low | DOC |
| EXE-22 | src/mlframe/feature_selection/filters/_benchmarks/bench_conditional_permutation_strata_arrs.py:45 | Low | DOC |
| EXE-23 | src/mlframe/feature_selection/filters/_benchmarks/bench_ratio_delta_fe.py:49 | Low | DOC |
| EXE-24 | src/mlframe/feature_selection/filters/_benchmarks/bench_target_encoding_apply.py:36 | Low | DOC |
| EXE-25 | src/mlframe/feature_selection/_benchmarks/bench_pairwise_modular_cost.py:71 | Low | DOC |
| EXE-26 | src/mlframe/training/composite/_benchmarks/bench_venn_abers_ivap_envelopes.py:28 | Low | DOC |
| EXE-27 | src/mlframe/feature_selection/_benchmarks/profiling/duplicate_d2h.py:27; fit_wall.py:32 | Info | RESOLVED |
| EXE-28 | src/mlframe/feature_selection/_benchmarks/profile_mrmr_line_profile.py:58-59 | Low | REJECTED |
| EXE-29 | Benchmark and profiling importlib sites: `_benchmarks/nvprof_autotag.py:44`, `fs_hybrid/_arms.py:86`, `fs_hybrid/_matched_k.py:56`, `bench_mrmr.py:85` | Info | DOC |

## Details

### Production dynamic imports

**EXE-01, calibration/post.py:79-80.** `_try_import_class(module_path, class_name)` calls `__import__` and then `getattr`. The live call sites are at lines 143 (driven by `_NEEDS_2D_CALIBRATORS`) and 154 (literal `"venn_abers"`, `"VennAbersCalibrator"`). I did not inspect the definition of `_NEEDS_2D_CALIBRATORS`. Both values are module-level constants, not user input. Severity Info. Disposition RESOLVED. Keep the callers literal, or switch to an explicit dict of classes.

**EXE-02, calibration/post.py:191, 233, 235, 239, 281.** `getattr(self.calibrator, self.fit_method_name)` and `getattr(self.calibrator, transform_name)`. The method name comes from the estimator's constructor config. This selects a method on an object the caller already built, so it cannot reach arbitrary code. Severity Info. Disposition REJECTED.

**EXE-03, data/datasets/__init__.py:80-81.** Module `__getattr__` runs `__import__(f"{__name__}.{home}")`. The `name` is first checked against `_LAZY_HOME`, so `home` can only be one of the static values in that dict. Severity Info. Disposition RESOLVED.

**EXE-04, estimators/pipelines.py:18.** `_LazyModule` calls `importlib.import_module(self._lm_name)`. The name is a constructor argument. All visible uses pass literal module paths such as matplotlib. Severity Info. Disposition DOC. Note in the class docstring that `name` must be a literal internal module path.

**EXE-05, engineered_recipes/_recipe_dispatch.py:26.** `_routed(module, func)` builds a handler from a static table (lines 219-273). The handler calls `getattr(importlib.import_module(module, __package__), func)`. The lookup key is the recipe `kind`, which comes from the persisted recipe. An unknown kind is refused by the dict, so a crafted recipe cannot choose a module or function name. Severity Info. Disposition RESOLVED. Keep the dict as the only route.

**EXE-06, _orthogonal_meta_scorer_fe.py:12-17.** The scorer name is lowercased and then looked up in `_SCORER_DISPATCH.get`. `None` is returned for unknown names, so `importlib.import_module` only runs with a dict-provided module name. Severity Info. Disposition RESOLVED.

**EXE-07, _permutation_null_resident_ktc.py:113.** `_permnull_variant(module, attr, doc)` is called with literal arguments in the module. Severity Info. Disposition RESOLVED.

**EXE-08, feature_selection/__init__.py:128.** `_LAZY_EXPORTS` is checked before the import. Severity Info. Disposition RESOLVED.

**EXE-09, training/_param_model_spec.py:48.** `SelectorSpec.target()` calls `getattr(importlib.import_module(self.target_module), self.target_attr)`. Both fields come from a frozen dataclass registry of literals. Severity Info. Disposition RESOLVED.

**EXE-10, training/composite/_estimator_dispatch.py:70-72.** `instantiate_recommended_estimator(recommendation, **kwargs)` imports `recommendation["module"]` after a check that it is `mlframe` or starts with `mlframe.`. It then runs `getattr(mod, recommendation["estimator"])` with no allow-list, and instantiates the result with caller kwargs.
- Attack path: a caller-supplied or persisted recommendation dict naming an `mlframe` module attribute that is a callable with side effects. Any module in the package is importable, and `getattr` can reach any name in it, including names imported into those modules (for example `os`).
- Today the recommendation dicts are built from literals at lines 37-45. The risk therefore only exists if a recommendation is ever read from user input or from a file.
- Severity Low. Disposition FUTURE. Fix: replace the `(module, estimator)` pair with a fixed dispatch dict of allowed classes, for example `{"TailCompositeEstimator": TailCompositeEstimator, "CompositeDistributionEstimator": CompositeDistributionEstimator}`. The existing prefix check can stay as a second guard.

**EXE-11, training/__init__.py:324-325.** `_LAZY_IMPORTS` is checked first, and the module path is built from a static tuple. Severity Info. Disposition RESOLVED.

**EXE-12, training/_polars_native_support.py:109.** `_version(library)` runs `__import__(library)`. The callers in this file pass literal library names. Only the local callers were checked. Severity Info. Disposition RESOLVED.

**EXE-13, training/core/_setup_helpers_pipeline_cache.py:155-156.** The loop variable is a literal tuple `("polars_ds", "polars")`. Severity Info. Disposition RESOLVED.

**EXE-14, system/kernel_tuning_cache/_discovery.py:81, 91.** Module names come from an `os.walk` over the installed package directory, then `import_module` runs on each file that registers a tuner. Anyone who can plant a `.py` file under the package directory can already run code, so this adds no exposure. Severity Info. Disposition REJECTED.

**EXE-15, preprocessing/opt_in_slow.py:58-59.** The name is checked against `OPT_IN_SLOW` (KeyError) and the caller's enabled list (PermissionError) before the import. Severity Info. Disposition RESOLVED.

**EXE-16, _optional_imports.py:12.** `import_optional(module, extra, purpose)` is called with literal names at quality.py:93, reports.py:426, boruta_shap/_fit_explain.py:635, and boruta_shap/_io_plot.py:123. Severity Info. Disposition RESOLVED.

**EXE-17, mlframe/__init__.py:482.** `_LAZY_TOPLEVEL` is checked first. Severity Info. Disposition RESOLVED.

**EXE-18, training/neural/_flat_torch_module/__init__.py:216-217.** `__setstate__` reads `("__optimizer_class_ref__", module, qualname)` from the unpickled state and calls `importlib.import_module(module)`, then fetches the attribute.
- The sink is pickle. Loading an untrusted model file already runs arbitrary code, so this import adds nothing beyond that.
- It is still a dynamic import driven by data inside the file, so it is recorded here. Severity Low. Disposition FUTURE. Fix: check that the module is `torch.optim` (or an allow-list of optimizer modules) before importing. Also document that model files are trusted input (see the deserialization audit).

### Eval-like and literal-eval sites

**EXE-19, data/datasets/_sapp.py:17.** `getattr(stats, distr_name).rvs(...)`. `distr_name` is an API argument. The attribute is looked up on `scipy.stats`, and `rvs` is then called. This is not code execution, but any `scipy.stats` attribute with an `rvs` method can be selected. Severity Low. Disposition FUTURE. Fix: allow-list of distribution names, for example `{"norm", "uniform", "expon", ...}`.

**EXE-20, data/datasets/_sapp.py:21.** `df.eval("X1+sin(X2)+...")`. This is a pandas query expression on a fixed literal. The `@` references are local variables, and no user input is interpolated. This is not Python `eval`. Severity Info. Disposition RESOLVED.

### Exec of git-history source in benchmarks

**EXE-21 to EXE-26.** Each of the six files below runs `exec(compile(src, name, "exec"), ns)` on source read with `git show HEAD:<path>`, and each has a `# nosec B102` comment that says so:
- EXE-21: `feature_engineering/_benchmarks/bench_anchor_slope_cache.py:36`
- EXE-22: `feature_selection/filters/_benchmarks/bench_conditional_permutation_strata_arrs.py:45`
- EXE-23: `feature_selection/filters/_benchmarks/bench_ratio_delta_fe.py:49`
- EXE-24: `feature_selection/filters/_benchmarks/bench_target_encoding_apply.py:36`
- EXE-25: `feature_selection/_benchmarks/bench_pairwise_modular_cost.py:71`
- EXE-26: `training/composite/_benchmarks/bench_venn_abers_ivap_envelopes.py:28`

The source is the repository's own committed code. An attacker would need push access to the repo, and at that point they already control the code. None of these files is imported by the library. Severity Low. Disposition DOC. Proposed text: "Benchmark-only; executes committed source from git HEAD; not reachable from library code or user input." Optional follow-up: switch to `importlib` on a file path, which would avoid `exec` entirely.

**EXE-27, profiling/duplicate_d2h.py:27 and profiling/fit_wall.py:32.** `ast.literal_eval(v)` on `key=value` command-line arguments. This is the correct use of `literal_eval`: it accepts only literals. Severity Info. Disposition RESOLVED.

**EXE-28, feature_selection/_benchmarks/profile_mrmr_line_profile.py:58-59.** `_resolve_target(sys.argv[1])` imports whatever dotted module path the operator passes, then fetches the attribute. This is an operator-run developer CLI. Anyone who can pass `argv` can already run Python. Severity Low. Disposition REJECTED. Reason: the operator already has code execution.

**EXE-29, benchmark and profiling importlib sites.** `profiling/nvprof_autotag.py:44` (generates an injected code template), `fs_hybrid/_arms.py:86`, `fs_hybrid/_matched_k.py:56` (fixed f-string with `__package__`), and `feature_selection/_benchmarks/bench_mrmr.py:85`. These are benchmark-only, and I did not trace the value source of each call in full. Severity Info. Disposition DOC. Proposed text: benchmark-only, not reachable from library code. Before the next audit round, trace the `nvprof_autotag.py` template fields and `fs_hybrid/_arms.py` module names to confirm they are literals.

### Out of scope (cross-reference)

- `estimators/pipelines.py` `_safe_pickle_verify_sidecar` and about 25 files that load pickle, joblib, or dill. These are code-execution sinks by design and belong to the deserialization audit class.
- `re.compile`, `torch .eval()` / `.compile()` methods, `pl.list.eval`, and `_recipe_dispatch` on its own are not Python dynamic execution and were excluded from the counts.
