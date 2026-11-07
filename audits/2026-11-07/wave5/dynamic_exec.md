# Dynamic code execution audit, wave 5 (mlframe, src/mlframe), 2026-11-07

Read-only. Baselines: audits/2026-10-31/dynamic_exec.md (EXE-01..29), audits/2026-11-06/wave4/dynamic_exec.md (DYN4-1, DYN4-2). Grep on HEAD for `eval(`, `exec(`, `compile(`, `__import__(`, `import_module(`, `yaml.load(`, jinja/Template, and `getattr(<module>, <variable>)` outside `_benchmarks/`; each hit traced to its inputs.

## Findings

| ID | Severity | File:line | Failure scenario | Disposition |
|---|---|---|---|---|
| DYN5-1 | Low | src/mlframe/feature_engineering/windowed_edge_diff.py:73; src/mlframe/feature_engineering/cross_sectional_neighbors.py:59-61 | `reducer = getattr(np, agg)` / `getattr(np, stat)` resolve any public numpy attribute from a caller string, then call it on a float array (with `axis=1` in the neighbors case). If `agg`/`agg_stats` ever come from a config file, a value such as `"save"` or `"load"` reaches a numpy I/O function instead of a reducer; today it only fails with a confusing TypeError (no file path argument), and a typo is detected only after the per-entity grouping work. No code-execution path exists because numpy has no eval-like callable reachable with one array argument. | FUTURE: replace with a dict allow-list (`mean, median, sum, min, max, std, var` and nan variants), validated at the top of the function. |
| DYN5-2 | Info | src/mlframe/training/composite/_estimator_dispatch.py:68-79 | DYN4-1 / EXE-10 is now RESOLVED on HEAD: module must be `mlframe` or `mlframe.*`, private names refused, and the target must be a class subclassing `sklearn.base.BaseEstimator` whose `__module__` is inside mlframe. No remaining failure scenario. | REJECTED (already fixed; tracker row can be closed as RESOLVED). |
| DYN5-3 | Info | src/mlframe/training/feature_handling/hf_provider.py:230; providers.py:213 | `trust_remote_code` can be switched on from a `hf://model?trust_remote_code=true` URI string, so a URI taken from a config file executes the hub repo's `modeling.py`. It is parsed strictly (hf_provider.py:131-144, `"False"` no longer truthy) and warns. Already dispositioned in the network/supply-chain audits (2026-10-31/network.md, 2026-11-05 wave1/wave2). | REJECTED here: explicit opt-in by design, owned by the network audit. |

Counts: High 0, Medium 0, Low 1, Info 2.

## Checked and safe (verified on HEAD)

- DYN4-2: `_flat_torch_module/__init__.py:215-225` unchanged; pickle-only sink, stays REJECTED.
- `calibration/post.py:79` `_try_import_class`: called with literal module/class pairs only.
- `training/_polars_native_support.py:109`: `library` comes from callers passing "catboost"/"lightgbm" literals (`accepts_polars`, line 115-122).
- `data/datasets/_sapp.py:31`: `getattr(stats, ...)` refuses private names and non-distributions (EXE-19 RESOLVED); `_sapp.py:68` `df.eval` uses a literal expression.
- `data/synthetic.py:104`: `dist_name` is drawn from the module's internal `cats`/`conts` lists (line 91), never from the caller.
- `feature_selection/filters/_usability_greedy_clf_host_batched.py:35`: internal name only.
- `_estimator_dispatch.py`, `_recipe_dispatch.py:41`, `_permutation_null_resident_ktc.py:113`, `_orthogonal_meta_scorer_fe.py:499`, `_param_model_spec.py:48`, `preprocessing/opt_in_slow.py:58`: static or dict allow-listed names.
- Lazy `__getattr__` loaders (`__init__.py:495`, `training/__init__.py:324`, `feature_selection/__init__.py:128`, `utils/__init__.py:65`, `votenrank/__init__.py:58`, `evaluation/__init__.py:102`, `data/datasets/__init__.py:80`): name checked against a static dict first.
- `_LazyModule` users (`estimators/pipelines.py:58`, `feature_engineering/mps.py:58`, `models/_optimization_shared.py:35`, `metrics/calibration/_calibration_plot.py:46`) and literal imports (`calibration/__init__.py:73-89`, `financial.py:32`, `training/cb/__init__.py:25`, `composite/ranking.py:294`, `metrics/_core_numba_warmup.py:395,812`, `training/pipeline/__init__.py:68-76`, `_setup_helpers_pipeline_cache.py:155`, `_optional_imports.py:12`).
- `kernel_tuning_cache/_discovery.py:81,91`: installed-package walk (EXE-14 REJECTED).
- `_neural_mi.py:493`: `from_pretrained` with a pinned revision from `MIST_PINNED_REVISIONS`, no `trust_remote_code`.
- No production `eval(`, `exec(`, builtin `compile(` or `yaml.load(`; remaining `.eval()` hits are torch `Module.eval()` / polars `list.eval` / docstrings.
