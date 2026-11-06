# Dynamic code execution audit, wave 4 (mlframe, src/mlframe), 2026-11-06

Read-only. Baseline: audits/2026-10-31/dynamic_exec.md (EXE-01..29). Grep on HEAD for `eval(`, `exec(`, `__import__(`, `import_module(`, `subprocess.*`, `yaml.load(` outside `_benchmarks/`, then each hit traced to its inputs.

## Findings

| ID | Severity | File:line | Failure scenario | Disposition |
|---|---|---|---|---|
| DYN4-1 | Low | src/mlframe/training/composite/_estimator_dispatch.py:68-77 | EXE-10 is partly fixed on HEAD: private names are refused, and the target must be a class with `fit` (lines 73-76). It is still not an allow-list. If a recommendation dict is ever read from a config or a persisted file, it can name any public mlframe estimator class with a `fit` attribute, and that class is built with the caller's kwargs. Today the only producer builds the dict from literals, so this cannot be reached. | FUTURE: replace the module and estimator pair with a dict of the two allowed composite classes. The current guards are enough while the producer stays literal. |
| DYN4-2 | Info | src/mlframe/training/neural/_flat_torch_module/__init__.py:215-225 | EXE-18 is fixed: the module must be `torch.optim` or one of its submodules. The qualname is still walked with `getattr` part by part (lines 224-225). A crafted pickle could therefore reach a non-optimizer object re-exported inside `torch.optim`, but only stores it on `state["optimizer"]` and never calls it. The sink is pickle, which already executes code by design. | REJECTED: adds no capability beyond the pickle load itself. Optional: check `isinstance(_obj, type)` and `issubclass(_obj, torch.optim.Optimizer)`. |

Counts: High 0, Medium 0, Low 1, Info 1.

## Previously dispositioned, verified on HEAD

- EXE-19 is RESOLVED. `_sapp.py:29-33` `_distribution` refuses private names and anything that is not a scipy distribution object.
- EXE-20: the `df.eval` at `_sapp.py:68` still evaluates a fixed literal expression.

## Checked and safe

- `votenrank/__init__.py:58`, `utils/__init__.py:65`, `feature_selection/__init__.py:128`, `__init__.py:495`, `training/__init__.py:324`, `evaluation/__init__.py:102`, `data/datasets/__init__.py:80`: lazy `__getattr__` imports. Each one checks the name against a static dict first.
- `_orthogonal_meta_scorer_fe.py:496-500`: the module comes from the `_SCORER_DISPATCH` dict, and the function name is the stem with a fixed suffix.
- `_recipe_dispatch.py:41`, `_permutation_null_resident_ktc.py:113`, `_param_model_spec.py:48`, `preprocessing/opt_in_slow.py:58`: the module and attribute names are static or allow-listed.
- `estimators/pipelines.py:58`, `feature_engineering/mps.py:58`, `models/_optimization_shared.py:35`, `metrics/calibration/_calibration_plot.py:46`: `_LazyModule` with literal module names.
- `calibration/post.py:79`, `training/_polars_native_support.py:109`, `_setup_helpers_pipeline_cache.py:155`, `training/pipeline/__init__.py:68-76`, `calibration/__init__.py:73-89`, `metrics/_core_numba_warmup.py:395,812`, `financial.py:32`, `training/cb/__init__.py:25`, `composite/ranking.py:294`: literal module names.
- `system/kernel_tuning_cache/_ensure.py:197-199`: list argv with no shell. The kernel name comes from a registered spec.
- `utils/native_import_probe.py:55`, `training/_polars_native_support.py:90`, `training/_gpu_state_probe.py:98,109`: fixed argv, no shell.
- `kernel_tuning_cache/_discovery.py:81,91`: walks the installed package directory, as in EXE-14 (REJECTED).
- No production `eval(`, `exec(` or `yaml.load(` found. The `bases.py` and `_base_predict.py` matches are docstrings and comments.
