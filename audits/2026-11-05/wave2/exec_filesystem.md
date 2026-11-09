# Wave 2: dynamic execution and filesystem audit (src/mlframe)

Scope: eval/exec/compile, `__import__`/`importlib` on variable names, subprocess, tempfile, path joins, `shutil.rmtree`/`os.remove`.
Method: grep over `src/mlframe`, then read each production (non-`_benchmarks`) candidate's code path.

**Result: no High or Medium findings.** Every reachable dynamic-import, subprocess and delete site read has an allow-list, a fixed argument, or a key validator in front of it.
Prior rounds (EXE-10, DYN2-1/2, FS2-1) appear correctly closed in the tree.

## Findings

### EXEC-W2-1 (Info) `src/mlframe/utils/private_temp.py:31-41`
- Statement: on Windows (`euid is None`) the ownership/mode check is skipped, so the private-dir guarantee rests only on `%TEMP%` being per-user.
- Scenario: a deployment that points `TMP`/`TEMP` at a shared directory (e.g. `C:\Temp`) lets another local user pre-create `<name>-<USERNAME>` and the cache is used as-is.
- Disposition: DOC (state the Windows assumption in the module docstring); not RESOLVED-needed because the default per-user `%TEMP%` is not shared.

### EXEC-W2-2 (Info) `src/mlframe/utils/private_temp.py:42-44`
- Statement: when the fixed directory is not private, each call returns a NEW `mkdtemp` directory that is never removed.
- Scenario: an attacker-owned or loose-mode dir on a POSIX host makes every cache open create and leak a fresh temp dir (disk growth, cache never hits). Safe, but a silent slow path.
- Disposition: DOC, or memoize the fallback path per `name` for the process lifetime (cheap fix).

### EXEC-W2-3 (Info) `src/mlframe/reporting/renderers/save.py:117-121`, `src/mlframe/reporting/_diagnostics_dispatch_extra.py:261-263`, `src/mlframe/training/_eval_helpers.py:437`
- Statement: output paths are built from caller-supplied `base_path`/`plot_file` without containment checks.
- Scenario: none from untrusted data; the components that come from data (model/target names) are sanitised upstream (`_slugify` in `_phase_helpers.py:820`, `_sanitize` in `_slice_pareto_plot.py:131`). The base dir itself is the caller's own config.
- Disposition: REJECTED (caller-chosen output location is the API contract, not a traversal).

## Sites checked and judged safe (no finding)
- `training/composite/_estimator_dispatch.py:68-77`: module restricted to `mlframe.*`, private names refused, must be a class with `fit`.
- `training/neural/_flat_torch_module/__init__.py:210-230`: pickle-supplied optimizer ref restricted to `torch.optim.*`.
- `calibration/post.py:76-95`, `data/datasets/__init__.py:77-80`, `training/core/_setup_helpers_pipeline_cache.py:153-155`, `training/_param_model_spec.py:46-48`: module names come from hard-coded tables.
- `data/datasets/_sapp.py:68`: `DataFrame.eval` on a constant expression.
- `exec(compile(...))` sites are all in `_benchmarks/` on `git show HEAD:` of the repo's own source.
- subprocess: all list-argv, no `shell=True`; production sites (`utils/native_import_probe.py`, `system/kernel_tuning_cache/_ensure.py`, `training/_polars_native_support.py`, `training/_gpu_state_probe.py`) use fixed executables.
- Deletes: `suite_artefact_cache.py:209` validates keys (`_SAFE_KEY` + no `..`); `composite/cache_store.py:310-313` routes through `_safe_key`; `transformer/_key_bank.py:255-299` removes only its own UUID tmp dir; `cb/_cb_gpu_monitor.py:558` removes its own `mkdtemp`; `integrations/mlflow.py:119-130` uses `mkstemp` in the target dir and removes only that file.
- `estimators/pipelines.py:159-175` uses a private scratch dir (comment and code agree).

Out of scope here (deserialization class): `training/io.py:676` `dill.load` under `safe=False` is an explicit, warned opt-in.
