# Dynamic code execution audit, round 2 (mlframe, src/mlframe)

Read-only. No source, test or config file was changed. This round re-checks the 2026-10-31 report
(`audits/2026-10-31/dynamic_exec.md`, EXE-01..EXE-29) against a fresh grep of `eval`, `exec`, `compile`,
`__import__`, `import_module`, `runpy`, `getattr` on dynamic names, `shell=True` and `os.system`. Findings
already in the prior report are cross-referenced, not re-reported as new.

## Summary

- New High: 0. New Medium: 0. New Low: 1. New Info: 6. Cross-references to prior IDs: 4.
- No `eval()` on a runtime string anywhere in `src/mlframe`. No `shell=True` or `os.system` in production code.
- Every `exec(compile(...))` is in a `_benchmarks` module and reads committed source from `git show HEAD:`.
- The one new real gap is the unrestricted `getattr` walk in `__setstate__` of the flat torch module (DYN2-1).
  It sits behind pickle, so it is not a new trust boundary.

## Findings

| ID | File:line | Severity | Proposed disposition |
|---|---|---|---|
| DYN2-1 | src/mlframe/training/neural/_flat_torch_module/__init__.py:211-220 | Low | FUTURE |
| DYN2-2 | src/mlframe/training/composite/_estimator_dispatch.py:70-72 | Low | FUTURE (duplicate of EXE-10) |
| DYN2-3 | src/mlframe/calibration/post.py:79, 143, 154 | Info | RESOLVED |
| DYN2-4 | src/mlframe/feature_selection/_benchmarks/kernel_arg_dtype_audit.py:190-193 | Info | RESOLVED |
| DYN2-5 | src/mlframe/feature_selection/_benchmarks/profiling/nvprof_autotag.py:40-44, 61-62 | Info | RESOLVED |
| DYN2-6 | src/mlframe/estimators/pipelines.py:57; models/_optimization_shared.py:35; feature_engineering/mps.py:58; metrics/calibration/_calibration_plot.py:46 | Info | RESOLVED |
| DYN2-7 | src/mlframe/data/synthetic.py:90-104 | Info | RESOLVED |
| DYN2-8 | src/mlframe/feature_selection/_benchmarks/bench_mrmr.py:85; feature_selection/_benchmarks/fs_hybrid/_manifest.py:110 | Info | RESOLVED |

## Details

**DYN2-1, training/neural/_flat_torch_module/__init__.py:211-220 (Low, FUTURE).**
`__setstate__` reads `("__optimizer_class_ref__", module, qualname)` from the unpickled state. It calls
`importlib.import_module(module)` and then walks `qualname.split(".")` with `getattr`, with no module
allow-list.
- Who controls the input: whoever wrote the model file. The file is read through the pickle path.
- What executes: any attribute reachable from any importable module (for example `os` via `module="os"`).
  The walk does not check that the result is an optimizer class.
- Why Low: a crafted pickle already runs arbitrary code during `pickle.load`, so this adds no new capability.
  The gap is that the state-level reference is unrestricted, which matters only if the loader is ever changed to
  restricted unpickling. That combination is what the denylist in `training/io.py:204` is meant to block.
- Note: the `except Exception` at line 221 swallows failures, so a bad reference silently leaves a string.
- Fix (deferred): require `module` to be `torch.optim` or `torch.optim.*`, and require the resolved object to be a
  `type` that subclasses `torch.optim.Optimizer`, before the walk. This extends EXE-18, which covered only the import.

**DYN2-2, training/composite/_estimator_dispatch.py:70-72 (Low, FUTURE, duplicate of EXE-10).**
Same finding as EXE-10. The only caller in this round is line 198, which passes a recommendation built from
literals. The `mlframe.` prefix guard is present. The allow-list fix stays FUTURE.

**DYN2-3, calibration/post.py:79, 143, 154 (Info, RESOLVED).**
`_try_import_class` runs `__import__` on its arguments. The only callers are the `_NEEDS_2D_CALIBRATORS` tuple
(line 88, a module-level literal) and the literal `("venn_abers", "VennAbersCalibrator")` at line 154. No config,
model or dataset value reaches it. This closes the "did not inspect" note in EXE-01.

**DYN2-4, feature_selection/_benchmarks/kernel_arg_dtype_audit.py:190-193 (Info, RESOLVED).**
`runpy.run_path` runs a fixed sibling file built from `__file__` (`bench_four_pillars.py`). The `--only` argument goes
into `sys.argv` as data for that script, not into the path or the code. The CLI operator already has code execution.

**DYN2-5, feature_selection/_benchmarks/profiling/nvprof_autotag.py:40-44, 61-62 (Info, RESOLVED).**
Module and function names come from a pstats dump file. They are written into the generated `nvprof_auto.py` with
`{mod!r}` and `{name!r}`, so each becomes a Python string literal and cannot break out of the quotes. The generated file
is then run by the developer. The `repr` quoting is the guard.

**DYN2-6, `_LazyModule` copies (Info, RESOLVED).**
EXE-04 covered `estimators/pipelines.py` only. The identical class in `models/_optimization_shared.py`,
`feature_engineering/mps.py` and `metrics/calibration/_calibration_plot.py` was not listed. Every call site
(pipelines.py:61-63, mps.py:62-63, _calibration_plot.py:50-51, _optimization_shared.py:39) passes a literal
module path. Keep the callers literal.

**DYN2-7, data/synthetic.py:90-104 (Info, RESOLVED).**
`getattr(stats, dist_name)` looks up a name taken from `stats._distr_params.distcont` and `distdiscrete` (lines 78-79
of the function), which are scipy's own lists. The caller's `include` and `exclude` only filter that list. A caller
cannot name an arbitrary attribute.

**DYN2-8, `__import__` in benchmark helpers (Info, RESOLVED).**
`bench_mrmr.py:85` imports `mod` from a literal tuple inside `_versions`. `fs_hybrid/_manifest.py:110` imports from
the literal `("numpy", "numba", ...)` tuple. Neither reads external input.

## Cross-references to the 2026-10-31 report (not re-reported)

- EXE-10 and DYN2-2: estimator dispatch, FUTURE.
- EXE-18 and DYN2-1: flat torch `__setstate__`, FUTURE. DYN2-1 adds the unrestricted attribute walk.
- EXE-21..26: `exec(compile(...))` of `git show HEAD:` in benchmarks, DOC. Re-verified: all six remain benchmark-only.
- EXE-28: `profile_mrmr_line_profile.py` import from argv, REJECTED. Re-verified.

## Scope limits

- Deserialization sinks (pickle, joblib, dill, cloudpickle) were excluded here; they belong to the deserialization class.
  `training/io.py:204` denylist was read only to confirm it is a restriction layer, not a dynamic-exec site.
- Coverage is grep-based. Non-literal `getattr` calls were checked only where they reached an import or a dispatch
  path. Other `getattr(obj, name)` calls on a caller-built object are out of class (EXE-02 pattern).
