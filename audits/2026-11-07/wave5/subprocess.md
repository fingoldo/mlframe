# Subprocess and shell audit, wave 5 (mlframe, src/mlframe), 2026-11-07

Read-only. Baselines: `audits/2026-10-31/subprocess.md`, `audits/2026-11-06/wave4/subprocess.md`, `audits/2026-11-06/wave4/_TRACKER.md`. HEAD re-checked.

`os.system`, `os.popen`, `os.exec*`, `os.spawn*`, `shell=True`: still zero occurrences in `src/mlframe`. Every call uses list-form argv.

## Status of prior findings on HEAD

- W4-SUB-01 (bare `nvidia-smi` in profiling): fixed. `profiling/env_sampled_fit.py:16-17` and `profiling/utilization.py:24,27` now call `_gpu_state_probe._resolve_smi()` (`training/_gpu_state_probe.py:83-92`), which searches only absolute PATH entries and raises if nothing is found. Both callers catch that error.
- W4-SUB-02 / CMD-10 (`ref` from the CLI goes into `git show` with no `--end-of-options`): unchanged at `reporting/_benchmarks/_e2e_slice_finder_ab_iter71.py:48,59`. Only the operator sets this value. The DOC disposition stands.
- CMD-01 (`f"import {name}"`): unchanged at `utils/native_import_probe.py:56`. Callers still pass only literals. FUTURE stands.
- CMD-02 and CMD-03: still fixed (`fs_hybrid/_staleness.py` SHA validation, `_gpu_state_probe.py:83-92`).

## Findings

| ID | Severity | File:line | Failure scenario | Disposition |
|----|----------|-----------|------------------|-------------|
| (none) | - | - | No new attacker-reachable command construction was found. | - |

Counts: High 0, Medium 0, Low 0, Info 0.

## Checked and safe

- `training/_gpu_state_probe.py:98,109`: the executable is resolved to an absolute path and the arguments are constant.
- `system/kernel_tuning_cache/_ensure.py:199`: argv is `sys.executable`, a fixed module and a registered kernel name.
- `training/_polars_native_support.py:90`: the source is constant and the name is passed as argv.
- `utils/native_import_probe.py:55`: uses `-P` and a timeout. CMD-01 is the only remaining issue there.
- `fs_hybrid/_manifest.py:74`, `fs_hybrid/_staleness.py:76`: fixed git queries plus a validated SHA.
- `profiling/family_batch.py:45`, `profiling/family_scale.py:176`: list argv built from `sys.executable` and operator CLI arguments.
- The benchmark `git show HEAD:<const>` helpers (`bench_cpx39_decorrelator.py:31`, `ab_grouped_10m.py:31,35`, `bench_anchor_*`, `bench_knn_dispersion_vectorize.py:31`, `bench_spectral_centroid_e2e.py:23`, `filters/_benchmarks/bench_*`, `bench_mrmr.py:50`, `bench_pairwise_modular_cost.py:61`): the argv is constant. Bare `git` is resolved with cwd set to the repo, and anyone who can plant a file there can already write the code.
- `bench_iter87_cumulative.py:119`, `bench_iter100_*:113`, `bench_iter101_*:119`, `bench_sf3_jmim_exponent_selection.py:80`: `sys.executable` list argv.
- `training/pipeline/_pipeline_extensions_pysr.py:12`: import only, with no call site.
