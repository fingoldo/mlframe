# Subprocess and shell audit, wave 4 (mlframe, src/mlframe), 2026-11-06

Read-only. Baseline: `audits/2026-10-31/subprocess.md` (CMD-01..CMD-12). HEAD re-checked.

`os.system`, `os.popen`, `os.exec*`, `os.spawn*`, `shell=True`: zero occurrences in `src/mlframe`. Every invocation is list-form argv.

## Status of prior findings on HEAD

- CMD-02 (manifest SHA into `git rev-list`): fixed. `fs_hybrid/_staleness.py:86,91` validates with `_SHA_RE.fullmatch` before `:94`.
- CMD-03 (bare `nvidia-smi` fallback): fixed in `training/_gpu_state_probe.py:88-92`. Only absolute PATH entries are searched, and no bare-name fallback remains.
- CMD-01 (`f"import {name}"`): still present at `utils/native_import_probe.py:56`. All 4 callers still pass the literal `"hnswlib"` (`_row_attention_ann.py:47`, `_knn_helper.py:56`, `_key_bank.py:217`, `_benchmarks/kernel_tuning_cache/auto_tune.py:216`). FUTURE stands.

## Findings

| ID | Severity | File:line | Failure scenario | Disposition |
|----|----------|-----------|------------------|-------------|
| W4-SUB-01 | Low | `feature_selection/_benchmarks/profiling/env_sampled_fit.py:16`, `feature_selection/_benchmarks/profiling/utilization.py:25` | Both call `subprocess.run(["nvidia-smi", ...])` by bare name. This is the class SUB2-02 fixed in `_gpu_state_probe.py`, but these sites were not fixed. On Windows, `CreateProcess` searches the working directory before PATH. A `nvidia-smi.exe` planted in the cwd of a profiling run (for example, a results directory someone else can write to) runs instead of the real tool. | RESOLVED-needed: reuse `_gpu_state_probe._resolve_smi()`. It is cheap, the fix already exists, and these are the only remaining bare-name `nvidia-smi` sites. |
| W4-SUB-02 | Info | `reporting/_benchmarks/_e2e_slice_finder_ab_iter71.py:48` | `ref` comes from the CLI and is passed into `git show` with no `--end-of-options`. A `ref` that starts with `-` is read as an option. The operator controls this value, so nothing crosses a trust boundary. | DOC. This was already noted under CMD-10. Restated here because the `_staleness.py` fix did not extend to it. |

Counts: High 0, Medium 0, Low 1, Info 1.

## Checked and safe

- `system/kernel_tuning_cache/_ensure.py:197-199`: argv is `sys.executable -P -m <fixed module> refresh <registered kernel_name>`, with a timeout set.
- `training/_polars_native_support.py:90`: the source is a constant and the library name is passed as `argv[1]`.
- `utils/native_import_probe.py:55-64`: uses `-P` (no cwd on `sys.path`), a timeout and UTF-8 decoding. The only residue is CMD-01 above.
- `fs_hybrid/_manifest.py:74`, `fs_hybrid/_staleness.py:76`: git takes fixed arguments plus a validated SHA.
- The `git show HEAD:<const>` benchmark helpers listed under CMD-10 have not changed and are still constant argv.
- `bench_iter87_cumulative.py:119`, `bench_iter100_*:113`, `bench_iter101_*:119`, `bench_sf3_jmim_exponent_selection.py:80`, `profiling/family_batch.py:45`, `profiling/family_scale.py:176`: list argv built from `sys.executable` and operator CLI arguments.
- `training/pipeline/_pipeline_extensions_pysr.py:12`: an import only, with no call site.
