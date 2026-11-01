# Subprocess and shell audit, 2026-11-01 (mlframe, src/mlframe)

Read-only audit. No source, test, or config file was edited and no git state was changed.
Prior round: `audits/2026-10-31/subprocess.md` (CMD-01 to CMD-12). Those IDs are not repeated here, except where this round
reopens one (see SUB2-01).

## Summary

- `os.system`, `os.popen`, `os.exec*`, `os.spawn*`, `pty.spawn`, and `shell=True`: zero hits in non-benchmark code in `src/mlframe`.
  The only `shell=True`-style text in the tree is in comments and docstrings.
- Every real invocation is list-form argv with no shell. No command string is built from data.
- Re-checked the production (non-`_benchmarks`) subprocess sites: `utils/native_import_probe.py`, `training/_polars_native_support.py`,
  `training/_gpu_state_probe.py`, `system/kernel_tuning_cache/_ensure.py`, `feature_selection/_benchmarks/fs_hybrid/_staleness.py`
  and `_manifest.py`, plus the `__init__.py` CUDA and OMP environment autoconfig. All of them match or extend the prior audit.
- Two new findings in this round, both about the working directory. Child interpreters started with `-c` or `-m` put the current
  directory on `sys.path`, and on Windows `shutil.which` searches the current directory first. Both are reachable without any
  user-supplied argument. Severity counts: **High 0, Medium 0, Low 2, Info 1** (3 findings).

## Findings table

| ID | File:line | Call | Severity | Proposed disposition |
|----|-----------|------|----------|----------------------|
| SUB2-01 | `src/mlframe/utils/native_import_probe.py:55-58`; `src/mlframe/training/_polars_native_support.py:90-93`; `src/mlframe/system/kernel_tuning_cache/_ensure.py:164`; benchmark workers (see details) | `python -c` / `python -m` children, cwd on `sys.path` | Low | FUTURE |
| SUB2-02 | `src/mlframe/training/_gpu_state_probe.py:86-88` (and `:99`) | `shutil.which("nvidia-smi")` resolves to a current-directory `nvidia-smi.exe` on Windows | Low | FUTURE |
| SUB2-03 | `src/mlframe/__init__.py:83-135` | CUDA_HOME / CUDA_PATH written into `os.environ`, inherited by every child | Info | REJECTED |

## Details

### SUB2-01: child interpreters import from the current directory (Low, FUTURE)

`python -c <code>` prepends `''` (the current directory) to `sys.path`. `python -m <module>` prepends the current directory as
well. A child started from a directory that holds a file or package with the same name as the module the child imports will
execute that file instead of the installed module.

Sites, all without `-P`, `-I`, or a neutral `cwd=`:

- `src/mlframe/utils/native_import_probe.py:55-58`: `[sys.executable, "-c", f"import {name}"]`. `name` is `hnswlib` at every caller
  today. Planted `hnswlib.py` in the current directory runs in the child.
- `src/mlframe/training/_polars_native_support.py:90-93`: `[sys.executable, "-c", _PROBE_SOURCE, library]`. `_PROBE_SOURCE` starts with
  `import polars as pl`. Planted `polars.py` runs in the child.
- `src/mlframe/system/kernel_tuning_cache/_ensure.py:164`: `[sys.executable, "-m", "mlframe.system.kernel_tuning_cache", "refresh", kernel_name]`.
  A `mlframe/` directory in the current directory shadows the package. The child then runs `refresh` from the shadow copy.
  This site was marked RESOLVED as CMD-05 for the subprocess mechanics (list argv, timeout, lock). It is reopened here only for
  the import-path question.
- Benchmark workers with the same pattern: `feature_selection/_benchmarks/bench_iter87_cumulative.py:119`,
  `bench_iter100_stratified_anchors_contour.py:113`, `bench_iter101_stratified_anchors_2d.py:119`,
  `feature_selection/filters/_benchmarks/bench_sf3_jmim_exponent_selection.py:80`. Lower priority, operator-run only.
- Not affected: `feature_selection/_benchmarks/profiling/family_batch.py:45` uses `-I`.

Failure scenario: a user runs a training script from a downloaded or shared project folder that contains a planted `polars.py`.
The parent process imports `polars` from site-packages, because a script's `sys.path[0]` is the script's directory. The
probe child then imports the planted file, and its code runs with the user's privileges. The escalation is from the script
directory to the current directory. The parent's own behavior is unchanged.

Impact is Low: it needs write access to a directory the user runs from, and the child inherits the same user account.

Proposed fix: pass `cwd=` a neutral directory (for example `tempfile.gettempdir()` or the interpreter's directory) to the
child, which is portable to Python 3.9 (`requires-python = ">=3.9"`, `pyproject.toml:13`). `-P` (Python 3.11+) is the
cleaner flag but is not available on the 3.9 and 3.10 floor. Do this when the probes are next touched.

Disposition: FUTURE. Real, low exposure, cheap fix.

### SUB2-02: `shutil.which` resolves the GPU tool from the current directory on Windows (Low, FUTURE)

`src/mlframe/training/_gpu_state_probe.py:86-88`:

```python
exe = shutil.which("nvidia-smi")
if exe is None:
    exe = "nvidia-smi"
```

This is broader than CMD-03 in the prior round, which framed the risk as the bare-name fallback only. On Windows the default
behavior of `shutil.which` is to search the current directory before `PATH`. The stdlib code this environment runs
(Python 3.14.3, `shutil._win_path_needs_curdir`) calls `NeedCurrentDirectoryForExePath`, which returns true unless the
`NoDefaultCurrentDirectoryInExePath` environment variable is set. So `shutil.which` returns a relative `nvidia-smi.exe` when one
sits in the current directory, and that file is then run by `subprocess.run` with the two query arguments (`:88` and `:99`).
This happens on every GPU-state probe, not only in the fallback branch. The same resolution applies on the Python 3.9 floor,
because the Windows lookup behavior is in the OS.

Failure scenario: a user runs mlframe from a folder that holds a planted `nvidia-smi.exe`. The next GPU probe (heartbeat or
`_cb_pool` probe) runs the planted binary with the user's privileges.

Impact is Low. It needs write access to the working directory, and the probe is not reachable from data.

Proposed fix: on Windows resolve the absolute System32 path (`os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")`)
when it exists, and otherwise return `None` without falling back to the bare name. An alternative that keeps `shutil.which`
is to reject any result that is not absolute.

Disposition: FUTURE.

### SUB2-03: CUDA_HOME / CUDA_PATH environment written at import (Info, REJECTED)

`src/mlframe/__init__.py:83-135` sets `os.environ["CUDA_HOME"]` and `CUDA_PATH` (from a complete `CUDA_PATH_V*` toolkit or the
pip NVVM directory) at `import mlframe`. The values are read from the operator's environment and the local filesystem, not
from data. Every child process inherits them, which is intended: the child numba code needs the same toolkit. No secret is
injected. The opt-out is documented in the docstring (`MLFRAME_NO_CUDA_AUTOCONFIG`).

Disposition: REJECTED. Environment inheritance by children is the design, and the source of the values is operator-controlled.

## Non-findings checked

- `os.system`, `os.popen`, `os.exec*`, `os.spawn*`, `pty.spawn`, `webbrowser`, `shell=True` in non-benchmark code: none.
- `utils/native_import_probe.py` `MLFRAME_*` override (prior CMD-07): environment trust, DOC stands.
- `reporting/_async_render.py:356`: `ProcessPoolExecutor` with `mp_context=get_context("spawn")`. Callable is internal, no argv.
- `reporting/renderers/_kaleido.py`: Chromium is started by the kaleido library, not by mlframe argv. Out of scope for this class.
- `training/_unpickle_policy.py:27`: `"Popen"`, `"system"` in a denylist. Security control, not an execution site.
- `training/io.py:204-206`: denylist of module names for the unpickler. Same, not an execution site.
- `feature_selection/filters/_neural_mi.py:297-319, 419-433`: `sys.path.insert` of the vendored package, in process. Not a child path, and the vendored path is inside the package.
- `feature_selection/_benchmarks/fs_hybrid/_staleness.py:75-125` (CMD-02): unchanged, FUTURE stands.
- `metrics/_core_numba_warmup.py:154-160`: a background thread triggers loky's physical-core probe. No argv from data.
