# Security audit: temporary files, file paths and file permissions

Scope: `src/mlframe` (read-only). Class: temporary files, path construction and traversal, symlink-following writes, file permissions, shared cache directories.

Method: grep for `tempfile|mktemp|NamedTemporaryFile|mkstemp|TemporaryDirectory|chmod|umask|0o666|0o777|/tmp`, grep for `open(...path|file|name|dir)` writes, and a read of each non-benchmark hit.

## Summary

- 16 findings: 0 High, 1 Medium, 5 Low, 10 Info.
- Runtime code uses `mkstemp`, `mkdtemp`, `O_EXCL` and `uuid` temp names correctly in the main write paths. No `chmod`, `umask`, `0o666` or `0o777` appears anywhere under `src/mlframe`, so no world-writable modes are created.
- The real exposure is fixed, predictable paths under the shared system temp directory (`tempfile.gettempdir()`). This matters on POSIX multi-user hosts. On Windows the temp dir is per user under `%LOCALAPPDATA%`, which lowers the risk.
- No write path takes HTTP or other network input. The path-traversal risks are in library helpers that trust a caller-supplied path, and in one documented helper.

## Findings table

| ID | file:line | Severity | Proposed status |
|---|---|---|---|
| FS-01 | `src/mlframe/integrations/mlflow.py:95` | Low | DOC |
| FS-02 | `src/mlframe/estimators/pipelines.py:163-166` | Medium | FUTURE |
| FS-03 | `src/mlframe/training/core/_setup_helpers_pipeline_cache.py:136-138` | Low | FUTURE |
| FS-04 | `src/mlframe/training/crash_diagnostics.py:130` | Low | FUTURE |
| FS-05 | `src/mlframe/training/feature_handling/hf_provider.py:113-124` | Low | FUTURE |
| FS-06 | `src/mlframe/feature_selection/filters/_benchmarks/bench_temporal_agg_fe.py:39`, `src/mlframe/reporting/_benchmarks/_e2e_slice_finder_ab_iter71.py:49-50` | Low | FUTURE |
| FS-07 | `src/mlframe/system/kernel_tuning_cache/_ensure.py:96-101, 131-134` | Info | RESOLVED |
| FS-08 | `src/mlframe/training/io.py:58-68, 155-167` | Info | RESOLVED |
| FS-09 | `src/mlframe/utils/disk_cache.py:420-430` | Info | RESOLVED |
| FS-10 | `src/mlframe/training/composite/cache_store.py:175-178, 563` | Info | RESOLVED |
| FS-11 | `src/mlframe/feature_selection/wrappers/rfecv/__init__.py:545-560` | Info | RESOLVED |
| FS-12 | `src/mlframe/feature_selection/wrappers/rfecv/_cb_border_cache.py:114-127, 183-190` | Info | RESOLVED |
| FS-13 | `src/mlframe/feature_selection/filters/_joblib_safe.py:350-365` | Info | RESOLVED |
| FS-14 | `src/mlframe/training/cb/_cb_gpu_monitor.py:490` | Info | RESOLVED |
| FS-15 | `src/mlframe/reporting/report_html.py:258`, `src/mlframe/reporting/renderers/plotly.py:418-424`, `src/mlframe/reporting/renderers/_kaleido.py:235-243`, `src/mlframe/training/ranking/_ranker_suite_train_helpers.py:484` | Info | REJECTED |
| FS-16 | whole `src/mlframe` tree (no `chmod`, `umask`, `0o666`, `0o777`) | Info | RESOLVED |

## Details

### FS-01 mlflow.py:95 (Low, DOC)
Pattern: `open(fname + extension, "w")`. `fname` is used without confinement. The docstring at lines 78-80 already states that `fname` must be internal and that `../` or absolute paths write outside the intended directory. Untrusted input does not reach it in-tree. Traversal: yes, if a caller passes user input. Disposition DOC: keep the docstring warning. Optional hardening, if callers ever change: resolve the path and check it is under a base directory.

### FS-02 estimators/pipelines.py:163-166 (Medium, FUTURE)
Pattern: `out_dir = output_dir or tempfile.gettempdir()`, then `joblib.dump(cv_results, os.path.join(out_dir, f"cv_results-{slugify(title)}.dump"))` followed by `write_sidecar`. The name is fixed and predictable, so on a multi-user POSIX host another local user can pre-create it as a symlink or plant a file. `joblib.dump` opens with `"wb"`, which follows symlinks, so the write lands wherever the link points. The later load path is a pickle load; the sidecar checksum does not help because an attacker who can write the dump can also write the sidecar. Untrusted input: `title` is a code-level string, so the attacker here is a local user, not a remote one. Severity Medium because of the pickle load of a file in a shared directory. Deferred: default `output_dir` should be a per-user directory (for example `~/.cache/mlframe`), created `0o700`, and the file should be written via `mkstemp` plus `os.replace`. Reason for deferral: changing the default location changes where users find their dumps.

### FS-03 _setup_helpers_pipeline_cache.py:136-138 (Low, FUTURE)
Pattern: fixed directory `tempfile.gettempdir()/mlframe_caches` with fixed file `polars_ds_pipeline_roundtrip.json`, created with `os.makedirs` and no mode. The file is read via `orjson` (JSON, not pickle), so code execution is not possible. Impact: a co-tenant can pre-create or poison the entries, and the pipeline round-trip cache then reports entries as valid and skips checks. Writes follow symlinks. Fix: per-user cache dir with `0o700`, and `O_NOFOLLOW` or `mkstemp` plus `os.replace` on save. Deferred because the impact is correctness of a cache, not code execution.

### FS-04 crash_diagnostics.py:130 (Low, FUTURE)
Pattern: fallback crash-output directory is `tempfile.gettempdir()` when no logging file handler is found. Crash artefacts are written there with predictable names, so the same symlink and pre-creation risk as FS-02 applies on shared POSIX hosts. Fix: fall back to a per-user directory created `0o700`. Deferred: fallback only, and crash artefacts are not read back.

### FS-05 hf_provider.py:113-124 (Low, FUTURE)
Pattern: lock directory is `HF_HOME/.mlframe-fhc-locks` or, failing that, `gettempdir()/mlframe-hf-locks`. The lock file is `hf_<blake2b>.lock`. The name is deterministic, so a co-tenant can pre-create the file or directory and block the lock (denial of service). On `OSError` the code falls back to the bare temp directory. Fix: create the lock directory `0o700` under a per-user root, and fail closed rather than falling back to the shared temp root. Deferred: the impact is availability only.

### FS-06 benchmark scripts with fixed names in the temp dir (Low, FUTURE)
`bench_temporal_agg_fe.py:39` writes `gettempdir()/_temporal_agg_fe_baseline.py`, and `_e2e_slice_finder_ab_iter71.py:49-50` writes `gettempdir()/_old_slice_finder_iter71.py` and then writes its content. Both are development scripts, not runtime code, but the write path is fixed in a shared directory, and the first one is a `.py` file that a later import could execute if its name is reused. Fix: use `tempfile.mkstemp` or a fresh `mkdtemp` (`0o700`). Deferred: benchmark-only and not reachable from users.

### FS-07 kernel_tuning_cache/_ensure.py:96-134 (Info, RESOLVED)
The lock is created with `open(path, "x")`, which is `O_EXCL`, under `~/.pyutilz/kernel_tuning` (per user, not shared). Stale locks are reclaimed by PID check. The unlink-then-retry window is small and only causes a wasted tuning run, not a corrupt file. Already safe.

### FS-08 training/io.py:58-68, 155-167 (Info, RESOLVED)
Atomic writer uses a monotonic counter plus PID plus uuid name with `os.open(O_CREAT | O_EXCL | O_WRONLY)`, then `os.replace`. The comment at lines 155-160 explains the deliberate choice over `mkstemp`, and `O_EXCL` gives the same exclusivity. Already safe.

### FS-09 utils/disk_cache.py:420-430 (Info, RESOLVED)
Temp name is `tmp_<uuid4>.pkl` in the cache dir, then `os.replace` under a per-key lock. The uuid is not guessable. Already safe for the write. The cache dir itself is caller-configured; sharing it is the operator's choice.

### FS-10 training/composite/cache_store.py:175-178, 563 (Info, RESOLVED)
`tempfile.mkstemp(dir=self.cache_dir, ...)`, `fsync`, then `os.replace`. The fd-ownership flag prevents a leaked descriptor. Already safe.

### FS-11 rfecv/__init__.py:545-560 (Info, RESOLVED)
`mkstemp(prefix=".rfecv_ckpt_", dir=dir_name)` creates a `0o600` file, then `os.fdopen`, then replace. The fd-adoption flag closes the descriptor on failure. Already safe.

### FS-12 rfecv/_cb_border_cache.py:114-127, 183-190 (Info, RESOLVED)
`mkstemp(suffix=".borders.tsv")` reserves a `0o600` file, which CatBoost then overwrites by path. Reads go back through the same path. The file is a short-lived scratch file with no secret content. Already safe.

### FS-13 feature_selection/filters/_joblib_safe.py:350-365 (Info, RESOLVED)
`mkstemp(prefix="mlframe_fitconst_", suffix=".mmap")` plus memmap, with `_release_fit_constant_entry` on failure. Already safe.

### FS-14 training/cb/_cb_gpu_monitor.py:490 (Info, RESOLVED)
`tempfile.mkdtemp` creates a `0o700` directory. Already safe.

### FS-15 caller-path writers: report_html.py:258, renderers/plotly.py:418-424, renderers/_kaleido.py:235-243, ranking/_ranker_suite_train_helpers.py:484 (Info, REJECTED)
These write to a path the caller supplies, with no confinement. Untrusted input does not reach them in-tree, because every caller is pipeline code that builds the path from configuration or model names. The ranker path uses `_safe_model_name` (sanitised). Not a risk in the current code; the library contract is "caller owns the path". Rejected.

### FS-16 permissions (Info, RESOLVED)
Searched the whole tree for `chmod`, `os.chmod`, `umask`, `0o666`, `0o777` and `0o77`. None occur in `src/mlframe`. Files therefore get the process umask default, and no directory is made world-writable by mlframe code. Already safe.

## Out of scope but noted
- Pickle and joblib loads of user-supplied bundle paths (for example `training/io.py:381`, `training/io.py:588`) are a deserialization class, not this one. Recommend a separate audit.
