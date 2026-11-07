# Wave 5: filesystem and temp files (2026-11-07)

Scope: src/mlframe (benchmarks excluded). Prior rulings: audits/2026-10-31/filesystem.md (FS-01..FS-16), audits/2026-11-06/wave4/filesystem.md (none new).

## 1. Findings

| ID | Severity | file:line | Failure scenario | Disposition |
|---|---|---|---|---|
| W5-FS-01 | Info | `utils/private_temp.py:36-45`; callers `training/core/_setup_helpers_pipeline_cache.py:137`, `training/crash_diagnostics.py:130`, `training/feature_handling/hf_provider.py:123`, `feature_selection/wrappers/rfecv/_cb_border_cache.py:124` | No failure: earlier FUTURE rows FS-03, FS-04, FS-05 (predictable shared /tmp names) are now routed through `private_temp_dir`, which names per user, creates 0o700, `lstat`s (so a planted symlink fails `S_ISDIR`) and falls back to `mkdtemp` when owner/mode is wrong. FS-02 (`estimators/pipelines.py:161`) uses `mkdtemp`. | RESOLVED-needed: update the 2026-10-31 TRACKER rows FS-02..FS-05 to RESOLVED (bookkeeping only) |

No new issue with a concrete failure scenario.

## 2. Checked and safe

- `feature_engineering/transformer/_key_bank.py:248-299`: uuid-suffixed tmp dir under caller cache_dir, rmtree only of that tmp dir, never of the published dir.
- `feature_selection/filters/_fe_family_budget.py:41,264-272`: per-user `~/.cache` dir, file key sanitised, pid+uuid tmp then `os.replace`.
- `feature_selection/filters/_joblib_safe.py:353`: `mkstemp`.
- `feature_selection/wrappers/rfecv/__init__.py:583`, `integrations/mlflow.py:123`, `training/composite/cache_store.py:178,555`: `mkstemp` in target dir + replace; removals only of own tmp or cache-owned entries.
- `training/_training_loop_refit.py:70-86`: snapshot deletion guarded by `commonpath(realpath)` containment and `islink` refusal.
- `reporting/renderers/_kaleido.py:104,272`: scratch is `<caller path>.partial-<uuid4>`, removed only by that name.
- `training/cb/_cb_gpu_monitor.py:490,558`: `mkdtemp` + rmtree of that dir.
- `utils/disk_cache.py`, `training/suite_artefact_cache.py`, `training/feature_handling/cache.py`, `cache_backend.py`, `locking.py`, `system/kernel_tuning_cache/_ensure.py`, `utils/numba_cache_guard.py`: unlink only of hashed keys / own tmp files inside the cache root.
- `reporting/_shared_arrays.py:91,127`: unlink of shared-memory segments this process created.
- No `os.chmod`, `umask`, 0o666/0o777 in src (FS-16 still holds).
