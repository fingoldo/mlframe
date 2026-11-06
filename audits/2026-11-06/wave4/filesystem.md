# Wave 4: filesystem and temp files (2026-11-06)

Read-only. Prior dispositions FS-01..FS-16 (audits/2026-10-31/filesystem.md) re-checked by grep on HEAD; site lines still present (pipelines.py:161/175 mkdtemp+rmtree, _cb_border_cache.py:129/192 mkstemp, rfecv/__init__.py:583 mkstemp in target dir, mlflow.py:119 mkstemp in target dir). No disposition found wrong.

## Findings

| ID | Severity | file:line | Failure scenario | Disposition |
|---|---|---|---|---|
| (none new) | - | - | No new site with a concrete failure scenario beyond FS-01..FS-16. | - |

## Checked and safe
- `src/mlframe/feature_engineering/transformer/_key_bank.py:254-299`: tmp dir under caller cache_dir with uuid suffix; rmtree only on that self-created tmp path; final dir never removed.
- `src/mlframe/feature_selection/filters/_fe_family_budget.py:41,233,264-272`: per-user `~/.cache` dir; filename via `_sanitize_budget_file_key` allowlist; pid+uuid tmp + `os.replace`; unlink only own tmp.
- `src/mlframe/reporting/renderers/_kaleido.py:99-104`: removes only its own scratch render path.
- `src/mlframe/system/kernel_tuning_cache/_ensure.py:138`, `src/mlframe/reporting/_shared_arrays.py:91,127`: unlink own lock / own shared-memory segments.
- `src/mlframe/feature_selection/filters/_joblib_safe.py:304,353`: mkstemp (0600, O_EXCL), unlink own file.
- `_benchmarks/*` mkdtemp/mkstemp/TemporaryDirectory sites: unpredictable names; `gettempdir()` roots in `fe_operator_factory/common/_paths.py:14` and `profiling/_paths.py:11` are developer-only, tagged tmp-path-ok (covered by FS-06 class).
- No `chmod`/`umask`/world-writable modes in src (FS-16 still holds).
