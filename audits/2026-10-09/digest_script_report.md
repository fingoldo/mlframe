# ci_failure_digest.py report

## What it does
For every workflow (`gh workflow list --all`), takes the two latest completed runs on a branch. For a run with conclusion failure it lists the failed jobs, downloads each failed job log and extracts:
- pytest `FAILED`/`ERROR` node ids, with the first three `E` lines of the matching failure section;
- mypy `error:` lines (file:line, message, error-code suffix), grouped per file;
- infrastructure classes: `ModuleNotFoundError: <pkg>`, pytest-timeout, INTERNALERROR, cancelled (also from job conclusion cancelled), job timed out.

Groups are keyed by (workflow, kind, test id or mypy file or infra class) and labelled NEW, PERSISTING or FIXED against the previous run of the same workflow. A green latest run after a red one gives all FIXED. Workflows whose latest run is older than `--days` are skipped. Output is Markdown (per-workflow counts table, then NEW, PERSISTING, FIXED sections) and optional JSON.

Every gh call goes through `retry_gh` (8 attempts, 4 s sleep, `GhError` afterwards); all JSON is parsed in Python (no `--jq`); subprocess with utf-8, 180 s timeout, no shell. Pure functions (`parse_log`, `classify`, `render_markdown`, ...) are separate from I/O, which takes an injectable `run_gh`.

## Usage
`python scripts/ci_failure_digest.py --branch master [--repo fingoldo/mlframe] [--days 2] [--out digest.md] [--json digest.json]`

## Results
- `tests/scripts/test_ci_failure_digest.py`: 15 passed (log `%TEMP%\digest_test.log`). It caught one real ordering bug in the E-line merge, fixed. No `__init__.py` in `tests/scripts` because sibling test dirs have none; the script is loaded by file path as `tests/test_composite_config_reference_drift.py` does.
- ruff check: clean (one `# noqa: PERF203` on the intentional try/except in the retry loop). black_filtered_apply --check: clean. mypy with the repo config on the script: no issues (scripts/ is not excluded from mypy here).
- Live run (`--branch master --days 2`, rc 0, `%TEMP%\digest_live.md`): one workflow had a recent failure, sklearn-matrix, with 0 NEW, 0 PERSISTING, 5 FIXED. Excerpt:
  - `[sklearn-matrix] infra ModuleNotFoundError: py_ci_shared (jobs: sklearn 1.6.1 on py3.11 / ubuntu-24.04, ... +1 more)`
  - `[sklearn-matrix] test tests/training/fuzz/test_fuzz_suite.py` with detail `ModuleNotFoundError: No module named 'py_ci_shared'`

  The real logs parsed correctly: collection ERROR lines became file-level test ids, matrix jobs merged into one group, and the infra class was detected.

## Limitations
- The FIXED detection compares only the last two completed runs, so a flaky test that passes once shows as FIXED.
- A job that fails with no parsable FAILED/ERROR/mypy line and no infra marker produces no group (nothing is reported for it).
- Logs of very large runs are downloaded serially; with 8 retries a dead network can take minutes.
- Test-id details come from the pytest short summary text, which pytest truncates to the terminal width.

## Paths
New: `C:\Users\Admin\Machine learning\mlframe\scripts\ci_failure_digest.py`, `C:\Users\Admin\Machine learning\mlframe\tests\scripts\test_ci_failure_digest.py`, `C:\Users\Admin\Machine learning\mlframe\audits\2026-10-09\digest_script_report.md`. No existing file was edited.
