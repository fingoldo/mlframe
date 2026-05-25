# Numba-disabled coverage CI pathway

## Problem

Per memory `reference_numba_coverage_blind.md`, every `@njit` function body is invisible to `sys.settrace` / `coverage.py`. mlframe ships ~64 source files with `@njit` / `@cuda.jit` decorators across critical hot paths: `feature_selection/filters/info_theory.py`, `cat_interactions.py`, `permutation.py`, `hermite_fe.py`, `feature_engineering/numerical.py`, `_numerical_numba.py`, `metrics/*` (10+ files), `core/{arrays,ewma,helpers}.py`, `preprocessing/outliers.py`, `feature_engineering/transformer/{_kernels_njit,_aggregation,_projection}.py`, etc. Current coverage reports show these as "untested" because the dispatch wrappers are reached but the kernel bodies themselves are JIT-compiled and bypass `settrace`.

The fix is `NUMBA_DISABLE_JIT=1` — disables JIT compilation entirely so kernel bodies execute as plain Python and become visible to coverage instrumentation. Cost: tests run 10-100x slower in this mode. Therefore this is a NIGHTLY / RC-only pathway, not the daily CI matrix.

## Proposed solution

### Part 1 — pytest marker (lands in this wave)

Add a `numba_disabled` marker that opts a test INTO the slow-coverage profile (rather than out of the fast profile). Tests that exercise numba kernel bodies and have value as coverage targets carry this marker.

`pyproject.toml` marker registration:

```toml
"numba_disabled: include in the NUMBA_DISABLE_JIT=1 nightly coverage run",
```

In this wave the marker is registered but applied to **zero** tests by default — that's intentional, so the daily suite is unaffected. The next wave should curate ~15-30 tests across the N1-N12 module list and mark them.

### Part 2 — helper script (lands in this wave)

`scripts/run_numba_coverage.sh` (POSIX) or `scripts/run_numba_coverage.ps1` (Windows) — operator-facing entry point that:

1. Sets `NUMBA_DISABLE_JIT=1` and `NUMBA_DISABLE_INTEL_SVML=1` in the environment.
2. Sets `PYTHONUNBUFFERED=1` per `feedback_pytest_unbuffered`.
3. Runs pytest with explicit coverage of the kernel-heavy packages:
   - `--cov=src/mlframe/feature_selection/filters`
   - `--cov=src/mlframe/feature_engineering`
   - `--cov=src/mlframe/metrics`
   - `--cov=src/mlframe/core`
   - `--cov=src/mlframe/preprocessing/outliers`
4. Selects tests via `pytest -m numba_disabled` once tests are tagged; until then runs the full kernel-package test set.
5. Writes coverage to `_results/coverage_numba_disabled.xml`.

NOTE per memory `feedback_pytest_no_cov`: on Windows, `--no-cov` is the default daily addopt. This script intentionally OVERRIDES that. It MUST be run from a clean environment with no parallel pytest processes (per memory `feedback_parallel_agents_heartbeat`).

### Part 3 — CI integration (ARCH-DEFER, needs orchestrator approval)

GitHub Actions workflow `.github/workflows/numba_coverage_nightly.yml` running on cron + manual dispatch, calling the script and uploading coverage XML to Codecov as a separate flag (`numba-disabled`).

This part is out-of-scope for Wave 3 — it changes CI workflow configuration. Orchestrator + user OK required.

## Alternatives considered

1. **Status quo (do nothing)** — leaves kernel bodies permanently invisible to coverage. Recurring blind spot.
2. **Inline non-jitted Python sibling per kernel** — would double maintenance burden across 64 files. Rejected as too invasive.
3. **Daily full-suite NUMBA_DISABLE_JIT** — would make daily CI 10x slower. Rejected.

## Risks

1. **JIT-flag side effects**: `NUMBA_DISABLE_JIT=1` disables ALL @njit, including non-mlframe deps (sklearn, lightgbm bindings in some configurations). Some library-side cached objects may fail to construct in the no-JIT mode.
2. **Performance test pollution**: any perf assertion (`assert wall_seconds < X`) would fail in this mode. The marker selection must exclude perf tests.
3. **Fastmath path divergence**: `@njit(fastmath=True)` kernels produce slightly different float-bit patterns than pure-Python interpretation; numeric-equality tests may need wider tolerances under no-JIT.

## Recommendation

Proceed with Parts 1 + 2 in Wave 3. Defer Part 3 (CI workflow) to a dedicated wave with explicit user OK, since the GH Actions YAML change is irrevocable on push without a force-push or revert PR.

## Status (2026-05-25 update)

**Part 1 + Part 2**: DONE in commit ``b26cfa93`` (w3c B1_N1_to_N12). Marker registered in ``pyproject.toml``; helper scripts ``scripts/run_numba_coverage.sh`` + ``scripts/run_numba_coverage.ps1`` shipped.

**Part 3**: DONE in commit ``06727c04`` (W8C AP5). Workflow ``.github/workflows/numba-coverage.yml`` active. Meta-sensor ``tests/test_meta/test_numba_coverage_workflow_exists.py`` pins the workflow file.

### Validation steps (operator-facing)

To manually trigger and observe a nightly run:

```bash
gh workflow run numba-coverage-nightly --ref master
gh run list --workflow=numba-coverage-nightly --limit=5
gh run view <RUN_ID> --log              # full log
gh run view <RUN_ID> --log-failed       # just the failed step
```

To preview locally (Windows):

```powershell
$env:NUMBA_DISABLE_JIT='1'; $env:PYTHONUNBUFFERED='1'
& "D:/ProgramData/anaconda3/python.exe" -m pytest tests/feature_engineering/ tests/feature_selection/ tests/metrics/ tests/training/ --no-cov -p no:randomly --timeout=600
```

To preview locally (POSIX):

```bash
bash scripts/run_numba_coverage.sh
```

### First-run observations (2026-05-25 nightly run id 26388114573)

The first scheduled run on master at 03:00 UTC 2026-05-25 surfaced two test failures, consistent with the "Risks" section above:

1. **``tests/feature_engineering/test_audit_regression_multi_agent_findings.py::TestMpsCorrectness::test_monotone_rise_then_fall``** -- ``IndexError: index 7 is out of bounds for axis 0 with size 7`` at ``src/mlframe/feature_engineering/mps.py:129``. This is a real prod bug surfaced by no-JIT execution of an off-by-one or boundary check that the JIT-compiled path was silently tolerating (numba's bounds-checking is relaxed by default; pure Python is strict). **Action**: flag for the Wave-11 backlog; investigate ``mps.py:129`` and ship a regression sensor that runs under both ``NUMBA_DISABLE_JIT=0`` and ``NUMBA_DISABLE_JIT=1``.

2. **``tests/feature_selection/test_biz_val_filters_hermite_fe.py::test_biz_njit_poly_eval_3x_faster_than_numpy_at_n2k``** -- ``AssertionError: njit hermeval must be >=3x faster than numpy at n=2k; got 0.01x (20.7us vs 3450.1us)``. This is the predicted Risk #2 (Performance test pollution): under ``NUMBA_DISABLE_JIT=1`` the @njit kernel runs as plain Python and is therefore SLOWER than the C-vectorized numpy baseline, not 3x faster. The perf-floor assertion is correct under normal CI and wrong under no-JIT. **Action**: gate the biz_value perf-floor tests via ``pytest.mark.skipif(os.environ.get('NUMBA_DISABLE_JIT') == '1')`` so they are skipped in the nightly run. The non-perf correctness assertions in the same file should still run.

Both items are tracked for the Wave-11 backlog under ``audit/critique_2026_05_24/FINAL_VERIFICATION.md`` (post-2026-05-25 amendment) and do not block the AP5 closure status; the workflow IS landed and IS producing actionable signal, which is the proposal's primary purpose.
