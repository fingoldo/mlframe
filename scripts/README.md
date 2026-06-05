# scripts/

Maintenance, benchmark, and pre-commit-hook helpers. Run them from the repo
root with the project's interpreter, e.g. `python scripts/<name>.py`.

## Benchmarks

- **`bench_slice_es_synthetics.py`** — exploratory benchmark of slice-stable early-stopping across four synthetic constructions (small/noisy val, drift, etc.), paired Wilcoxon vs single-val ES. Prints the leaderboard to stdout and writes JSON results under `benchmarks/`. Run: `python scripts/bench_slice_es_synthetics.py`.
- **`bench_slice_es_synthetics_v2.py`** — follow-up wave testing aggressive-overfit constructions and minimally-conservative aggregator settings. Stdout + `benchmarks/` JSON. Run: `python scripts/bench_slice_es_synthetics_v2.py`.
- **`bench_slice_es_synthetics_v3.py`** — third wave covering heavy-tail noise, tiny n_train, rare-positive classification, and non-LGB boosters. Stdout + `benchmarks/` JSON. Run: `python scripts/bench_slice_es_synthetics_v3.py`.
- **`bench_slice_es_100k_kfold10.py`** — the heavier n=100k, KFold(10) benchmark over CB/XGB/LGB for regression and binary classification. Stdout + `benchmarks/slice_es_100k_kfold10*.json` / log. Run: `python scripts/bench_slice_es_100k_kfold10.py`.
- **`bench_tiny_rerank_parallel.py`** — wall-time benchmark of the parallel `_tiny_model_rerank` path (serial vs `tiny_rerank_n_jobs=N`) on a 200k-row shape. Prints timings to stdout. Run: `python scripts/bench_tiny_rerank_parallel.py`.

## Coverage / tooling

- **`numba_coverage_report.py`** — parses a `coverage.xml` from the nightly `NUMBA_DISABLE_JIT=1` workflow and emits a structured JSON report of which `@njit` / `@cuda.jit` kernel lines become visible with JIT disabled, sorted worst-blinded first. Output JSON goes to the path you pass on the command line.
- **`run_numba_coverage.ps1`** / **`run_numba_coverage.sh`** — Windows / POSIX twins that run the test suite with `NUMBA_DISABLE_JIT=1` to collect kernel-body coverage, then invoke `numba_coverage_report.py`. Run: `pwsh scripts/run_numba_coverage.ps1` (or `bash scripts/run_numba_coverage.sh`).

## Pre-commit hook wrappers

- **`format_warn.py`** — warn-only formatting / lint check (`ruff format --check --diff` + `ruff check`) over staged Python files. Never rewrites files and always exits 0, surfacing differences as warnings without blocking the commit.
- **`run_meta_tests.py`** — wrapper for the meta-tests pre-commit hook; sets `NUMBA_DISABLE_JIT=1` + `MLFRAME_SKIP_NUMBA_PREWARM=1` before pytest so the AST / config / structure checks don't hang on CUDA driver init on dev machines.

## Removed: `scripts/repros/`

The former `scripts/repros/` directory held one-off reproduction scripts for
bugs fixed in 2026-05. Those bugs are now covered by the fuzz suite and by
`tests/training/test_mrmr_polars_fe.py`, so the standalone repros were removed
to avoid drift against the real regression tests.
