# Contributing to mlframe

Thanks for the interest. This document covers the workflow and the bar for code, tests, and docs.

## Development setup

```bash
git clone https://github.com/fingoldo/mlframe.git
cd mlframe

# Install pyutilz as a sibling editable dependency (not yet on PyPI):
git clone https://github.com/fingoldo/pyutilz.git ../pyutilz
pip install -e ../pyutilz

# Install mlframe with the extras you need + dev tooling:
pip install -e ".[boosting,calibration,viz,dev]"

# Install pre-commit hooks (detect-secrets + meta-tests):
pre-commit install
```

Python 3.9 is the floor. CI verifies 3.9 through 3.13.

## Running tests

```bash
pytest                                  # full suite
pytest -m fast                          # fast representative subset (~15s)
pytest -m "not slow and not gpu"        # what CI runs
pytest --cov=src/mlframe --cov-report=html
```

Markers in use: `fast`, `slow`, `integration`, `gpu`, `multigpu`, `benchmark`, `windows_only`, `linux_only`.

On Windows, `pytest-cov` occasionally fails with a `.coverage` file-lock `PermissionError` — pass `--no-cov` if you hit it.

## Code style

- `black` + `ruff` with line length 160 (configured in `pyproject.toml`).
- Type hints on new public APIs; do not weaken existing annotations.
- Comments belong only where the *why* is non-obvious — hidden constraints, subtle invariants, workarounds. Don't restate what the code already says.
- No `Co-Authored-By` lines in commits.

Run formatters and linters locally:

```bash
ruff check src/mlframe
black --check src/mlframe
mypy src/mlframe        # gradual; not blocking yet
bandit -r src/mlframe -ll
```

## What every new feature ships with

1. **Unit tests** covering the happy path and the obvious edges.
2. **A quantitative "business value" test** that locks in the win the feature is meant to deliver (accuracy lift, speed reduction, RAM saving) so a future change cannot silently regress it.
3. **A `@pytest.mark.fast` representative subset** that exercises every code path in <15s — so the full suite stays runnable in fast mode without blind spots.
4. **A `cProfile` hotspot check** for any non-trivial Python feature; if a hot loop is the bottleneck, consider `numba.njit`, parallelism, or `cupy` / `cuda` where it materially helps.

## Bug fixes

Every bug fix needs a regression test that **fails on pre-fix code** and **passes on post-fix code**. This applies to pre-existing bugs you find while working on something else, not just bugs you authored.

## sklearn compatibility

The composite-target wrapper surface (`mlframe.training.composite`, `mlframe.training.composite_estimator`) is tested across scikit-learn 1.5, 1.6, 1.7, 1.8 by the `sklearn-matrix` workflow. When touching that surface, expect that workflow to run and stay green.

## Commits

- Create new commits rather than amending; if a pre-commit hook fails, fix the issue and make a new commit.
- Use clear, conventional-ish prefixes (`fix:`, `feat:`, `perf:`, `refactor:`, `test:`, `docs:`, `ci:`, `chore:`).
- Reference issues with `#NNN` in the body when relevant.
- Do not skip hooks (`--no-verify`) without an explicit reason.

## Pull requests

1. Branch from `master`.
2. Run `pytest -m "not slow and not gpu"` locally before pushing.
3. PR title is short; the body explains *why* and lists the user-visible change.
4. Update [CHANGELOG.md](CHANGELOG.md) in the same PR if the change is user-facing.

## Reporting bugs

Open an issue at <https://github.com/fingoldo/mlframe/issues> with:

- mlframe version (`python -c "import mlframe; print(mlframe.__version__)"`)
- scikit-learn / lightgbm / xgboost / catboost versions
- A minimal repro
- The full traceback

## Security

If you find a vulnerability, please do not open a public issue. Email <fingoldo@gmail.com>.

## License

By contributing, you agree your contributions are licensed under the [MIT License](LICENSE).
