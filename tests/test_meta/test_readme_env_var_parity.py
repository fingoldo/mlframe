"""Meta-test: every environment variable production code reads via
``os.environ.get(...)``/``os.getenv(...)`` is documented in README.md.

Uses the baseline/grandfather variant of the shared ``py_ci_shared.readme_env_var_parity``
check: mlframe has no "## Environment variables" section yet, so every var currently
read is grandfathered on first run -- only a NEW undocumented var (introduced after this
baseline was captured) fails. Documenting the grandfathered vars is a separate, deliberate
improvement this check doesn't demand up front.
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.readme_env_var_parity import assert_no_new_undocumented_env_vars

import mlframe

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent
README_PATH = Path(mlframe.__file__).resolve().parent.parent.parent / "README.md"
_BASELINE_PATH = Path(__file__).resolve().parent / "_readme_env_var_baseline.json"

# Mirrors this repo's pytest addopts (--ignore=legacy --ignore=benchmarks --ignore=profiling).
_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "tests", "legacy", "benchmarks", "profiling", "explore")


def _production_py_files() -> list[Path]:
    """Every production ``.py`` file under mlframe/, outside tests/legacy/benchmarks/profiling."""
    return [py for py in MLFRAME_DIR.rglob("*.py") if not any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS)]


def test_no_new_undocumented_env_vars():
    """Fail only on an env var read by production code that isn't in the grandfathered baseline."""
    assert_no_new_undocumented_env_vars(
        files=_production_py_files(),
        readme_path=README_PATH,
        baseline_path=_BASELINE_PATH,
    )
