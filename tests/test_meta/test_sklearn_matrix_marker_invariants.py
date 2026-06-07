"""B1 sklearn_matrix marker convention meta-test.

The multi-sklearn-version CI matrix (``.github/workflows/sklearn-matrix-ci.yml``) selects tests via
``pytest -m sklearn_matrix`` rather than an explicit per-file list. A new composite-target test file added without
``pytestmark = pytest.mark.sklearn_matrix`` would silently drop out of the matrix and a breaking sklearn change would
ship undetected.

This meta-test pins:

1. The marker is registered in ``pyproject.toml`` (catches a typo in the marker name across either site).
2. Every ``tests/training/test_composite*.py`` file carries the marker at module level (the convention scope as of
   Wave 17 / B1 cleanup -- composite-target wrapper + delegate-property surface is the historical sklearn-version
   bite point).
3. At least N tests are collected under ``-m sklearn_matrix`` (rough drift signal: if the count drops below the floor,
   either a file lost the marker or the test bodies collapsed).
"""
from __future__ import annotations

from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# Explicit allowlist of composite-target test files that the multi-sklearn CI matrix exercises (matches the file list
# historically enumerated in ``.github/workflows/sklearn-matrix-ci.yml`` before B1 marker-based selection landed; the
# workflow now uses ``pytest -m sklearn_matrix`` and this allowlist becomes the source of truth for the marker
# convention). Adding a file to the matrix is a two-step opt-in: declare ``pytestmark = pytest.mark.sklearn_matrix`` at
# module level AND append the path here. Other ``test_composite*.py`` files (transforms unit-tests, bayesian-alpha
# helpers, bootstrap, etc.) deliberately stay outside the matrix because they exercise mlframe-internal helpers, not
# sklearn delegate-property surface.
_SKLEARN_MATRIX_FILES = (
    "tests/training/test_composite.py",
    "tests/training/test_composite_polish.py",
    "tests/training/test_composite_polish_refinement.py",
    "tests/training/composite/test_composite_provenance.py",
    "tests/training/test_composite_perf.py",
    "tests/training/test_composite_business_value_locks.py",
    "tests/training/composite/test_composite_discovery.py",
    "tests/training/test_composite_gate_and_edges.py",
    "tests/training/composite/test_composite_ensemble.py",
    "tests/training/test_composite_x_feature_selection.py",
)


def test_sklearn_matrix_marker_registered_in_pyproject():
    """The marker name MUST appear in ``[tool.pytest.ini_options].markers`` so ``--strict-markers`` (set in pyproject)
    does not reject ``pytestmark = pytest.mark.sklearn_matrix`` at collection."""
    content = _PYPROJECT.read_text(encoding="utf-8")
    assert '"sklearn_matrix:' in content, (
        "marker ``sklearn_matrix`` is not registered in pyproject.toml [tool.pytest.ini_options].markers; "
        "add it so ``--strict-markers`` does not reject the module-level ``pytestmark`` on composite_*.py files."
    )


def test_every_allowlisted_file_declares_marker():
    """Every file in ``_SKLEARN_MATRIX_FILES`` MUST contain ``pytest.mark.sklearn_matrix`` at module level so the CI
    ``pytest -m sklearn_matrix`` selection collects it. Files outside the allowlist may carry the marker too (it's the
    sole source of truth for selection at runtime); the allowlist is a floor, not a ceiling."""
    missing = []
    for rel in _SKLEARN_MATRIX_FILES:
        path = _REPO_ROOT / rel
        assert path.is_file(), f"allowlisted file does not exist: {rel} (rename / delete should also update _SKLEARN_MATRIX_FILES)"
        if "pytest.mark.sklearn_matrix" not in path.read_text(encoding="utf-8"):
            missing.append(rel)
    assert not missing, (
        "the following allowlisted composite-target test files do NOT declare ``pytest.mark.sklearn_matrix`` at "
        "module level; the multi-sklearn CI matrix selects via ``-m sklearn_matrix`` so an unmarked file will "
        "silently skip the matrix and a breaking sklearn change will ship undetected:\n  "
        + "\n  ".join(missing)
    )


def test_workflow_uses_marker_selection():
    """``.github/workflows/sklearn-matrix-ci.yml`` MUST use ``-m sklearn_matrix`` selection (not an explicit
    file list) so adding a file to ``_SKLEARN_MATRIX_FILES`` and pasting the marker is enough -- no third edit
    site in YAML."""
    workflow = _REPO_ROOT / ".github" / "workflows" / "sklearn-matrix-ci.yml"
    if not workflow.is_file():
        pytest.skip(f"workflow not present at {workflow}; convention can't be verified")
    content = workflow.read_text(encoding="utf-8")
    assert "-m sklearn_matrix" in content, (
        "sklearn-matrix-ci.yml does not invoke ``pytest -m sklearn_matrix``; if you switched back to an explicit "
        "file list, update the allowlist comments here and consider removing the marker convention entirely."
    )
