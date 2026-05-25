"""Verify sklearn-matrix CI runs the sklearn-compliance harness against every pinned sklearn minor.

`tests/test_sklearn_compliance.py` and `tests/test_sklearn_compliance_composite.py` parametrise `sklearn.utils.estimator_checks.check_estimator` over every estimator-shaped wrapper class (CompositeTargetEstimator, _LagPredictDeployableModel, ESTransformedTargetRegressor, EstimatorWithEarlyStopping, RFECV wrapper, PdOrdinalEncoder, PdKBinsDiscretizer). They need to fire on every sklearn minor in the matrix so a version-specific incompatibility surfaces immediately rather than at a customer site.
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SKLEARN_MATRIX_YAML = REPO_ROOT / ".github" / "workflows" / "sklearn-matrix-ci.yml"


def _read_yaml_text() -> str:
    return SKLEARN_MATRIX_YAML.read_text(encoding="utf-8")


def test_sklearn_matrix_workflow_exists() -> None:
    assert SKLEARN_MATRIX_YAML.is_file(), f"missing workflow file {SKLEARN_MATRIX_YAML}"


def test_sklearn_matrix_runs_compliance_basic() -> None:
    """tests/test_sklearn_compliance.py must be listed in a pytest invocation step."""

    text = _read_yaml_text()
    assert "tests/test_sklearn_compliance.py" in text, "sklearn-matrix workflow must run tests/test_sklearn_compliance.py on every pinned sklearn minor"


def test_sklearn_matrix_runs_compliance_composite() -> None:
    """tests/test_sklearn_compliance_composite.py must be listed in a pytest invocation step."""

    text = _read_yaml_text()
    assert "tests/test_sklearn_compliance_composite.py" in text, "sklearn-matrix workflow must run tests/test_sklearn_compliance_composite.py on every pinned sklearn minor"


def test_sklearn_matrix_pins_all_four_minors() -> None:
    """The matrix must still pin 1.5 / 1.6 / 1.7 / 1.8 so a sklearn minor regression on any of them is caught."""

    text = _read_yaml_text()
    for minor_prefix in ("1.5.", "1.6.", "1.7.", "1.8."):
        assert minor_prefix in text, f"sklearn-matrix must pin a {minor_prefix}x release"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--no-cov"]))
