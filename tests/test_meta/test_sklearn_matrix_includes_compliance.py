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
    """The matrix must pin every SUPPORTED sklearn minor so a minor-specific regression is caught.

    1.5 is intentionally NOT pinned: category-encoders (a hard mlframe dependency) does an unconditional
    ``from sklearn.utils import Tags`` at module top, and ``sklearn.utils.Tags`` only exists from 1.6, so
    sklearn 1.5 cannot import mlframe at all. The supported floor is therefore 1.6 and the matrix pins
    1.6 / 1.7 / 1.8 (mirrored by the workflow's own floor comment).
    """

    text = _read_yaml_text()
    for minor_prefix in ("1.6.", "1.7.", "1.8."):
        assert minor_prefix in text, f"sklearn-matrix must pin a {minor_prefix}x release"
    assert "1.5." not in text, "sklearn 1.5 cannot import mlframe (category-encoders needs sklearn.utils.Tags from 1.6); it must NOT be pinned"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--no-cov"]))
