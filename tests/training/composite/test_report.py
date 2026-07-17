"""Unit + biz_value tests for the unified composite explainability report.

``composite_report(estimator, X=None, y=None, fmt=...)`` is pure orchestration
over provenance / attribution / conformal helpers. The smoke tests assert each
section renders + is non-empty; the no-data path (X=None) still renders the
static sections; the biz_value test asserts the report for a fitted
``linear_residual`` composite contains the formula, the base share, and the
n_train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sklearn.linear_model import LinearRegression

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.report import composite_report


def _fit_linear_residual(n=400, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(10.0, 2.0, n)
    f1 = rng.normal(0.0, 1.0, n)
    # base-DOMINATED target so base_share is clearly high.
    y = 3.0 * base + 0.2 * f1 + rng.normal(0.0, 0.3, n)
    X = pd.DataFrame({"base": base, "f1": f1})
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="linear_residual",
        base_column="base",
    )
    est.fit(X, pd.Series(y))
    return est, X, pd.Series(y)


def test_report_renders_all_sections_with_data():
    est, X, y = _fit_linear_residual()
    md = composite_report(est, X)
    assert isinstance(md, str) and md.strip()
    for section in (
        "# Composite explainability report",
        "## Provenance",
        "## Fitted parameters",
        "## Base-vs-residual attribution",
        "## Conformal calibration",
        "## Diagnostics",
    ):
        assert section in md, f"missing section: {section}"
    # Each section non-empty: the line after each header is not blank-only.
    for section in ("## Provenance", "## Fitted parameters", "## Diagnostics"):
        idx = md.index(section)
        tail = md[idx + len(section) :].lstrip("\n")
        assert tail.strip(), f"empty section: {section}"


def test_report_no_data_renders_static_sections():
    est, _, _ = _fit_linear_residual()
    md = composite_report(est, X=None)
    assert "## Provenance" in md
    assert "## Fitted parameters" in md
    assert "## Conformal calibration" in md
    # The data-dependent sections degrade gracefully, not crash.
    assert "No data supplied" in md
    assert "y_clip_low" in md  # fitted params still listed


def test_report_html_format():
    est, X, _ = _fit_linear_residual()
    h = composite_report(est, X, fmt="html")
    assert h.startswith("<div")
    assert "<h1>" in h and "<h2>" in h and "<table" in h


def test_report_bad_fmt_raises():
    est, _, _ = _fit_linear_residual()
    with pytest.raises(ValueError):
        composite_report(est, fmt="latex")


def test_report_includes_conformal_coverage_when_y_given():
    est, X, y = _fit_linear_residual()
    # Calibrate on the same data (smoke: coverage table renders).
    est.calibrate_conformal(X, y, alpha=0.1)
    md = composite_report(est, X, y)
    assert "Interval coverage" in md
    assert "conformal" in md
    assert "empirical_cov" in md


def test_biz_val_report_contains_formula_base_share_and_n_train():
    """biz_value: the report for a fitted linear_residual composite contains
    the formula, the base share, and the n_train."""
    est, X, y = _fit_linear_residual(n=600)
    md = composite_report(est, X)

    # 1. The formula: linear_residual forward formula text.
    assert "T = " in md and "* base" in md, "forward formula not rendered"

    # 2. The base share: a base-DOMINATED target -> base_share should be high.
    from mlframe.training.composite.attribution import attribution_summary

    summ = attribution_summary(est, X)
    assert summ["base_share"] > 0.7, f"expected base-dominated, got {summ['base_share']}"
    assert "base share" in md
    # The rendered share value appears in the report.
    assert f"{summ['base_share']:.6g}" in md

    # 3. The n_train.
    n_train = est.fitted_params_["n_train_valid"]
    assert f"**n_train**: {n_train}" in md
