"""Direct behavioural tests for the FE held-out accuracy gate
(``_fe_accuracy_gate.measure_feature_uplift`` / ``keep_engineered_over_source``).

The gate was default-ON and consumed only by ``_mrmr_fit_impl`` but had ZERO
direct tests (gaps_fe_masking-03). It also shipped a fail-open/fail-closed
CONTRADICTION: ``keep_engineered_over_source``'s docstring promises "Fail-open
(True) on degenerate input" yet a probe exception (or any degenerate path)
returned ``0.0`` -> ``0.0 >= threshold`` is False -> the candidate was silently
DROPPED (fail-CLOSED). The fix makes every can't-measure path return ``None``
(the fail-open sentinel); these tests pin both the genuine win/redundant
decisions AND the fail-open / fail-closed contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _fe_accuracy_gate as gate
from mlframe.feature_selection.filters._fe_accuracy_gate import (
    keep_engineered_over_source,
    measure_feature_uplift,
    infer_classification,
    _FE_UPLIFT_MIN,
)


def _quadratic_problem(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = (x ** 2 > 1.0).astype(int)  # even-symmetric: x alone is near-useless linearly
    eng = (x ** 2)[:, None]         # the engineered He2-like feature linearises it
    return x[:, None], eng, y


# --------------------------------------------------------------------------- win
def test_uplift_positive_when_engineered_linearises_target():
    src, eng, y = _quadratic_problem()
    up = measure_feature_uplift(src, eng, y, classification=True)
    assert up is not None
    assert up > _FE_UPLIFT_MIN, f"x^2 must clear the uplift floor on an even-symmetric target (got {up})"
    assert keep_engineered_over_source(src.ravel(), eng, y) is True


# ---------------------------------------------------------------- redundant/no-win
def test_redundant_engineered_does_not_clear_floor():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(2000)
    y = (x + 0.1 * rng.standard_normal(2000) > 0).astype(int)  # linear in x
    eng = (2.0 * x + 1.0)[:, None]  # exact affine of x -> zero incremental linear usability
    up = measure_feature_uplift(x[:, None], eng, y, classification=True)
    assert up is not None
    assert up < _FE_UPLIFT_MIN, f"an affine copy of the source adds no held-out uplift (got {up})"
    assert keep_engineered_over_source(x, eng, y) is False


# ---------------------------------------------------------- degenerate -> fail-OPEN
@pytest.mark.parametrize("n", [10, 30])
def test_too_few_rows_returns_none_and_keeps(n):
    rng = np.random.default_rng(2)
    src = rng.standard_normal(n)
    eng = rng.standard_normal((n, 1))
    y = rng.integers(0, 2, n)
    assert measure_feature_uplift(src[:, None], eng, y, classification=True) is None
    assert keep_engineered_over_source(src, eng, y) is True  # fail-open, not dropped


def test_single_class_y_returns_none_and_keeps():
    rng = np.random.default_rng(3)
    src = rng.standard_normal(200)
    eng = rng.standard_normal((200, 1))
    y = np.zeros(200, dtype=int)  # one class
    assert measure_feature_uplift(src[:, None], eng, y, classification=True) is None
    assert keep_engineered_over_source(src, eng, y) is True


def test_shape_mismatch_returns_none_and_keeps():
    rng = np.random.default_rng(4)
    src = rng.standard_normal(200)
    eng = rng.standard_normal((150, 1))  # mismatched length
    y = rng.integers(0, 2, 200)
    assert measure_feature_uplift(src[:, None], eng, y, classification=True) is None
    assert keep_engineered_over_source(src, eng, y) is True


# ------------------------------------------------------------- MNAR -> fail-CLOSED
def test_missing_source_is_fail_closed():
    # >2% non-finite in the raw source: the signal may live in the NaN pattern
    # (MNAR) which the dropna'd probe cannot assess -> a transform of it must NOT
    # out-rank the raw column. This is a DELIBERATE fail-closed (distinct from the
    # exception/degenerate fail-open).
    rng = np.random.default_rng(5)
    src = rng.standard_normal(2000)
    src[: int(0.1 * 2000)] = np.nan
    eng = rng.standard_normal((2000, 1))
    y = rng.integers(0, 2, 2000)
    assert keep_engineered_over_source(src, eng, y) is False


# ----------------------------------------------- exception path -> fail-OPEN (regression)
def test_probe_exception_is_fail_open(monkeypatch):
    """A probe-internal exception must KEEP the candidate (fail-open), NOT drop it.
    Pre-fix the except branch returned 0.0 -> keep returned False (fail-closed),
    silently evicting a candidate whenever the probe raised."""
    src, eng, y = _quadratic_problem()

    import sklearn.linear_model as _lm

    class _Boom(_lm.LogisticRegression):
        def fit(self, *a, **k):
            raise RuntimeError("simulated probe failure")

    monkeypatch.setattr(gate, "measure_feature_uplift", measure_feature_uplift)
    monkeypatch.setattr("sklearn.linear_model.LogisticRegression", _Boom)

    up = measure_feature_uplift(src, eng, y, classification=True)
    assert up is None, "a probe exception must surface as None (the fail-open sentinel)"
    assert keep_engineered_over_source(src.ravel(), eng, y) is True


def test_infer_classification_basic():
    assert infer_classification(np.array([0, 1, 0, 1])) is True
    assert infer_classification(np.array(["a", "b", "a"])) is True
    assert infer_classification(np.linspace(0, 1, 500)) is False
