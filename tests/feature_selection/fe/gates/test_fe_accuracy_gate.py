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
    bin_y_for_class_mi,
    _FE_UPLIFT_MIN,
)


def _quadratic_problem(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = (x**2 > 1.0).astype(int)  # even-symmetric: x alone is near-useless linearly
    eng = (x**2)[:, None]  # the engineered He2-like feature linearises it
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


def test_bin_y_for_class_mi_string_labels_do_not_crash():
    """Regression test: string class labels (e.g. 'A'..'E') used to raise ``ValueError: invalid literal
    for int() with base 10`` from a bare ``arr.astype(np.int64)`` on non-numeric dtype -- numpy tries to
    parse each label as a decimal integer literal. Reproduced live via
    ``TestMultiClass.test_string_label_5class`` (MRMR(verbose=0).fit(X, y_vals) with 5 string labels)."""
    y = np.array(["A", "B", "C", "D", "E", "A", "B", "C"])
    codes = bin_y_for_class_mi(y)
    assert codes.dtype == np.int64
    assert codes.min() == 0
    assert codes.max() == 4
    # Dense 0..k-1 codes, same relative ordering as np.unique's sorted labels.
    np.testing.assert_array_equal(codes, [0, 1, 2, 3, 4, 0, 1, 2])


def test_bin_y_for_class_mi_numeric_classification_unchanged():
    """Numeric/bool classification labels must keep the direct int64 cast (bit-identical to before)."""
    y_int = np.array([0, 1, 2, 0, 1, 2])
    np.testing.assert_array_equal(bin_y_for_class_mi(y_int), y_int.astype(np.int64))
    y_bool = np.array([True, False, True, False])
    np.testing.assert_array_equal(bin_y_for_class_mi(y_bool), y_bool.astype(np.int64))


# ------------------------------------------------------ baseline-CV sibling cache
def test_sibling_baseline_cv_is_cached_and_equivalent():
    """Wave 13 finding #1: several engineered SIBLINGS derived from the same raw source
    (x__He2, x__He3, x__T2, x__L2, ...) share an IDENTICAL X_base/y/seed, so the
    ``_score(X_base)`` CV baseline is deterministic across sibling calls and must be
    computed ONCE, not refit from scratch for every sibling. This pins (1) the cached
    result is bit-identical to an uncached computation and (2) the sklearn baseline
    fit ``Ridge.fit``/``LogisticRegression.fit`` on ``X_base`` is invoked far fewer times
    across 4 siblings sharing one base than the naive 4x."""
    import sklearn.linear_model as _lm

    gate._BASELINE_CV_MEMO.clear()

    rng = np.random.default_rng(11)
    n = 2000
    x = rng.standard_normal(n)
    y = (x**2 > 1.0).astype(int)
    X_base = x[:, None]
    siblings = [
        (x**2)[:, None],
        (x**3)[:, None],
        np.sin(x)[:, None],
        np.abs(x)[:, None],
    ]

    fit_calls = {"n": 0}
    _orig_fit = _lm.LogisticRegression.fit

    def _counting_fit(self, *a, **k):
        fit_calls["n"] += 1
        return _orig_fit(self, *a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_lm.LogisticRegression, "fit", _counting_fit)
        ups_cached = [measure_feature_uplift(X_base, eng, y, classification=True, seed=0) for eng in siblings]
    calls_cached = fit_calls["n"]

    # Uncached reference: clear the memo before EVERY sibling so each recomputes its own baseline.
    fit_calls["n"] = 0
    ups_uncached = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_lm.LogisticRegression, "fit", _counting_fit)
        for eng in siblings:
            gate._BASELINE_CV_MEMO.clear()
            ups_uncached.append(measure_feature_uplift(X_base, eng, y, classification=True, seed=0))
    calls_uncached = fit_calls["n"]

    for a, b in zip(ups_cached, ups_uncached):
        assert a is not None and b is not None
        assert a == pytest.approx(b, abs=1e-12), "cached sibling baseline must be bit-identical to uncached"

    # 10-fold CV: uncached does 2 fits/fold (X_aug + X_base) x 10 folds x 4 siblings = 80.
    # cached does 2 fits/fold for sibling 1 (populates the cache) then 1 fit/fold (X_aug only)
    # for the remaining 3 siblings = 10*2 + 3*10*1 = 50 -- a real reduction, not just a memo no-op.
    assert calls_cached < calls_uncached, (calls_cached, calls_uncached)
    gate._BASELINE_CV_MEMO.clear()


def test_bin_y_for_class_mi_continuous_still_qcut_binned():
    """Continuous y must still route through qcut binning, unaffected by the string-label fix."""
    rng = np.random.default_rng(0)
    y = rng.standard_normal(2000)
    codes = bin_y_for_class_mi(y, nbins=10)
    assert codes.dtype == np.int64
    assert codes.min() >= 0
    assert codes.max() <= 9
    assert len(np.unique(codes)) > 1
