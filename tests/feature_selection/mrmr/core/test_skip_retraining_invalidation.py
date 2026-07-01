"""Regression pins for the 2026-06-10 ``skip_retraining`` PARAMS-invalidation fix.

THE HOLE (user-hit, both selectors):
The in-object identity skip compared only DATA signatures:

  MRMR  (``_mrmr_fit_impl``):  ``(X.shape, y.shape, y_hash, x_hash, x_cols)``
  RFECV (``_rfecv_fit_init``): ``(X.shape, y.shape, columns_key, y_hash, x_hash)``

SELECTOR PARAMS were absent from BOTH. Refitting the SAME instance with
changed settings (``set_params`` or direct attribute assignment) on identical
data silently replayed the prior fit -- a selection computed under the OLD
params. The process-wide ``MRMR._FIT_CACHE`` already folded
``_hashable_params_signature``; the in-object layer did not (asymmetric
guarantees, same bug class as the 2026-05-30 X-content fix).

THE FIX: fold ``_hashable_params_signature(self.get_params(deep=True))`` into
both in-object signatures. ``get_params`` reads CURRENT attribute values at
fit time, so params changed after a previous fit are captured on the next
``fit`` call; ``deep=True`` also expands the wrapped estimator's params so
in-place estimator mutation invalidates RFECV's skip too.

Matrix pinned here (tiny fixtures, single-process):
  (a)  MRMR: same object, same X/y, CHANGED param -> refit happens, result
       reflects the NEW param (the user's exact bug).
  (a2) MRMR: changed numeric param (quantization_nbins) -> full fit re-runs
       (screen_predictors invoked again).
  (b)  MRMR: same object, same X, CHANGED y -> refit happens.
  (c)  MRMR: truly identical X/y/params -> skip DOES fire (perf preserved).
  (d)  cross-object _FIT_CACHE: clones with DIFFERENT params -> no stale
       replay; IDENTICAL clones -> cache hit (no full re-fit).
  (e)  RFECV versions of (a)-(c), incl. in-place wrapped-estimator mutation.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression


# ----------------------------------------------------------------------------------------------------------------------------
# Helpers / fixtures (tiny: n<=800, p<=4 -- RAM-light by design)
# ----------------------------------------------------------------------------------------------------------------------------


def _mrmr_data(seed: int = 0, n: int = 800):
    """Two independent informative features + one noise feature."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = ((x1 + x2) > 0).astype(np.int64)
    X = pd.DataFrame({
        "good1": x1,
        "good2": x2,
        "noise": rng.standard_normal(n),
    })
    return X, pd.Series(y)


class _ScreenCallCounter:
    """Context manager counting full-fit entries via ``screen_predictors``.

    ``_fit_impl`` lazily imports ``screen_predictors`` from ``.mrmr`` at every
    fit call, so rebinding the module attribute is picked up immediately. Both
    skip layers (in-object identity skip and ``_FIT_CACHE`` replay) return
    BEFORE ``screen_predictors`` -- a count increment proves a full re-fit ran.
    """

    def __init__(self):
        self.count = 0

    def __enter__(self):
        from mlframe.feature_selection.filters import mrmr as _mrmr_mod

        self._mod = _mrmr_mod
        self._orig = _mrmr_mod.screen_predictors

        def _counting(*args, **kwargs):
            self.count += 1
            return self._orig(*args, **kwargs)

        _mrmr_mod.screen_predictors = _counting
        return self

    def __exit__(self, *exc):
        self._mod.screen_predictors = self._orig
        return False


class CountingLR(LogisticRegression):
    """LogisticRegression with a class-level fit counter (survives sklearn clone)."""

    n_fits = 0

    def fit(self, *args, **kwargs):
        type(self).n_fits += 1
        return super().fit(*args, **kwargs)


def _rfecv_data(seed: int = 7, n: int = 200):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "x3": rng.standard_normal(n),
        "noise": rng.standard_normal(n),
    })
    y = ((X["x1"] + X["x2"]) > 0).astype(np.int64).to_numpy()
    return X, y


def _new_rfecv(**overrides):
    from mlframe.feature_selection.wrappers import RFECV

    kwargs = dict(
        estimator=CountingLR(solver="liblinear", random_state=0),
        cv=3,
        verbose=0,
        random_state=0,
        max_refits=4,
        leave_progressbars=False,
    )
    kwargs.update(overrides)
    return RFECV(**kwargs)


def _selected_names(rfecv, X=None):
    # ``must_exclude`` drops columns at fit entry, so ``support_`` lives in the SANITIZED
    # universe -- map through the fitted ``feature_names_in_``, never the caller's X columns.
    return [c for c, keep in zip(rfecv.feature_names_in_, rfecv.support_) if keep]


# ----------------------------------------------------------------------------------------------------------------------------
# (a) MRMR: changed param on the same object -> refit reflects the NEW param
# ----------------------------------------------------------------------------------------------------------------------------


def test_mrmr_refit_on_changed_param_same_data_reflects_new_param():
    """The user's exact bug: same object + same X/y + set_params -> the
    second fit must NOT replay the stale selection computed under old params."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _mrmr_data(seed=0)
    m = MRMR(verbose=0, n_jobs=1, fe_max_steps=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y)
        names_1 = list(m.get_feature_names_out())
        assert "good1" in names_1 and "good2" in names_1, f"fixture sanity: expected both informative features selected, got {names_1}"
        # Restrict the candidate pool via set_params -- ANY param change must invalidate the in-object skip.
        m.set_params(factors_names_to_use=["good1"])
        m.fit(X, y)
        names_2 = list(m.get_feature_names_out())
    assert "good2" not in names_2, (
        f"stale replay: selection still contains 'good2' after restricting "
        f"factors_names_to_use to ['good1'] on identical data; got {names_2}"
    )


def test_mrmr_refit_on_changed_param_via_attribute_assignment():
    """Direct attribute assignment (not set_params) must equally invalidate:
    get_params reads current attribute values at the next fit call."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _mrmr_data(seed=3)
    m = MRMR(verbose=0, n_jobs=1, fe_max_steps=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y)
        with _ScreenCallCounter() as counter:
            m.quantization_nbins = 4  # direct attribute assignment, the sneakiest path
            m.fit(X, y)
    assert counter.count >= 1, "changed quantization_nbins via attribute assignment did not trigger a full re-fit (stale identity skip)"


# ----------------------------------------------------------------------------------------------------------------------------
# (b) MRMR: changed y on the same object -> refit (different selection on a different target)
# ----------------------------------------------------------------------------------------------------------------------------


def test_mrmr_refit_on_changed_y_same_x():
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(2)
    n = 600
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    y_a = pd.Series((X["a"] > 0).astype(np.int64))
    y_b = pd.Series((X["b"] > 0).astype(np.int64))
    m = MRMR(verbose=0, n_jobs=1, fe_max_steps=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y_a)
        names_a = list(m.get_feature_names_out())
        m.fit(X, y_b)
        names_b = list(m.get_feature_names_out())
    assert names_a != names_b, f"changed TARGET did not change the selection: {names_a} == {names_b}"
    assert "a" in names_a and "b" in names_b


# ----------------------------------------------------------------------------------------------------------------------------
# (c) MRMR: truly identical X/y/params -> the fast path is preserved
# ----------------------------------------------------------------------------------------------------------------------------


def test_mrmr_identical_refit_still_skips():
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _mrmr_data(seed=1, n=500)
    m = MRMR(verbose=0, n_jobs=1, fe_max_steps=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y)
        names_1 = list(m.get_feature_names_out())
        with _ScreenCallCounter() as counter:
            m.fit(X, y)  # bit-identical inputs + unchanged params
        names_2 = list(m.get_feature_names_out())
    assert counter.count == 0, "identity skip regressed: identical refit ran a full fit"
    assert names_1 == names_2


# ----------------------------------------------------------------------------------------------------------------------------
# (d) cross-object _FIT_CACHE: different params -> no stale replay; identical clones -> cache hit
# ----------------------------------------------------------------------------------------------------------------------------


def test_fit_cache_no_stale_replay_between_clones_with_different_params():
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _mrmr_data(seed=4, n=500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m1 = MRMR(verbose=0, n_jobs=1, fe_max_steps=0)
        m1.fit(X, y)
        names_full = list(m1.get_feature_names_out())
        assert "good1" in names_full and "good2" in names_full
        # Different params + same data: the _FIT_CACHE key folds the params signature -> MUST miss.
        m2 = MRMR(verbose=0, n_jobs=1, fe_max_steps=0, factors_names_to_use=["good1"])
        with _ScreenCallCounter() as counter:
            m2.fit(X, y)
        names_restricted = list(m2.get_feature_names_out())
    assert counter.count >= 1, "_FIT_CACHE replayed across clones with DIFFERENT params"
    assert "good2" not in names_restricted, f"stale cross-object replay: {names_restricted}"


def test_fit_cache_hit_between_identical_clones():
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _mrmr_data(seed=5, n=500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m1 = MRMR(verbose=0, n_jobs=1, fe_max_steps=0)
        m1.fit(X, y)
        names_1 = list(m1.get_feature_names_out())
        m2 = MRMR(verbose=0, n_jobs=1, fe_max_steps=0)  # identical constructor params
        with _ScreenCallCounter() as counter:
            m2.fit(X, y)
        names_2 = list(m2.get_feature_names_out())
    assert counter.count == 0, "_FIT_CACHE miss for identical clones on identical data (perf feature regressed)"
    assert names_1 == names_2


# ----------------------------------------------------------------------------------------------------------------------------
# (e) RFECV versions of (a)-(c)
# ----------------------------------------------------------------------------------------------------------------------------


def test_rfecv_refit_on_changed_param_same_data_reflects_new_param():
    """RFECV's user-bug analogue: changed selector param + identical data must refit;
    the new support_ must reflect the NEW param (must_exclude drops a previously-kept feature)."""
    X, y = _rfecv_data(seed=7)
    rfecv = _new_rfecv()
    CountingLR.n_fits = 0
    rfecv.fit(X, y)
    fits_after_first = CountingLR.n_fits
    assert fits_after_first > 0
    assert "x1" in _selected_names(rfecv, X), "fixture sanity: informative 'x1' should be selected initially"
    rfecv.set_params(must_exclude=["x1"])
    rfecv.fit(X, y)
    assert CountingLR.n_fits > fits_after_first, "changed selector param on identical data did not trigger a re-fit (stale identity skip)"
    assert "x1" not in _selected_names(rfecv, X), "support_ does not reflect the NEW must_exclude param -- stale selection replayed"


def test_rfecv_refit_on_changed_pure_selector_param():
    """Pure selector-param change with NO X/y change at all must invalidate the skip.

    Unlike ``must_exclude`` (which drops columns from the sanitized X, so the DATA slots of the
    signature already differ), ``n_features_selection_rule`` leaves the data signature untouched --
    pre-fix this was the exact RFECV stale-replay hole (params absent from the signature)."""
    X, y = _rfecv_data(seed=11)
    rfecv = _new_rfecv()
    rfecv.fit(X, y)
    CountingLR.n_fits = 0
    rfecv.set_params(n_features_selection_rule="one_se_min")
    rfecv.fit(X, y)
    assert CountingLR.n_fits > 0, "pure selector-param change on identical data did not trigger a re-fit (stale identity skip)"


def test_rfecv_refit_on_inplace_estimator_param_mutation():
    """deep=True coverage: mutating the WRAPPED estimator's hyperparams in place
    (same estimator object) must invalidate RFECV's identity skip."""
    X, y = _rfecv_data(seed=8)
    rfecv = _new_rfecv()
    CountingLR.n_fits = 0
    rfecv.fit(X, y)
    fits_after_first = CountingLR.n_fits
    rfecv.estimator.set_params(C=0.01)  # in-place mutation; id(estimator) unchanged
    rfecv.fit(X, y)
    assert CountingLR.n_fits > fits_after_first, "in-place estimator hyperparam change on identical data did not trigger a re-fit"


def test_rfecv_refit_on_changed_y_same_x():
    X, y1 = _rfecv_data(seed=9)
    y2 = ((X["x3"]) > 0).astype(np.int64).to_numpy()
    rfecv = _new_rfecv()
    CountingLR.n_fits = 0
    rfecv.fit(X, y1)
    fits_after_first = CountingLR.n_fits
    rfecv.fit(X, y2)
    assert CountingLR.n_fits > fits_after_first, "changed TARGET content on same X did not trigger a re-fit"
    assert "x3" in _selected_names(rfecv, X), f"selection does not reflect the new target: {_selected_names(rfecv, X)}"


def test_rfecv_identical_refit_still_skips():
    X, y = _rfecv_data(seed=10)
    rfecv = _new_rfecv()
    rfecv.fit(X, y)
    names_1 = _selected_names(rfecv, X)
    CountingLR.n_fits = 0
    rfecv.fit(X, y)  # bit-identical inputs + unchanged params
    assert CountingLR.n_fits == 0, "RFECV identity skip regressed: identical refit re-ran estimator fits"
    assert _selected_names(rfecv, X) == names_1
