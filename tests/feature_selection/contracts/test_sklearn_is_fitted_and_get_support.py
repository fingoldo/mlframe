"""Wave 9.1 loop-iter-43 regression: ``MRMR.__sklearn_is_fitted__``
and ``MRMR.get_support`` must honour sklearn API contracts.

Pre-fix at ``mrmr.py:142`` the class declared only
``BaseEstimator, TransformerMixin`` with no
``__sklearn_is_fitted__``. ``check_is_fitted()`` fell back to its
heuristic scanning for ANY trailing-underscore attr.
``_mrmr_fit_impl`` sets ``feature_names_in_`` / ``n_features_in_``
around line 241, but ``support_`` ~700 lines later at line 942 - so
a fit() that crashed mid-screen left a half-fit instance that
``check_is_fitted`` accepted but ``transform`` then refused with
``NotFittedError``. Silent contract divergence; downstream gates
saw "fitted" while transform refused.

Also: ``MRMR`` exposed no ``get_support()``. Any SelectorMixin
consumer (sklearn Pipeline introspection, RFECV, monitoring hooks)
that expected the documented sklearn-selector API was broken.

Fix at mrmr.py:1294 (class body):
- ``__sklearn_is_fitted__`` returns ``hasattr(self, "support_") and
  hasattr(self, "feature_names_in_")``.
- ``get_support(indices=False)`` returns a boolean mask of length
  ``n_features_in_`` (or indices when ``indices=True``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _fit_full():
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
            "c": rng.standard_normal(n),
        }
    )
    y = pd.Series((X["a"] > 0).astype(np.int64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(verbose=0).fit(X, y)
    return sel


def test_check_is_fitted_rejects_partial_fit():
    """A half-fit instance with feature_names_in_ but no support_
    must FAIL check_is_fitted (pre-fix it passed silently).
    """
    from mlframe.feature_selection.filters.mrmr import MRMR
    from sklearn.utils.validation import check_is_fitted
    from sklearn.exceptions import NotFittedError

    m = MRMR()
    m.feature_names_in_ = np.asarray(["a", "b", "c"], dtype=object)
    m.n_features_in_ = 3
    with pytest.raises(NotFittedError):
        check_is_fitted(m)


def test_check_is_fitted_accepts_full_fit():
    """A fully-fit instance must PASS check_is_fitted."""
    from sklearn.utils.validation import check_is_fitted

    sel = _fit_full()
    check_is_fitted(sel)  # Must not raise.


def test_get_support_boolean_mask():
    """``get_support()`` returns a boolean mask of length n_features_in_."""
    sel = _fit_full()
    mask = sel.get_support()
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == (sel.n_features_in_,)


def test_get_support_indices_match_support_array():
    """``get_support(indices=True)`` returns the same indices as
    ``self.support_``.
    """
    sel = _fit_full()
    idxs = sel.get_support(indices=True)
    assert set(int(i) for i in idxs) == set(int(i) for i in sel.support_)


def test_get_support_consistent_with_mask():
    """``np.where(get_support())[0]`` == ``get_support(indices=True)``."""
    sel = _fit_full()
    mask = sel.get_support()
    idxs = sel.get_support(indices=True)
    np.testing.assert_array_equal(np.where(mask)[0], idxs)


def test_get_support_on_unfitted_raises():
    """``get_support`` on an unfitted estimator must raise NotFittedError."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from sklearn.exceptions import NotFittedError

    m = MRMR()
    with pytest.raises(NotFittedError):
        m.get_support()
