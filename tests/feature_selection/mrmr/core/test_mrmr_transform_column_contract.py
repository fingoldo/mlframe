"""transform()'s column contract: unfitted accessors raise, columns align by name, extras are ignored, missing selected columns raise by name.

Every existing test hands the identical frame object back to transform, so positional and by-name alignment were indistinguishable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from mlframe.feature_selection.filters.mrmr import MRMR


def _fitted(seed=0, n=500):
    """A light fit whose selection includes x0 and x1 (the signal columns)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(5)})
    y = ((X["x0"] + 0.8 * X["x1"]) > 0).astype(np.int64).to_numpy()
    MRMR._FIT_CACHE.clear()
    m = MRMR(random_seed=1, n_jobs=1, verbose=0, fe_max_steps=0, full_npermutations=3, baseline_npermutations=2).fit(X, y)
    return m, X


def test_get_feature_names_out_unfitted_raises_not_fitted():
    """sklearn contract: the accessor on an unfitted estimator raises NotFittedError."""
    with pytest.raises(NotFittedError):
        MRMR().get_feature_names_out()


def test_transform_with_reordered_columns_realigns_by_name():
    """The same columns in reverse order transform to exactly the same output as the fitted order."""
    m, X = _fitted()
    expected = np.asarray(m.transform(X), dtype=np.float64)
    got = np.asarray(m.transform(X[list(reversed(X.columns))]), dtype=np.float64)
    assert np.array_equal(got, expected), "transform aligned columns by position instead of by name"


def test_transform_with_extra_unseen_column_ignores_it_and_matches():
    """An appended unseen column leaves the output byte-identical."""
    m, X = _fitted(seed=2)
    expected = np.asarray(m.transform(X), dtype=np.float64)
    X_extra = X.assign(_unseen=np.random.default_rng(9).normal(size=len(X)))
    assert np.array_equal(np.asarray(m.transform(X_extra), dtype=np.float64), expected)


def test_transform_with_missing_fitted_column_raises_naming_it():
    """Dropping a selected column raises, and the message names that column."""
    m, X = _fitted(seed=3)
    selected = list(map(str, m.get_feature_names_out()))
    assert selected, "fixture precondition: something must be selected"
    dropped = selected[0]
    with pytest.raises((ValueError, RuntimeError), match=dropped):
        m.transform(X.drop(columns=[dropped]))
