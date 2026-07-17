"""Wave 11a monolith-split sensor for ``mlframe.training.core._phase_composite_post``.

Carve pattern: ``_LagPredictDeployableModel`` extracted to sibling. Parent re-exports the class so existing imports (``from ._phase_composite_post import _LagPredictDeployableModel``) keep resolving. Identity preserved for downstream isinstance / sklearn.clone integration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def parent_module():
    """Parent module."""
    from mlframe.training.core import _phase_composite_post

    return _phase_composite_post


@pytest.fixture(scope="module")
def lag_sibling():
    """Lag sibling."""
    from mlframe.training.core import _phase_composite_post_lag_predict

    return _phase_composite_post_lag_predict


def test_lag_predict_identity(parent_module, lag_sibling):
    """Lag predict identity."""
    assert parent_module._LagPredictDeployableModel is lag_sibling._LagPredictDeployableModel


def test_facade_loc_budget(parent_module):
    """Facade loc budget."""
    path = Path(parent_module.__file__)
    n_lines = len(path.read_text(encoding="utf-8").splitlines())
    assert n_lines < 1000, f"facade is {n_lines} LOC, expected < 1000"


def test_lag_predict_smoke_round_trip(parent_module):
    """Exercise the carved class end-to-end: get_params, set_params, fit, predict."""
    model = parent_module._LagPredictDeployableModel(lag_column="lag_target")
    assert model.get_params() == {"lag_column": "lag_target"}

    # set_params: only lag_column is accepted
    model.set_params(lag_column="other")
    assert model.lag_column == "other"
    model.set_params(lag_column="lag_target")

    # set_params: reject unknown
    with pytest.raises(ValueError, match="no parameter"):
        model.set_params(unknown_param=1)

    # fit no-op + predict on pandas
    X = pd.DataFrame({"lag_target": [1.5, 2.5, 3.5], "other": [9, 9, 9]})
    model.fit(X, y=None)
    pred = model.predict(X)
    np.testing.assert_array_equal(pred, np.array([1.5, 2.5, 3.5]))


def test_lag_predict_missing_column_raises(parent_module):
    """Lag predict missing column raises."""
    model = parent_module._LagPredictDeployableModel(lag_column="missing")
    X = pd.DataFrame({"present": [1, 2, 3]})
    with pytest.raises(KeyError, match="missing"):
        model.predict(X)


def test_lag_predict_sklearn_clone_compatible(parent_module):
    """Must still be sklearn.clone()-able since the cross-target ensemble path clones the component during honest-OOF refit. Pre-fix the clone failed -> NNLS dropped the component -> ensemble landed at suboptimal RMSE."""
    from sklearn.base import clone

    original = parent_module._LagPredictDeployableModel(lag_column="lag_a")
    cloned = clone(original)
    assert isinstance(cloned, parent_module._LagPredictDeployableModel)
    assert cloned.lag_column == "lag_a"
    # Independence: mutating clone doesn't affect original
    cloned.lag_column = "lag_b"
    assert original.lag_column == "lag_a"
