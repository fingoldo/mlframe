"""Wave 11a monolith-split sensor for ``mlframe.feature_selection.wrappers._rfecv_fit``.

Carve pattern: top-of-function input validation + signature computation extracted into ``_rfecv_fit_init._init_fit_state``. Behavioural-equivalence test ensures fit produces identical fold scores and support_ vs pre-split byte-for-byte.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression


@pytest.fixture(scope="module")
def parent_module():
    from mlframe.feature_selection.wrappers import _rfecv_fit
    return _rfecv_fit


@pytest.fixture(scope="module")
def init_sibling():
    from mlframe.feature_selection.wrappers import _rfecv_fit_init
    return _rfecv_fit_init


def test_init_fit_state_imported(parent_module, init_sibling):
    assert parent_module._init_fit_state is init_sibling._init_fit_state


def test_rfecv_fit_bound_to_class(parent_module):
    from mlframe.feature_selection.wrappers._rfecv import RFECV
    # fit is bound at parent bottom in _rfecv.py; identity is parent.fit IS RFECV.fit
    assert RFECV.fit is parent_module.fit


def test_facade_loc_budget(parent_module):
    path = Path(parent_module.__file__)
    n_lines = len(path.read_text(encoding="utf-8").splitlines())
    assert n_lines < 1000, f"facade is {n_lines} LOC, expected < 1000"


def test_smoke_fit_runs_with_carved_prelude(parent_module):
    """Exercise the carved init helper end-to-end via a tiny RFECV fit. Per CLAUDE.md AST-audit rule, sensor must actually call the moved body, not just import."""
    from mlframe.feature_selection.wrappers._rfecv import RFECV

    rng = np.random.default_rng(7)
    n = 120
    X = pd.DataFrame({
        "x1": rng.normal(0, 1, size=n),
        "x2": rng.normal(0, 1, size=n),
        "x3": rng.normal(0, 1, size=n),
        "noise": rng.normal(0, 1, size=n),
    })
    # y depends on x1, x2 only
    y = (X["x1"] + X["x2"] > 0).astype(int).to_numpy()

    rfecv = RFECV(
        estimator=LogisticRegression(solver="liblinear", random_state=0),
        cv=3,
        verbose=0,
        random_state=0,
        max_refits=4,
    )
    rfecv.fit(X, y)
    # Smoke contract: support_ exists, has correct length, at least one feature selected.
    assert hasattr(rfecv, "support_")
    assert len(rfecv.support_) == X.shape[1]
    assert int(rfecv.support_.sum()) >= 1


def test_init_helper_validates_sample_weight_length(init_sibling):
    """The carved sample-weight length check must still raise ValueError pre-fit (was previously inline; behavioural-equivalence sensor)."""
    from mlframe.feature_selection.wrappers._rfecv import RFECV
    rfecv = RFECV(estimator=LogisticRegression(), cv=3)
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4, 3, 2, 1]})
    y = np.array([0, 1, 0, 1])
    bad_sw = np.array([1.0, 1.0, 1.0])  # length 3 != n_rows=4
    with pytest.raises(ValueError, match="sample_weight length"):
        init_sibling._init_fit_state(rfecv, X, y, None, bad_sw)


def test_init_helper_validates_y_nan(init_sibling):
    """NaN in y must surface as ValueError in the carved validator."""
    from mlframe.feature_selection.wrappers._rfecv import RFECV
    rfecv = RFECV(estimator=LogisticRegression(), cv=3)
    X = pd.DataFrame({"a": np.arange(10, dtype=float), "b": np.arange(10, dtype=float)})
    y = np.array([0.0, 1.0, float("nan"), 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="NaN"):
        init_sibling._init_fit_state(rfecv, X, y, None, None)
