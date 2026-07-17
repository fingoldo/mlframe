"""Regression sensor for S26: ``CompositeTargetEstimator.from_fitted_inner``
assigns ``estimator_`` / ``fitted_params_`` outside the ``__init__`` signature.
``sklearn.base.clone()`` copies only init params, so a clone of a
from_fitted_inner-built instance is a SILENT unfitted estimator — the first
``predict`` call on the clone raises ``NotFittedError`` (or, worse, returns a
garbage prediction if the inner base_estimator happens to be re-usable).

Fix (Variant A from the audit): override ``__sklearn_clone__`` on the
class so a clone-attempt on a from_fitted_inner instance raises with a
concrete actionable message. Standard ``fit()``-built instances must still
clone normally (regression guard for the standard sklearn flow).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sklearn.base import clone
from sklearn.linear_model import LinearRegression


@pytest.fixture
def fitted_inner_kit():
    """A small fitted LinearRegression on a 2-column frame so the inner
    has feature_names_in_ and is callable for predict checks.
    """
    X = pd.DataFrame({"base": [1.0, 2.0, 3.0, 4.0, 5.0], "f1": [0.5, 1.0, 1.5, 2.0, 2.5]})
    y = np.array([1.5, 3.0, 4.5, 6.0, 7.5])
    inner = LinearRegression()
    inner.fit(X, y)
    return X, y, inner


def test_clone_of_from_fitted_inner_raises_with_actionable_message(fitted_inner_kit):
    """The post-hoc ``from_fitted_inner`` path is fundamentally
    incompatible with ``sklearn.clone`` because fitted state lives on
    underscore-suffixed attrs outside the init signature. Refuse clone
    with a clear message instead of returning a silent unfitted shell.
    """
    from mlframe.training.composite import CompositeTargetEstimator

    _X, y, inner = fitted_inner_kit
    cte = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner,
        transform_name="diff",
        base_column="base",
        transform_fitted_params={},
        y_train=y,
    )
    # Sanity: source is fitted
    assert hasattr(cte, "estimator_")
    assert hasattr(cte, "fitted_params_")

    # Clone should refuse.
    with pytest.raises((TypeError, NotImplementedError, RuntimeError)) as exc_info:
        clone(cte)
    msg = str(exc_info.value).lower()
    assert "from_fitted_inner" in msg or "clone" in msg, f"Refusal message must mention from_fitted_inner / clone; got: {exc_info.value!r}"


def test_clone_of_standard_fit_built_instance_still_works(fitted_inner_kit):
    """Regression guard: standard sklearn flow (unfitted instance ->
    pipeline -> clone -> fit) must keep working. Only from_fitted_inner
    instances are off-limits to clone.
    """
    from mlframe.training.composite import CompositeTargetEstimator

    _X, _y, _ = fitted_inner_kit
    cte = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="diff",
        base_column="base",
    )
    # No fit() yet -- this is the path sklearn.Pipeline / GridSearchCV uses.
    cloned = clone(cte)
    assert cloned is not cte
    # Clone must be a fresh, unfitted instance carrying the same init params.
    assert cloned.transform_name == "diff"
    assert cloned.base_column == "base"
    assert type(cloned.base_estimator) is LinearRegression
    # Clone must NOT carry any underscore fitted state.
    assert not hasattr(cloned, "estimator_")
    assert not hasattr(cloned, "fitted_params_")


def test_fitted_instance_via_fit_clones_to_unfitted_shell(fitted_inner_kit):
    """A wrapper that was fitted via its OWN ``fit()`` (not via
    from_fitted_inner) is the standard sklearn clone scenario: clone
    returns an unfitted shell carrying the init params, fitted state
    is dropped. This is the desired behaviour and must not regress.
    """
    from mlframe.training.composite import CompositeTargetEstimator

    X, y, _ = fitted_inner_kit
    cte = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="diff",
        base_column="base",
    )
    cte.fit(X, y)
    # Now fitted.
    assert hasattr(cte, "estimator_")
    # sklearn.clone of a fit()-built instance must succeed (drops fitted state).
    cloned = clone(cte)
    assert cloned is not cte
    assert cloned.transform_name == "diff"
    assert not hasattr(cloned, "estimator_")
