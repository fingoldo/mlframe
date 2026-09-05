"""`ESTransformedTargetRegressor()` must fit with no `regressor` argument.

`regressor=None` is the documented default, and the branch that handles it imported `LinearRegression` from
`mlframe.linear_model` -- a module that does not exist and never did; that line was its only mention anywhere
in the repository. Because the import is function-local and sits inside `if self.regressor is None:`, it was
invisible to import-time checks, and every existing test passed an explicit regressor, so nothing exercised
the default path at all.

The assertion is on the FIT SUCCEEDING and on the resulting estimator, not on the import: what the caller is
promised is that the default works, and pinning the module path would just re-freeze the typo.
"""

from __future__ import annotations

import numpy as np
import pytest


def _linear_fixture(n: int = 60):
    """A trivially learnable linear target, so a failure means the plumbing broke, not the fit."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 3))
    y = 2.0 * x[:, 0] - 1.5 * x[:, 1] + 0.25
    return x, y


def test_the_documented_default_fits_without_a_regressor():
    """No `regressor=` argument is the documented default and must reach a working inner estimator."""
    from mlframe.estimators.custom import ESTransformedTargetRegressor

    x, y = _linear_fixture()
    model = ESTransformedTargetRegressor().fit(x, y)

    assert hasattr(model, "regressor_"), "the default branch produced no inner regressor"
    assert hasattr(model.regressor_, "predict"), f"the default inner regressor is not an estimator: {type(model.regressor_).__name__}"


def test_the_default_predicts_the_linear_target_it_was_given():
    """The default path must not merely construct -- it has to produce a usable model.

    Loose bound on purpose: this pins that the default is WIRED, not the quality of a linear fit, so it
    cannot start failing because of an unrelated transformer change.
    """
    from mlframe.estimators.custom import ESTransformedTargetRegressor

    x, y = _linear_fixture()
    pred = ESTransformedTargetRegressor().fit(x, y).predict(x)

    assert np.shape(pred) == np.shape(y), f"expected one prediction per row, got {np.shape(pred)} for {np.shape(y)}"
    assert np.isfinite(pred).all(), "the default path produced non-finite predictions"
    assert np.corrcoef(np.asarray(pred).ravel(), y)[0, 1] > 0.9, "the default inner regressor did not learn a trivially linear target"


def test_an_explicit_regressor_still_wins():
    """The branch this fix touched is the `is None` arm; the explicit arm must be unaffected."""
    from sklearn.tree import DecisionTreeRegressor

    from mlframe.estimators.custom import ESTransformedTargetRegressor

    x, y = _linear_fixture()
    model = ESTransformedTargetRegressor(regressor=DecisionTreeRegressor(max_depth=2, random_state=0)).fit(x, y)

    assert isinstance(model.regressor_, DecisionTreeRegressor), f"an explicit regressor was replaced by {type(model.regressor_).__name__}"


@pytest.mark.parametrize("shape", [(40, 1), (40, 5)])
def test_the_default_handles_both_narrow_and_wide_inputs(shape):
    """A one-column frame and a several-column frame both go through the same default branch."""
    from mlframe.estimators.custom import ESTransformedTargetRegressor

    rng = np.random.default_rng(1)
    x = rng.normal(size=shape)
    y = x[:, 0] * 1.5

    ESTransformedTargetRegressor().fit(x, y)
