"""The offset wrapper adds back exactly the base it removed: ``predict = regressor(selected features) + beta * u(X)``, and with no base recorded it is the plain regressor."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mlframe.feature_selection.filters._target_offset_regressor import TargetOffsetRegressor
from mlframe.feature_selection.filters.mrmr import MRMR


def _frame(seed, n=3000):
    """A frame with a linear base column ``u`` (slope 4) and a product term, plus a noise column; the target is their sum and noise."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n)
    x2, x3, nz = (rng.standard_normal(n) for _ in range(3))
    y = 4.0 * u + x2 * x3 + 0.1 * rng.standard_normal(n)
    return pd.DataFrame({"u": u, "x2": x2, "x3": x3, "nz": nz}), y


def test_predict_adds_back_the_recorded_base_exactly():
    """With a base recorded, the prediction differs from the regressor on the selected features by exactly beta times the base on the same rows."""
    X, y = _frame(0)
    sel = MRMR(verbose=0, n_jobs=1, random_state=0, fe_target_mix_enable=True, fe_target_mix_max_bases=1, fe_max_steps=1)
    # offset_alpha close to 1 accepts any offset whose held-out gain is not clearly negative: the invariant under test is the add-back, not the acceptance decision
    w = TargetOffsetRegressor(selector=sel, regressor=Ridge(alpha=1e-6), offset_alpha=0.999).fit(X, y)
    assert w.offset_base_ is not None
    Xte = X.iloc[:200]
    plain = w.regressor_.predict(w.selector_.transform(Xte))
    assert np.allclose(w.predict(Xte) - plain, w.offset_beta_ * Xte["u"].to_numpy(), atol=1e-9)


def test_no_base_gives_the_plain_regressor():
    """With the mix off no base is recorded, the offset is zero and the predictions are those of the regressor on the selected features."""
    X, y = _frame(1)
    sel = MRMR(verbose=0, n_jobs=1, random_state=1, fe_target_mix_enable=False, fe_max_steps=1)
    w = TargetOffsetRegressor(selector=sel, regressor=Ridge(alpha=1e-6)).fit(X, y)
    assert w.offset_base_ is None
    assert np.allclose(w.predict(X.iloc[:100]), w.regressor_.predict(w.selector_.transform(X.iloc[:100])))


def test_harmful_offset_is_rejected_on_held_out_rows():
    """A deliberately wrong offset (the base with the wrong slope) loses the held-out paired comparison, so the decision is False."""
    X, y = _frame(2)
    sel = MRMR(verbose=0, n_jobs=1, random_state=2, fe_target_mix_enable=False, fe_max_steps=1)
    w = TargetOffsetRegressor(selector=sel, regressor=Ridge(alpha=1e-6)).fit(X, y)
    w.offset_base_ = {"raw": "u"}
    w.offset_beta_ = -10.0
    Z = w.selector_.transform(X)
    even = np.arange(len(y)) % 2 == 0
    assert w._offset_helps_held_out(Z, y, X, even) is False
