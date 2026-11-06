"""``fe_max_steps=0`` means no feature engineering: the residual-rescue target mix does not fit its child MRMR either."""

from __future__ import annotations

import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._target_mix_fe as tm
from mlframe.feature_selection.filters.mrmr import MRMR

# The budget under test, named so the conflict with the family flag below reads as the contract being exercised, not a leftover pin.
_NO_FE_BUDGET = 0


def test_target_mix_child_not_fit_without_fe(monkeypatch):
    """With the target mix enabled but ``fe_max_steps=0`` the second pass never runs and no engineered feature is returned."""
    calls = []

    def _spy(*args, **kwargs):
        """Record the call and propose nothing."""
        calls.append(1)
        return {}, []

    monkeypatch.setattr(tm, "second_pass_recipes", _spy)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((600, 6)), columns=[f"x{i}" for i in range(6)])
    y = X["x0"] * X["x1"] + 0.5 * X["x2"] + 0.1 * rng.standard_normal(600)
    sel = MRMR(verbose=0, random_state=0, fe_max_steps=_NO_FE_BUDGET, fe_target_mix_enable=True).fit(X, y)
    assert calls == []
    assert not list(getattr(sel, "_engineered_features_", None) or [])
