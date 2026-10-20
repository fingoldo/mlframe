"""``score_pair_mi`` routes ``estimator='minde'`` and ``'dpmine'`` to their neural estimators, and nothing else changes.

The neural estimators are replaced by spies, so the routing and the nats conversion are checked without training.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype


@pytest.mark.parametrize("estimator", ["minde", "dpmine"])
def test_score_pair_mi_routes_neural_estimator_by_name(monkeypatch, estimator):
    """The named neural estimator receives the arrays and its kwargs, and its value is returned as a float."""
    nn = importlib.import_module("mlframe.feature_selection.filters._neural_mi")
    calls = []

    def spy(x, y, **kwargs):
        """Record the call and return a fixed value."""
        calls.append((x.shape, y.dtype, dict(kwargs)))
        return 0.25

    monkeypatch.setattr(nn, f"{estimator}_mi", spy)
    mod = importlib.import_module("mlframe.feature_selection.filters._mi_dispatch")
    rng = np.random.default_rng(0)
    x = rng.normal(size=64)
    y = (x > 0).astype(np.int64)
    out = mod.score_pair_mi(x, y, estimator=estimator, estimator_kwargs={"seed": 7})
    assert out == 0.25
    assert calls == [((64,), np.dtype(_crit_np_dtype()), {"seed": 7})]


def test_score_pair_mi_still_rejects_unknown_estimator():
    """Unknown estimator names still raise, so the new branch did not swallow them."""
    mod = importlib.import_module("mlframe.feature_selection.filters._mi_dispatch")
    with pytest.raises(ValueError, match="unknown estimator"):
        mod.score_pair_mi(np.zeros(8), np.zeros(8), estimator="not_a_real_one")
