"""Cluster 15 regressions: a TypeError raised inside a user importance callable is not retried, and a callable prescreen cannot inject names the search never offered (audit 2026-11-20, findings 15-F8, 15-F9)."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.wrappers._helpers_importance import get_feature_importances


def test_typeerror_inside_a_user_importance_callable_runs_once():
    """A callable without a target parameter that raises TypeError used to be re-run without target (15-F9)."""
    calls = {"n": 0}

    def getter(model, data, reference_data):
        """Fail with a TypeError on the first call."""
        calls["n"] += 1
        raise TypeError("a real bug inside the callable")

    X = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
    with pytest.raises(TypeError, match="a real bug"):
        get_feature_importances(model=object(), current_features=["a", "b"], importance_getter=getter, data=X, reference_data=X, target=np.zeros(10))
    assert calls["n"] == 1
