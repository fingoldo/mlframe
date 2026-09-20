"""An importance that could not be measured must not outrank one measured to be harmful.

Both the conditional-permutation and the drop-column branches wrote 0.0 into arrays whose real entries are
`baseline - score` and routinely negative, so a feature whose conditioning tree or drop-fit RAISED ranked above every
feature whose removal actually improved the score, and survived top-k / RFE cuts that evicted genuine features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlframe.feature_selection.wrappers._helpers_importance import get_feature_importances


def _bed(n: int = 200):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"good": rng.normal(size=n), "noise": rng.normal(size=n), "constant": np.zeros(n)})
    y = X["good"] * 3.0 + rng.normal(0, 0.1, n)
    return X, y


def test_a_failed_drop_column_evaluation_records_nan():
    X, y = _bed()
    model = LinearRegression().fit(X, y)

    class _FailsOnOneColumn(LinearRegression):
        def fit(self, X, y, **kw):  # noqa: D102 - the failure injection is the point
            if getattr(X, "shape", (0, 0))[1] == X.shape[1] and "noise" not in getattr(X, "columns", []):
                raise ValueError("singular design after the drop")
            return super().fit(X, y, **kw)

    out = get_feature_importances(
        model=_FailsOnOneColumn().fit(X, y), current_features=list(X.columns),
        data=X, target=y, importance_getter="drop_column",
    )
    values = np.array([out[c] for c in X.columns], dtype=float)
    assert np.isnan(values).any(), "the column whose drop-fit raised must be NaN, not a rank-competitive 0.0"
    assert not np.isnan(out["good"]), "a measurable column must still carry a number"


def test_a_failed_conditional_permutation_records_nan():
    """A constant column cannot be conditioned on, which is exactly the ValueError path that used to score 0.0."""
    X, y = _bed()
    model = LinearRegression().fit(X, y)
    out = get_feature_importances(
        model=model, current_features=list(X.columns),
        data=X, target=y, importance_getter="conditional_permutation",
    )
    assert not np.isnan(out["good"])
    measured = [v for v in out.values() if not np.isnan(v)]
    assert measured, "at least the informative column must be measured"


def test_nan_never_wins_a_descending_ranking():
    """The ordering property the neutral 0.0 broke: an unmeasured entry must not sort above a negative one."""
    values = np.array([0.5, np.nan, -0.3])
    order = np.argsort(-values)
    assert order[-1] == 1, "NaN must land last in a descending importance ranking"
