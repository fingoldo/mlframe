"""Stability selection must measure importance on rows the bootstrap estimator did NOT fit on.

It computed permutation importance on the same subsample it had just fitted, although `importance_getter='auto'` is
documented as held-out and the CV-fold path passes a held-out split. A tree ensemble memorises a high-cardinality
ID-like column on each subsample; permuting it on those same rows destroys the memorised mapping, so its importance is
large in every bootstrap, `selection_freq` reaches 1.0 and pure noise enters `support_`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.wrappers.rfecv._stability_select import _fit_stability_selection


def test_importance_is_measured_on_the_held_out_complement(monkeypatch):
    """The rows handed to the importance call must be disjoint from the rows the estimator was fitted on."""
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=[f"r{i}" for i in range(n)])
    y = pd.Series(X["a"] * 2 + rng.normal(0, 0.1, n), index=X.index)

    fitted_rows: list[set] = []
    scored_rows: list[set] = []

    class _RecordingTree(DecisionTreeRegressor):
        def fit(self, X, y, **kw):  # noqa: D102 - recording is the point
            fitted_rows.append(set(getattr(X, "index", range(len(X)))))
            return super().fit(X, y, **kw)

    import mlframe.feature_selection.wrappers.rfecv._stability_select as ss

    original = ss.get_feature_importances

    def recording_importances(*args, **kwargs):
        data = kwargs.get("data")
        scored_rows.append(set(getattr(data, "index", range(len(data)))))
        return original(*args, **kwargs)

    monkeypatch.setattr(ss, "get_feature_importances", recording_importances)

    class _Holder:
        """The attributes `_fit_stability_selection` reads off its estimator object."""

        stability_n_bootstrap = 3
        stability_threshold = 0.6
        stability_top_k = 1
        verbose = 0
        must_include = None
        n_repeats = 2
        random_state = 0
        estimators = None
        estimator = _RecordingTree(random_state=0)
        importance_getter = "auto"

    # The signature tuple is only rewritten at the very end; a two-element placeholder is enough to reach it.
    try:
        _fit_stability_selection(_Holder(), X, y, signature=("features", "params"))
    except Exception as exc:  # the bootstrap loop has already run by then; bookkeeping after it is not under test
        assert fitted_rows, f"the bootstrap loop did not run: {exc}"

    assert fitted_rows and scored_rows and len(fitted_rows) == len(scored_rows)
    pairs = list(zip(fitted_rows, scored_rows))
    assert pairs and len(pairs) == 3, "one importance call per bootstrap"
    for fit_rows, score_rows in pairs:
        assert score_rows, "the importance call must receive rows"
        assert not (fit_rows & score_rows), "importance must be measured out of bag, not on the fitted rows"
