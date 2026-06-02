"""RFECV Pareto-front diagnostics (read-only): pareto_front_ + pareto_knee_.

These are additive read-outs over cv_results_ + the stability curve; they change NO default (the knee
was benchmarked and LOST 0/6 on downstream AUC vs argmax/one_se_max, so it is a parsimony diagnostic only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_rfecv_pareto_front_and_knee():
    pytest.importorskip("sklearn")
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV, FIConfig, SearchConfig

    X, y = make_classification(n_samples=1200, n_features=16, n_informative=6, n_redundant=4, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(16)])
    r = RFECV(
        estimator=RandomForestClassifier(n_estimators=40, max_depth=6, random_state=0),
        cv=3, scoring=None, verbose=0,
        fi_config=FIConfig(importance_getter="feature_importances_", n_features_selection_rule="one_se_max"),
        search_config=SearchConfig(max_refits=12, max_runtime_mins=2), random_state=0,
    )
    r.fit(X, pd.Series(y))

    front = r.pareto_front_()
    assert isinstance(front, list) and front, "front should be non-empty after a fit"
    ns = [p["n"] for p in front]
    assert ns == sorted(ns), "front must be sorted by n"
    assert all(p["n"] > 0 and np.isfinite(p["mean"]) for p in front)
    # Pareto property: no front point dominates another on (mean MAX, n MIN).
    for a in front:
        for b in front:
            if a is b:
                continue
            assert not (b["mean"] >= a["mean"] and b["n"] <= a["n"] and (b["mean"] > a["mean"] or b["n"] < a["n"])), \
                "front contains a dominated point"
    knee = r.pareto_knee_()
    assert knee in ns, "knee must be one of the front's N values"
    # Read-only: default selection rule unaffected (support_ came from one_se_max, not the knee).
    assert r.n_features_ >= 1
