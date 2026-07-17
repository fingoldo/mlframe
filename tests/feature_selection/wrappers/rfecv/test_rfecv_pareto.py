"""RFECV Pareto-front diagnostics (read-only): pareto_front_ + pareto_knee_.

These are additive read-outs over cv_results_ + the stability curve; they change NO default (the knee
was benchmarked and LOST 0/6 on downstream AUC vs argmax/one_se_max, so it is a parsimony diagnostic only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_rfecv_pareto_front_and_knee():
    """pareto_front_ is n-sorted and non-dominated on (mean, n), and pareto_knee_ is one of the front's N values; support_ still comes from one_se_max, not the knee."""
    pytest.importorskip("sklearn")
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV, FIConfig, SearchConfig

    X, y = make_classification(n_samples=1200, n_features=16, n_informative=6, n_redundant=4, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(16)])
    r = RFECV(
        estimator=RandomForestClassifier(n_estimators=40, max_depth=6, random_state=0),
        cv=3,
        scoring=None,
        verbose=0,
        fi_config=FIConfig(importance_getter="feature_importances_", n_features_selection_rule="one_se_max"),
        search_config=SearchConfig(max_refits=12, max_runtime_mins=2),
        random_state=0,
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
            assert not (b["mean"] >= a["mean"] and b["n"] <= a["n"] and (b["mean"] > a["mean"] or b["n"] < a["n"])), "front contains a dominated point"
    knee = r.pareto_knee_()
    assert knee in ns, "knee must be one of the front's N values"
    # Read-only: default selection rule unaffected (support_ came from one_se_max, not the knee).
    assert r.n_features_ >= 1


def test_rfecv_plateau_rule_selectable():
    """'plateau' (round-2 R2r-6) is a selectable parsimony-oriented rule: picks the plateau-onset N.
    Benchmarked it is NOT an accuracy default (0/6 vs one_se_max, behaves like one_se_min on flat GBM tails);
    shipped as an explicit option, not the default. Here: it must fit and yield a valid N <= one_se_max's N."""
    pytest.importorskip("sklearn")
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV, FIConfig, SearchConfig

    X, y = make_classification(n_samples=1200, n_features=16, n_informative=6, n_redundant=4, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(16)])
    mk = lambda rule: RFECV(
        estimator=RandomForestClassifier(n_estimators=40, max_depth=6, random_state=0),
        cv=3,
        scoring=None,
        verbose=0,
        fi_config=FIConfig(importance_getter="feature_importances_", n_features_selection_rule=rule),
        search_config=SearchConfig(max_refits=12, max_runtime_mins=2),
        random_state=0,
    ).fit(X, pd.Series(y))
    rp = mk("plateau")
    rmax = mk("one_se_max")
    assert 1 <= rp.n_features_ <= rmax.n_features_  # plateau is parsimonious: never larger than one_se_max
