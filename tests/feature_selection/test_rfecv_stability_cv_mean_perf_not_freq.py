"""Regression: the stability-selection path must not mislabel in-sample selection frequency as an
honest held-out CV score.

Pre-fix, ``_fit_stability_selection`` wrote the bootstrap selection frequency into
``cv_results_['cv_mean_perf']`` -- the key the rest of the codebase reads as a genuine OOF/held-out
score (select_optimal_nfeatures_, diagnostics). The fix surfaces the frequency under its own
``selection_frequency`` key and leaves ``cv_mean_perf`` NaN (no held-out score is computed here).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from mlframe.feature_selection.wrappers.rfecv import RFECV


def _fit_stability_rfecv():
    rng = np.random.default_rng(0)
    n = 60
    informative = rng.normal(size=(n, 2))
    noise = rng.normal(size=(n, 4))
    X = pd.DataFrame(
        np.column_stack([informative, noise]),
        columns=[f"f{i}" for i in range(6)],
    )
    y = (informative[:, 0] + informative[:, 1] > 0).astype(int)

    sel = RFECV(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=0),
        stability_selection=True,
        stability_n_bootstrap=10,
        stability_threshold=0.5,
        random_state=0,
        verbose=0,
    )
    sel.fit(X, y)
    return sel


def test_cv_mean_perf_is_not_silently_the_selection_frequency():
    sel = _fit_stability_rfecv()
    cv = sel.cv_results_

    assert "selection_frequency" in cv, "selection frequency must be surfaced under its own key"
    freq = cv["selection_frequency"][0]
    assert 0.0 <= freq <= 1.0

    # cv_mean_perf must NOT carry the selection frequency; the stability path computes no held-out score.
    mean_perf = cv["cv_mean_perf"][0]
    assert math.isnan(mean_perf), "cv_mean_perf must be NaN (no honest held-out score), not the selection frequency"
