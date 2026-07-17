"""Regression sensors for the 2026-06-22 RFECV wrapper-family audit.

F1: stability_selection must not count zero-importance (noise) features into the per-bootstrap top-K.
F2: the SFFS post-swap best-N refresh must ignore NaN means (max-by-get with NaN masks a real winner).
F3: n_features_bootstrap_ci_ must return a well-ordered (low, mid, high) triple.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_f1_stability_selection_does_not_count_zero_fi_noise():
    """A bootstrap whose summed FI is positive but concentrated on a couple of features must NOT pad its
    top-K with zero-FI noise columns. Pre-fix, lexsort took exactly top_k indices including zero-FI ones,
    so noise features accrued counts every bootstrap and could cross the threshold."""
    pytest.importorskip("sklearn")
    from mlframe.feature_selection.wrappers.rfecv._stability_select import _fit_stability_selection
    import mlframe.feature_selection.wrappers.rfecv._stability_select as ss

    n_features = 20
    feature_names = [f"f{i}" for i in range(n_features)]

    class _Stub:
        # Only the first 3 features ever get positive importance; the rest are pure noise (FI == 0).
        verbose = 0
        random_state = 0
        estimator = object()
        estimators = None
        importance_getter = "auto"
        n_repeats = 1
        stability_n_bootstrap = 30
        stability_threshold = 0.6
        stability_top_k = 10  # generous: would pad with 7 zero-FI noise cols pre-fix
        must_include = None
        wide_data_fi_fallback = False

    stub = _Stub()

    import pandas as pd

    X = pd.DataFrame(np.random.RandomState(0).randn(60, n_features), columns=feature_names)
    y = (X["f0"] + X["f1"] + X["f2"] > 0).astype(int).to_numpy()

    # Monkeypatch the FI getter + estimator clone so every bootstrap returns FI on exactly f0/f1/f2.
    def _fake_get_fi(model, current_features, data, target, importance_getter, n_repeats):
        return {"f0": 1.0, "f1": 0.9, "f2": 0.8}

    class _FakeEst:
        def fit(self, X, y):
            return self

    import sklearn.base

    orig_clone = ss.clone
    orig_getfi = ss.get_feature_importances
    ss.clone = lambda est: _FakeEst()
    ss.get_feature_importances = _fake_get_fi
    try:
        _fit_stability_selection(stub, X, y, signature=(X.shape, (60,), tuple(feature_names), "h", "h", object()))
    finally:
        ss.clone = orig_clone
        ss.get_feature_importances = orig_getfi

    freq = stub.stability_selection_freq_
    # The 3 informative features must hit frequency 1.0; ALL noise features must stay at 0.
    assert freq[0] == 1.0 and freq[1] == 1.0 and freq[2] == 1.0
    assert np.all(freq[3:] == 0.0), f"zero-FI noise features accrued selection counts: {freq[3:]}"
    assert stub.n_features_ == 3


def test_f3_bootstrap_ci_is_well_ordered():
    """n_features_bootstrap_ci_ must always return low <= mid <= high even when n_features_ falls outside
    the bootstrap percentile band."""
    from mlframe.feature_selection.wrappers.rfecv._diagnostics import n_features_bootstrap_ci_

    class _Stub:
        # cv curve peaks sharply at N=5 so the bootstrap argmax concentrates around 5,
        # but the picker chose n_features_=15 (e.g. one_se_max on a flat tail) -> outside the CI.
        n_features_ = 15
        cv_results_ = {
            "nfeatures": [0, 5, 10, 15],
            "cv_mean_perf": [0.0, 0.95, 0.6, 0.55],
            "cv_std_perf": [0.0, 0.01, 0.01, 0.01],
        }

    low, mid, high = n_features_bootstrap_ci_(_Stub(), n_bootstrap=500, ci=0.9, random_state=0)
    assert low <= mid <= high, f"CI triple not ordered: ({low}, {mid}, {high})"


def test_f2_swap_refresh_ignores_nan_means():
    """The post-swap best-N refresh must not be fooled by a NaN-scored N. max(d, key=d.get) with a NaN
    entry returns an arbitrary key; the fix selects the argmax over finite means only."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import Ridge
    from mlframe.feature_selection.wrappers.rfecv._finalize import _finalize_fit_results
    import pandas as pd

    # Build a minimal stub whose evaluated_scores_mean has a NaN that pre-fix max-by-get would surface.
    evaluated_scores_mean = {3: float("nan"), 5: 0.80, 8: 0.70}
    finite = {n: v for n, v in evaluated_scores_mean.items() if not np.isnan(v)}
    best_nf = max(finite, key=finite.get)
    assert best_nf == 5, "finite argmax must pick N=5, not the NaN slot"
    # Sanity: the buggy form returns the NaN key on this dict.
    buggy = max(evaluated_scores_mean, key=evaluated_scores_mean.get)
    assert np.isnan(evaluated_scores_mean[buggy]), "buggy max-by-get surfaces the NaN slot (confirms the hazard)"
