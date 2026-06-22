"""Regression sensors for the 2026-06-22 ShapProxiedFS audit fixes.

Each test fails on the pre-fix code (verified via git stash) and passes post-fix.
CPU-only; no GPU required.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_single_class_holdout_auc_loss_is_nan_not_sentinel():
    """A single-class holdout has no defined AUC; the loss layer must return NaN (dropped by every
    downstream finite-mask) instead of a magic 1.0 that masquerades as a measured loss and biases
    the corrector / stable-score ranking on rare-class anchors."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_loss import (
        _loss_from_predictions,
    )

    loss = _loss_from_predictions(np.array([0.4, 0.6, 0.55]), np.array([1, 1, 1]), True, "auc")
    assert np.isnan(loss), "single-class AUC holdout must yield NaN, not a 1.0 sentinel"


def test_positive_class_proba_handles_single_class_fit():
    """A booster trained on a single-class anchor returns a 1-column predict_proba; the safe extractor
    must not raise IndexError (the pre-fix hard ``[:, 1]`` did) on rare-class anchors."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_loss import (
        _positive_class_proba,
    )

    class _OneColEst:
        def predict_proba(self, X):
            return np.ones((len(X), 1))

    out = _positive_class_proba(_OneColEst(), np.zeros((5, 3)))
    assert out.shape == (5,)
    assert np.allclose(out, 1.0)

    class _TwoColEst:
        def predict_proba(self, X):
            n = len(X)
            return np.column_stack([np.full(n, 0.3), np.full(n, 0.7)])

    out2 = _positive_class_proba(_TwoColEst(), np.zeros((4, 2)))
    assert np.allclose(out2, 0.7), "binary path must still take the positive (column-1) probability"


def test_serial_su_treats_one_realized_bin_as_constant():
    """Serial SU pairwise edges must skip a column with a single POPULATED bin (constant) even when its
    frequency vector is padded to length>1 via nbins_per_feature -- matching the kernel path (nonzero_bins<=1).
    Pre-fix the serial path keyed off ``freqs.size`` and treated such a column as non-constant, then ran
    ``compute_su_from_classes`` on it; with a partner sharing that bin it can spuriously link (SU>0 from the
    degenerate joint), diverging from the kernel clustering for the same data."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su import (
        cluster_correlated_features_su,
    )

    n = 200
    rng = np.random.default_rng(0)
    # Two genuinely-correlated discrete features (high SU) and one CONSTANT (one realized bin) padded to nbins=4.
    a = (rng.random(n) < 0.5).astype(np.int64)
    b = a.copy()
    b[: n // 20] = 1 - b[: n // 20]  # strong linkage with a
    const = np.zeros(n, dtype=np.int64)  # realizes only bin 0
    bins = {"a": a, "b": b, "const": const}
    nbins_hint = {"a": 2, "b": 2, "const": 4}  # force const's marginal to be padded to length 4

    labels = cluster_correlated_features_su(
        bins, threshold=0.3, feature_names=["a", "b", "const"], nbins_per_feature=nbins_hint,
        use_parallel=False, use_gpu=False,
    )
    labels = np.asarray(labels)
    assert labels.shape[0] == 3
    # const carries SU=0 with every partner -> own singleton cluster, never merged with a or b.
    assert labels[2] != labels[0]
    assert labels[2] != labels[1]


def test_interaction_shap_binary_base_uses_positive_class():
    """When shap returns a list-of-2 (binary) interaction tensor, the positive-class base must be selected.
    Pre-fix the guard read ``len(Phi)`` AFTER reassigning Phi to an array, so it tested the ROW count (n)
    not the class count and silently picked the negative-class base for n!=2. We force the shap fallback
    path and stub shap.TreeExplainer so the assertion is on the list-handling branch, not on shap itself."""
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions as itx

    n, p = 7, 3
    X = np.zeros((n, p))
    y = np.array([0, 1, 0, 1, 0, 1, 0])

    class _FakeExplainer:
        def __init__(self, model, **kw):
            pass

        def shap_interaction_values(self, X):
            neg = np.zeros((len(X), p, p))
            pos = np.full((len(X), p, p), 0.5)  # positive class carries the signal
            return [neg, pos]

        expected_value = np.array([0.1, 0.9])  # [neg_base, pos_base]

    import shap as _shap_mod

    orig_te = _shap_mod.TreeExplainer
    orig_fit = itx._fit_one
    orig_unwrap = itx._unwrap_estimator
    _shap_mod.TreeExplainer = _FakeExplainer
    itx._fit_one = lambda *a, **k: object()
    itx._unwrap_estimator = lambda est: est
    try:
        Phi, base = itx.compute_interaction_tensor(object(), X, y, classification=True, backend="shap")
    finally:
        _shap_mod.TreeExplainer = orig_te
        itx._fit_one = orig_fit
        itx._unwrap_estimator = orig_unwrap

    assert np.allclose(base, 0.9), "binary shap base must select the positive-class expected_value"
    assert np.allclose(Phi, 0.5), "binary shap interaction tensor must select the positive class"
