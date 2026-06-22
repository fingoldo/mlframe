"""Regression sensors for the 2026-06-22 ShapProxiedFS audit round-2 fixes.

Covers L3 (multiclass honest loss), CA1 (monotone corrector), IX2/IX3 (interaction NaN-guard /
per-row base), T2 (xgb split-feature parse), P1f (empty-sample SU guard), SR1 (NaN argpartition),
RF1 (full-set random baseline). Each fails on the pre-fix logic and passes post-fix. CPU-only; GPU
paths (T1, SR1 GPU mirror, T4) are fixed statically and cannot be executed on this box.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------------------------- L3
def test_multiclass_logloss_from_proba_matrix():
    """A >2-class holdout must route to multiclass log-loss from the full (n, C) probability matrix
    instead of the binary closed form. Pre-fix ``_loss_from_predictions`` only handled a 1-D positive-
    class vector + ``log_loss(labels=[0,1])`` / ``brier_score_loss``, which is silently-wrong or raises
    on 3 classes."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_loss import (
        _loss_from_predictions,
    )

    y = np.array([0, 1, 2, 1, 0, 2])
    # Confident-correct probabilities (rows sum to 1) -> low multiclass log-loss.
    proba = np.eye(3)[y] * 0.8 + 0.1
    proba = proba / proba.sum(axis=1, keepdims=True)
    loss = _loss_from_predictions(proba, y, True, "logloss")
    assert np.isfinite(loss) and loss > 0
    # A uniform (uninformative) matrix must score strictly worse (higher loss) than the confident one.
    uniform = np.full((6, 3), 1.0 / 3.0)
    loss_uniform = _loss_from_predictions(uniform, y, True, "logloss")
    assert loss_uniform > loss, "uninformative multiclass proba must yield a higher log-loss"


def test_multiclass_brier_metric_routes_to_logloss_not_binary():
    """``metric='brier'`` on a multiclass matrix must NOT call binary ``brier_score_loss`` (raises on a
    matrix); it folds into multiclass log-loss. Pre-fix this path crashed."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_loss import (
        _loss_from_predictions,
    )

    y = np.array([0, 1, 2, 2, 1, 0])
    proba = np.eye(3)[y] * 0.7 + 0.1
    proba = proba / proba.sum(axis=1, keepdims=True)
    loss = _loss_from_predictions(proba, y, True, "brier")
    assert np.isfinite(loss) and loss > 0


def test_multiclass_ovr_auc_from_proba_matrix():
    """``metric='auc'`` on a multiclass matrix must route to one-vs-rest macro AUC (1 - auc), not the
    binary scalar ``roc_auc_score(y, p)`` which raises on a matrix / multiclass y."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_loss import (
        _loss_from_predictions,
    )

    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    proba = np.eye(3)[y] * 0.8 + 0.1  # perfectly separable -> auc ~ 1 -> loss ~ 0
    proba = proba / proba.sum(axis=1, keepdims=True)  # OvR multiclass AUC needs rows summing to 1
    loss = _loss_from_predictions(proba, y, True, "auc")
    assert np.isfinite(loss) and loss < 0.1


def test_classification_proba_returns_full_matrix_for_multiclass():
    """The proba extractor must return the full (n, C) matrix for a >2-class fit (so the loss layer can
    score multiclass) and the positive-class column for a binary fit. Pre-fix it hard-indexed ``[:, 1]``
    (binary-only)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_loss import (
        _classification_proba,
    )

    class _MultiEst:
        def predict_proba(self, X):
            n = len(X)
            return np.column_stack([np.full(n, 0.2), np.full(n, 0.3), np.full(n, 0.5)])

    out = _classification_proba(_MultiEst(), np.zeros((4, 2)))
    assert out.shape == (4, 3), "multiclass fit must yield the full probability matrix"


# --------------------------------------------------------------------------------------------- CA1
def test_corrector_never_inverts_proxy_order():
    """The bias corrector must never invert the raw proxy ordering (the docstring's 'never worse'
    guarantee). With anchors where honest loss DECREASES as proxy INCREASES (an inverting relation), the
    fit would learn a negative proxy response; the order-guard must reject it and fall back to identity,
    so the corrector's predicted ranking stays positively correlated with the proxy."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_calibrate import fit_proxy_corrector

    rng = np.random.default_rng(0)
    proxy = np.linspace(0.1, 0.9, 40)
    honest = 1.0 - proxy + rng.normal(0, 0.01, 40)  # strictly inverting relation
    cards = rng.integers(2, 8, 40).astype(float)
    redunds = rng.random(40)
    corr = fit_proxy_corrector(proxy, honest, cards, redunds, min_anchors=12)
    pred = corr.predict(proxy, cards, redunds)
    pr = np.argsort(np.argsort(proxy))
    pe = np.argsort(np.argsort(pred))
    rho = float(np.corrcoef(pr, pe)[0, 1])
    assert rho >= 0.0, "corrector must not invert the proxy ordering (CA1 guarantee)"


def test_corrector_kept_when_order_preserving():
    """When honest loss is positively related to proxy, the fitted corrector is kept (not forced to
    fallback) -- the order-guard must not reject a legitimate calibration."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_calibrate import fit_proxy_corrector

    rng = np.random.default_rng(1)
    proxy = np.linspace(0.1, 0.9, 60)
    honest = 0.5 * proxy + 0.2 + rng.normal(0, 0.01, 60)  # monotone increasing
    cards = rng.integers(2, 8, 60).astype(float)
    redunds = rng.random(60) * 0.1
    corr = fit_proxy_corrector(proxy, honest, cards, redunds, min_anchors=12)
    assert corr.fallback is False, "an order-preserving calibration must be retained"


# ---------------------------------------------------------------------------------------- IX2 / IX3
def test_interaction_product_columns_are_nan_inf_safe():
    """Engineered product columns must be sanitised: a NaN or inf operand cannot poison the in-sample
    SHAP fit. Pre-fix ``aug[nm] = X[a] * X[b]`` left NaN/inf in the frame."""
    import pandas as pd

    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions as itx

    n = 20
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.random(n), "b": rng.random(n)})
    X.loc[0, "a"] = np.nan
    X.loc[1, "b"] = np.inf
    y = (rng.random(n) < 0.5).astype(int)

    captured = {}

    def _fake_compute_shap_matrix(model_template, aug, y_arr, **kw):
        captured["aug"] = aug
        p = aug.shape[1]
        phi = np.zeros((len(aug), p))
        base = np.zeros(len(aug))
        return phi, base, y_arr

    def _fake_brute(*a, **k):
        return [(0.5, (0,))]

    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain as expl
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_search as srch
    orig_csm, orig_bf = expl.compute_shap_matrix, srch.brute_force_top_n
    expl.compute_shap_matrix = _fake_compute_shap_matrix
    srch.brute_force_top_n = _fake_brute
    try:
        kept = [(0.9, 0.9, "a", "b")]
        itx.sparse_interaction_candidates(object(), X, y, kept, classification=True, metric="logloss")
    finally:
        expl.compute_shap_matrix = orig_csm
        srch.brute_force_top_n = orig_bf

    aug = captured["aug"]
    prod_cols = [c for c in aug.columns if c.startswith("_suprod_")]
    assert prod_cols, "a product column must have been engineered"
    assert np.isfinite(aug[prod_cols].to_numpy()).all(), "product columns must be NaN/inf-free"


def test_broadcast_base_scalar_and_per_row():
    """``_broadcast_base`` must broadcast a scalar to (n,) AND accept an already per-row (n,) base without
    collapsing it to base[0]. Pre-fix ``np.full(n, float(base))`` raised on a per-row base."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions import _broadcast_base

    assert np.allclose(_broadcast_base(0.3, 5), 0.3)
    assert _broadcast_base(0.3, 5).shape == (5,)
    per_row = np.array([0.1, 0.2, 0.3, 0.4])
    out = _broadcast_base(per_row, 4)
    assert np.allclose(out, per_row), "a per-row base must be preserved, not collapsed to base[0]"
    with pytest.raises(ValueError):
        _broadcast_base(np.array([0.1, 0.2]), 5)


# ----------------------------------------------------------------------------------------------- T2
def test_resolve_split_feature_does_not_lstrip_all_f():
    """``_resolve_split_feature`` must strip only the single positional ``f`` prefix (``^f\\d+$``), not all
    leading 'f' chars, and must raise on a feature name missing from a provided fmap. Pre-fix
    ``int(str(sp).lstrip('f'))`` would silently mis-parse and the missing-key case fell back to a
    positional parse."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import _resolve_split_feature

    assert _resolve_split_feature("f12", None) == 12
    # fmap present + key present -> use the map.
    assert _resolve_split_feature("price", {"price": 3}) == 3
    # fmap present + key MISSING -> raise (no silent positional fallback).
    with pytest.raises(KeyError):
        _resolve_split_feature("missing_feat", {"price": 3})
    # A non-default name with no fmap cannot be positionally parsed -> raise (not lstrip to garbage).
    with pytest.raises(ValueError):
        _resolve_split_feature("ffx", None)


# ---------------------------------------------------------------------------------------------- P1f
def test_pairwise_su_edges_empty_samples_no_zero_division():
    """The njit pairwise-SU kernel must return early (no edges) on a zero-sample matrix instead of
    dividing by ``n_samples`` (P1f). Exercises the guard directly."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su import _pairwise_su_edges

    bins_packed = np.zeros((3, 0), dtype=np.int32)
    nbins = np.array([1, 1, 1], dtype=np.int64)
    freqs_packed = np.zeros(3, dtype=np.float64)
    freqs_offsets = np.array([0, 1, 2, 3], dtype=np.int64)
    h_marginals = np.zeros(3, dtype=np.float64)
    constant_mask = np.zeros(3, dtype=np.bool_)
    flags = _pairwise_su_edges(bins_packed, nbins, freqs_packed, freqs_offsets,
                               h_marginals, constant_mask, 0.5)
    assert flags.shape == (3, 3)
    assert not flags.any(), "no edges on an empty (zero-sample) matrix"


# ---------------------------------------------------------------------------------------------- SR1
def test_subsetrank_nan_loss_never_selected_as_top():
    """A NaN proxy loss (degenerate single-class slice) must never be argpartition-selected as 'top'
    (lowest); non-finite losses are mapped to +inf so they sink. Pre-fix argpartition ordering with NaN
    was undefined and could surface a NaN subset as the winner."""
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_subsetrank as sr

    n, f = 40, 4
    rng = np.random.default_rng(0)
    phi = rng.normal(0, 1, (n, f))
    base = np.zeros(n)
    y = (rng.random(n) < 0.5).astype(np.float64)

    real_scan = sr._subset_loss_scan_njit

    def _nan_poisoning_scan(phi_, base_, y_, combos, code, out):
        real_scan(phi_, base_, y_, combos, code, out)
        out[0] = np.nan  # poison the first combo's loss

    orig = sr._subset_loss_scan_njit
    sr._subset_loss_scan_njit = _nan_poisoning_scan
    try:
        res = sr.brute_force_top_n(phi, base, y, classification=True, metric="logloss",
                                   min_card=1, max_card=2, top_n=5, parallel=False)
    finally:
        sr._subset_loss_scan_njit = orig

    for loss, _comb in res:
        assert np.isfinite(loss), "a NaN loss must never be selected into the top-N"


# ---------------------------------------------------------------------------------------------- RF1
def test_refine_random_baseline_skipped_when_winner_is_full_set():
    """The winner's-curse random baseline is meaningful only when the winner is strictly smaller than the
    full feature set (k < f): at k >= f a same-size random sample would BE the full set, a tautological
    baseline. Pre-fix ``k = min(k, f)`` always produced a baseline even at k == f."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_refine import (
        _random_baseline_is_meaningful,
    )

    assert _random_baseline_is_meaningful(3, 10) is True
    assert _random_baseline_is_meaningful(10, 10) is False, "k == f must skip the random baseline"
    assert _random_baseline_is_meaningful(0, 10) is False
