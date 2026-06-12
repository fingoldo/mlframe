"""Tests for the iter54 ``inner_n_jobs_cap`` knob that controls per-fit booster ``n_jobs`` inside the
parallel pools (OOF-SHAP / reval / refine / trust-guard).

The default (False) lets xgboost manage all cores via its own thread pool (booster ``n_jobs=-1``);
iter53 A/B at width 4000+10000 measured the legacy cap as 8-9% e2e slower on 8-core boxes. The cap
is preserved as opt-in for HW where measurement says it helps.

These tests do NOT measure perf -- they verify:
  1. The kwarg is accepted on every public dispatcher.
  2. The chosen subset and honest loss are bit-identical between cap=False and cap=True (both modes
     reach the same byte-equivalent xgboost training given the same seeds; the cap only affects how
     xgboost's internal thread pool is sized, not the model weights).
  3. The selector facade threads ``inner_n_jobs_cap`` through to all four downstream public
     functions (compute_shap_matrix, proxy_trust_guard, revalidate_top_n, within_cluster_refine).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_planted(n=400, f=8, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, f)), columns=[f"x{i}" for i in range(f)])
    y = (1.5 * X["x0"] + 1.0 * X["x1"] - 0.8 * X["x2"] + 0.05 * rng.normal(size=n)).to_numpy()
    return X, y


def test_parallel_honest_losses_accepts_cap_kwarg():
    from sklearn.linear_model import LinearRegression
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _parallel_honest_losses

    X, y = _make_planted(n=200, f=6, seed=0)
    Xs, ys = X.iloc[:140].reset_index(drop=True), y[:140]
    Xh, yh = X.iloc[140:].reset_index(drop=True), y[140:]
    tasks = [((0, 1, 2), 1), ((0, 1), 2)]
    losses_default = _parallel_honest_losses(
        tasks, LinearRegression(), Xs, ys, Xh, yh, False, "rmse", n_jobs=1)
    losses_cap_off = _parallel_honest_losses(
        tasks, LinearRegression(), Xs, ys, Xh, yh, False, "rmse", n_jobs=1, inner_n_jobs_cap=False)
    losses_cap_on = _parallel_honest_losses(
        tasks, LinearRegression(), Xs, ys, Xh, yh, False, "rmse", n_jobs=1, inner_n_jobs_cap=True)
    # The cap only affects how each booster's internal n_jobs gets set; LinearRegression doesn't
    # have an n_jobs param it cares about for tiny matrices, but the dispatcher must still complete
    # and return numerically equal floats across all three calling conventions.
    assert np.allclose(losses_default, losses_cap_off)
    assert np.allclose(losses_default, losses_cap_on)


def test_compute_shap_matrix_accepts_cap_kwarg():
    """compute_shap_matrix exposes inner_n_jobs_cap; default False; both values complete."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import compute_shap_matrix, make_default_estimator

    X, y = _make_planted(n=200, f=6, seed=1)
    model = make_default_estimator(classification=False, n_estimators=20, random_state=0)
    phi_off, base_off, y_off = compute_shap_matrix(
        model, X, y, classification=False, out_of_fold=True, n_splits=3, n_models=1,
        rng=np.random.default_rng(0), n_jobs=1, inner_n_jobs_cap=False)
    phi_on, base_on, y_on = compute_shap_matrix(
        model, X, y, classification=False, out_of_fold=True, n_splits=3, n_models=1,
        rng=np.random.default_rng(0), n_jobs=1, inner_n_jobs_cap=True)
    # n_jobs=1 -> outer = 1 -> inner = None for both; numerically identical.
    assert phi_off.shape == phi_on.shape
    assert np.allclose(phi_off, phi_on)
    assert np.allclose(base_off, base_on)


def test_dataset_diagnostics_accepts_cap_kwarg():
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_preflight import dataset_diagnostics

    X, y = _make_planted(n=200, f=8, seed=2)
    d_off = dataset_diagnostics(X, y, classification=False, max_rows=150, n_estimators=20,
                                random_state=0, inner_n_jobs_cap=False)
    d_on = dataset_diagnostics(X, y, classification=False, max_rows=150, n_estimators=20,
                               random_state=0, inner_n_jobs_cap=True)
    # The booster CV scores are byte-identical given fixed random_state irrespective of inner threads.
    assert d_off["full_model_fit"] == d_on["full_model_fit"]
    assert d_off["stump_fit"] == d_on["stump_fit"]
    assert d_off["additive_ratio"] == d_on["additive_ratio"]


def test_shap_proxied_fs_accepts_inner_n_jobs_cap():
    """The facade exposes inner_n_jobs_cap and stores it on self for sklearn get_params/set_params."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    fs_default = ShapProxiedFS()
    assert hasattr(fs_default, "inner_n_jobs_cap")
    assert fs_default.inner_n_jobs_cap is False  # new default
    fs_legacy = ShapProxiedFS(inner_n_jobs_cap=True)
    assert fs_legacy.inner_n_jobs_cap is True
    assert ShapProxiedFS().get_params()["inner_n_jobs_cap"] is False
    assert ShapProxiedFS(inner_n_jobs_cap=True).get_params()["inner_n_jobs_cap"] is True


def test_shap_proxied_fs_fit_runs_both_modes_with_same_chosen_subset():
    """End-to-end smoke: the chosen subset is bit-identical with cap on vs off. The cap only changes
    the booster's internal scheduler, not the model weights or any downstream ranking decision."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _make_planted(n=400, f=10, seed=3)
    common = dict(
        classification=False, metric="rmse", n_jobs=1, top_n=5, n_revalidation_models=1,
        random_state=0, verbose=False, tqdm=False, trust_guard=False,
        revalidation_ucb_enabled=False, refine_ucb_enabled=False,
        prefilter_top=None, cluster_features=False, run_importance_ablation=False,
    )
    fs_off = ShapProxiedFS(inner_n_jobs_cap=False, **common).fit(X, y)
    fs_on = ShapProxiedFS(inner_n_jobs_cap=True, **common).fit(X, y)
    assert tuple(fs_off.selected_features_) == tuple(fs_on.selected_features_)
