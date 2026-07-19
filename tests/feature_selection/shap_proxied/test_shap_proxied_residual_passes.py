"""Unit tests for gt_09 two-phase residual attribution: param validation, clone round-trip,
``within_cluster_refine(protected_cols=...)``, the classification pseudo-residual formula, and
``report["residual_pass"]`` presence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

pytest.importorskip("shap")
pytest.importorskip("xgboost")

from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
from mlframe.feature_selection.shap_proxied_fs._shap_proxied_fit_residual import compute_residual_target

# --------------------------------------------------------------------------- param validation


def _tiny_fixture(seed=0, n=200, p=15):
    """Small classification fixture for fast param-validation / report-shape tests."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
    logit = 2.0 * X["f0"] + 2.0 * X["f1"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, pd.Series(y)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(residual_passes=-1),
        dict(residual_passes=3),
        dict(residual_merge="bogus"),
        dict(residual_lambda=-0.1),
        dict(residual_top_k=0),
        dict(residual_top_k=-5),
        dict(residual_exclude_top=-1),
    ],
)
def test_residual_param_validation_rejects_invalid(kwargs):
    """Invalid residual_* params raise ValueError at fit time (not at __init__, per sklearn clone contract)."""
    X, y = _tiny_fixture()
    sel = ShapProxiedFS(random_state=0, verbose=False, n_jobs=1, **kwargs)
    with pytest.raises(ValueError):
        sel.fit(X, y)


def test_residual_passes_two_is_accepted():
    """residual_passes=2 is the documented ceiling and must NOT raise at validation."""
    X, y = _tiny_fixture()
    sel = ShapProxiedFS(
        random_state=0,
        verbose=False,
        n_jobs=1,
        trust_guard=False,
        run_importance_ablation=False,
        residual_passes=2,
        residual_merge="rescue",
    )
    sel.fit(X, y)  # must not raise
    assert sel.shap_proxy_report_["residual_pass"]["n_passes"] == 2


# --------------------------------------------------------------------------- clone round-trip


def test_residual_params_survive_sklearn_clone():
    """sklearn clone() must preserve every residual_* param verbatim."""
    sel = ShapProxiedFS(
        residual_passes=1,
        residual_merge="blend",
        residual_lambda=0.5,
        residual_top_k=10,
        residual_exclude_top=3,
    )
    cloned = clone(sel)
    assert cloned.residual_passes == 1
    assert cloned.residual_merge == "blend"
    assert cloned.residual_lambda == 0.5
    assert cloned.residual_top_k == 10
    assert cloned.residual_exclude_top == 3


# --------------------------------------------------------------------------- protected_cols


def test_within_cluster_refine_protected_cols_never_dropped():
    """A protected member survives greedy pruning even at parsimony_tol=1.0 (accepts any loss increase)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import within_cluster_refine

    rng = np.random.default_rng(0)
    n, f = 1200, 8
    X = pd.DataFrame(rng.normal(size=(n, f)), columns=[f"x{i}" for i in range(f)])
    y = (1.5 * X["x0"] + 1.0 * X["x1"] - 0.8 * X["x2"] + 0.05 * rng.normal(size=n)).to_numpy()
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]
    member_cols = [0, 1, 2, 3, 4, 5, 6, 7]

    refined_unprotected = within_cluster_refine(
        member_cols,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        parsimony_tol=1.0,
        n_jobs=1,
        refine_n_estimators=None,
    )
    # At parsimony_tol=1.0 the greedy pruner accepts near-any loss increase; unprotected refine
    # should compact well below the full 8-column set.
    assert len(refined_unprotected) < len(member_cols)

    refined_protected = within_cluster_refine(
        member_cols,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        parsimony_tol=1.0,
        n_jobs=1,
        refine_n_estimators=None,
        protected_cols={2},
    )
    assert 2 in refined_protected, "protected member 2 was dropped despite protected_cols={2}"


def test_within_cluster_refine_protected_cols_none_is_byte_identical():
    """protected_cols=None (default) must not change behaviour -- a byte-identity spot check."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import within_cluster_refine

    rng = np.random.default_rng(1)
    n, f = 900, 6
    X = pd.DataFrame(rng.normal(size=(n, f)), columns=[f"x{i}" for i in range(f)])
    y = (1.2 * X["x0"] - 0.9 * X["x1"] + 0.05 * rng.normal(size=n)).to_numpy()
    Xs, ys = X.iloc[:700].reset_index(drop=True), y[:700]
    Xh, yh = X.iloc[700:].reset_index(drop=True), y[700:]
    member_cols = [0, 1, 2, 3, 4, 5]

    r1 = within_cluster_refine(member_cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse", parsimony_tol=0.02, n_jobs=1)
    r2 = within_cluster_refine(
        member_cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse", parsimony_tol=0.02, n_jobs=1, protected_cols=None
    )
    assert r1 == r2


# --------------------------------------------------------------------------- residual target formula


def test_classification_residual_is_y_minus_sigmoid_margin():
    """Classification residual target must be exactly y - sigmoid(base + sum(phi))."""
    base = np.array([0.1, -0.2, 0.0])
    phi = np.array([[0.5, -0.1], [0.2, 0.3], [-0.4, 0.4]])
    y = np.array([1.0, 0.0, 1.0])
    residual = compute_residual_target(phi, base, y, classification=True)
    margin = base + phi.sum(axis=1)
    expected = y - 1.0 / (1.0 + np.exp(-margin))
    np.testing.assert_allclose(residual, expected, rtol=1e-10)


def test_regression_residual_is_plain_proxy_residual():
    """Regression residual target must be exactly y - (base + sum(phi))."""
    base = np.array([1.0, 2.0])
    phi = np.array([[0.5, 0.5], [-0.5, 1.5]])
    y = np.array([3.0, 2.0])
    residual = compute_residual_target(phi, base, y, classification=False)
    np.testing.assert_allclose(residual, y - (base + phi.sum(axis=1)), rtol=1e-10)


# --------------------------------------------------------------------------- report presence


def test_residual_pass_report_keys_present_when_enabled():
    """report['residual_pass'] must expose all documented telemetry keys when residual_passes>0."""
    X, y = _tiny_fixture()
    sel = ShapProxiedFS(
        random_state=0,
        verbose=False,
        n_jobs=1,
        trust_guard=False,
        run_importance_ablation=False,
        residual_passes=1,
        residual_merge="rescue",
    )
    sel.fit(X, y)
    rp = sel.shap_proxy_report_["residual_pass"]
    for key in ("n_passes", "merge", "lambda_", "rescued", "excluded_top", "pass2_top_importance", "residual_std_before", "residual_std_after"):
        assert key in rp, f"report['residual_pass'] missing key {key!r}"


def test_residual_pass_absent_report_key_at_default_zero_passes():
    """residual_passes=0 (default) must not add residual_pass to the report -- opt-in, no-op contract."""
    X, y = _tiny_fixture()
    sel = ShapProxiedFS(random_state=0, verbose=False, n_jobs=1, trust_guard=False, run_importance_ablation=False)
    sel.fit(X, y)
    assert "residual_pass" not in sel.shap_proxy_report_
