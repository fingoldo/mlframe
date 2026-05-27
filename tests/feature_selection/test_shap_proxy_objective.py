"""Unit tests for the SHAP-proxy objective: coalition margin + proper loss reductions.

Locks the H3 correction: the loss must be a *proper* metric (Brier/log-loss/RMSE/MAE), not the
original kernel's MAE-of-0/1-labels-vs-log-odds-margin. We verify numeric agreement with sklearn /
numpy references so a future refactor of the njit reductions can't silently drift.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_coalition_margin_is_base_plus_selected_sum():
    from mlframe.feature_selection._shap_proxy_objective import coalition_margin

    rng = np.random.default_rng(0)
    phi = rng.normal(size=(50, 6))
    base = np.full(50, 0.3)
    idx = [0, 2, 4]
    got = coalition_margin(phi, base, idx)
    expected = base + phi[:, idx].sum(axis=1)
    np.testing.assert_allclose(got, expected)
    # empty subset -> just the base value
    np.testing.assert_allclose(coalition_margin(phi, base, []), base)


def test_score_margin_matches_numpy_references():
    from mlframe.feature_selection._shap_proxy_objective import score_margin

    rng = np.random.default_rng(1)
    margin = rng.normal(size=200)
    y_reg = rng.normal(size=200)
    # MAE (code 0) and MSE (code 1) on raw margin
    np.testing.assert_allclose(score_margin(margin, y_reg, 0), np.mean(np.abs(y_reg - margin)), rtol=1e-10)
    np.testing.assert_allclose(score_margin(margin, y_reg, 1), np.mean((y_reg - margin) ** 2), rtol=1e-10)

    y_bin = (rng.random(200) > 0.5).astype(float)
    p = 1.0 / (1.0 + np.exp(-margin))
    np.testing.assert_allclose(score_margin(margin, y_bin, 2), np.mean((p - y_bin) ** 2), rtol=1e-9)  # Brier
    pe = np.clip(p, 1e-7, 1 - 1e-7)
    bce = -np.mean(y_bin * np.log(pe) + (1 - y_bin) * np.log(1 - pe))
    np.testing.assert_allclose(score_margin(margin, y_bin, 3), bce, rtol=1e-6)  # log-loss


def test_proxy_loss_auc_is_one_minus_auc():
    from mlframe.feature_selection._shap_proxy_objective import proxy_loss

    rng = np.random.default_rng(2)
    margin = rng.normal(size=300)
    y = (margin + 0.5 * rng.normal(size=300) > 0).astype(float)  # margin is predictive
    loss = proxy_loss(margin, y, "auc")
    assert 0.0 <= loss <= 0.5  # predictive -> AUC > 0.5 -> loss < 0.5
    # single-class slice -> worst loss, no crash
    assert proxy_loss(margin, np.ones_like(y), "auc") == 1.0


def test_resolve_metric_rejects_cross_task():
    from mlframe.feature_selection._shap_proxy_objective import resolve_metric

    assert resolve_metric(True, None) == "brier"
    assert resolve_metric(False, None) == "rmse"
    with pytest.raises(ValueError):
        resolve_metric(True, "rmse")  # regression metric on a classification task
    with pytest.raises(ValueError):
        resolve_metric(False, "auc")
