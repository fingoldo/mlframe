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


@pytest.mark.parametrize("code", [0, 1, 2, 3])
@pytest.mark.parametrize("n", [200, 12000])
def test_score_margin_parallel_matches_serial(code, n):
    """score_margin_parallel (prange) must match the serial kernel to summation round-off.

    Locks the iter106 dispatcher: the parallel twin is routed in for tall margins, so its loss must
    track the serial loop closely enough that subset ranking is unaffected. The reduction order
    differs under prange so we assert rtol=1e-12 (float64 sum-reorder), not bit-equality. Both n
    below and above the default crossover (10000) are exercised so the kernel itself is checked
    regardless of which side of the dispatch threshold the row count falls on.
    """
    from mlframe.feature_selection._shap_proxy_objective import score_margin, score_margin_parallel

    rng = np.random.default_rng(code * 7 + n)
    margin = rng.normal(size=n)
    y = (rng.normal(size=n) if code in (0, 1) else (rng.random(n) > 0.5).astype(float))
    serial = score_margin(margin, y, code)
    parallel = score_margin_parallel(margin, y, code)
    np.testing.assert_allclose(parallel, serial, rtol=1e-12, atol=1e-12)


def test_score_margin_auto_routes_by_row_count(monkeypatch):
    """score_margin_auto picks parallel at/above the crossover, serial below it -- same value either
    way. Verifies the dispatch boundary uses margin.shape[0] against the resolved threshold."""
    import mlframe.feature_selection._shap_proxy_objective as O

    calls = {"serial": 0, "parallel": 0}
    real_serial, real_parallel = O.score_margin, O.score_margin_parallel

    def spy_serial(m, y, c):
        calls["serial"] += 1
        return real_serial(m, y, c)

    def spy_parallel(m, y, c):
        calls["parallel"] += 1
        return real_parallel(m, y, c)

    monkeypatch.setattr(O, "score_margin", spy_serial)
    monkeypatch.setattr(O, "score_margin_parallel", spy_parallel)
    monkeypatch.setattr(O, "_score_margin_parallel_min_rows", lambda: 1000)

    rng = np.random.default_rng(0)
    y_small = (rng.random(500) > 0.5).astype(float)
    O.score_margin_auto(rng.normal(size=500), y_small, 2)
    assert calls == {"serial": 1, "parallel": 0}

    y_big = (rng.random(2000) > 0.5).astype(float)
    O.score_margin_auto(rng.normal(size=2000), y_big, 2)
    assert calls == {"serial": 1, "parallel": 1}


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
