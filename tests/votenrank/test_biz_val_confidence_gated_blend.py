"""biz_value test for ``votenrank.confidence_gated_blend``.

The win: an auxiliary rule-based model that is very accurate on a narrow high-confidence slice of rows but
uninformative (noise) elsewhere should improve overall log-loss when blended in ONLY on that confident slice,
while a naive fixed-weight blend applied everywhere dilutes the main ensemble with noise on the rest of the
data and ends up worse than the gated version (and can even be worse than not blending at all).
"""
from __future__ import annotations

import numpy as np

from mlframe.votenrank.confidence_gated_blend import confidence_gated_blend


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _make_data(seed: int):
    rng = np.random.default_rng(seed)
    n = 4000
    z = rng.normal(0, 1, n)
    y = (z + rng.normal(0, 0.6, n) > 0).astype(int)

    ensemble_pred = 1.0 / (1.0 + np.exp(-(z + rng.normal(0, 0.6, n))))

    confident_mask = np.abs(z) > 1.3
    auxiliary_pred = np.where(confident_mask, y + rng.normal(0, 0.03, n), rng.uniform(0.3, 0.7, n))
    auxiliary_pred = np.clip(auxiliary_pred, 0.01, 0.99)
    auxiliary_confidence = np.where(confident_mask, 0.95, 0.3)

    return y, ensemble_pred, auxiliary_pred, auxiliary_confidence


def test_biz_val_confidence_gated_blend_beats_pure_ensemble_and_naive_fixed_blend():
    y, ensemble_pred, auxiliary_pred, auxiliary_confidence = _make_data(seed=0)

    loss_pure_ensemble = _log_loss(y, ensemble_pred)

    naive_blend = 0.92 * ensemble_pred + 0.08 * auxiliary_pred
    loss_naive_blend = _log_loss(y, naive_blend)

    gated_blend = confidence_gated_blend(
        ensemble_pred, auxiliary_pred, auxiliary_confidence, confidence_threshold=0.6, gated_weight=0.6, default_weight=0.0
    )
    loss_gated_blend = _log_loss(y, gated_blend)

    assert loss_gated_blend < loss_pure_ensemble, (
        f"gated blend should beat the pure ensemble: gated={loss_gated_blend:.4f} pure={loss_pure_ensemble:.4f}"
    )
    assert loss_gated_blend < loss_naive_blend, (
        f"gated blend should beat a naive fixed-weight-everywhere blend: gated={loss_gated_blend:.4f} naive={loss_naive_blend:.4f}"
    )


def test_confidence_gated_blend_all_low_confidence_returns_default_weight_blend():
    ensemble_pred = np.array([0.2, 0.5, 0.8])
    auxiliary_pred = np.array([0.9, 0.9, 0.9])
    confidence = np.array([0.1, 0.1, 0.1])
    result = confidence_gated_blend(ensemble_pred, auxiliary_pred, confidence, confidence_threshold=0.5, gated_weight=0.5, default_weight=0.0)
    assert np.allclose(result, ensemble_pred)


def test_confidence_gated_blend_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        confidence_gated_blend(np.array([0.1, 0.2]), np.array([0.1]), np.array([0.9, 0.9]), 0.5, 0.5)


def test_confidence_gated_blend_invalid_weight_raises():
    import pytest

    with pytest.raises(ValueError):
        confidence_gated_blend(np.array([0.1]), np.array([0.2]), np.array([0.9]), 0.5, gated_weight=1.5)


def test_confidence_gated_blend_backends_are_bit_identical():
    rng = np.random.default_rng(0)
    n = 5000
    ensemble = rng.uniform(0, 1, n)
    aux = rng.uniform(0, 1, n)
    conf = rng.uniform(0, 1, n)

    baseline = confidence_gated_blend(ensemble, aux, conf, 0.6, 0.5, force_backend="numpy")
    njit_result = confidence_gated_blend(ensemble, aux, conf, 0.6, 0.5, force_backend="njit")
    njit_parallel_result = confidence_gated_blend(ensemble, aux, conf, 0.6, 0.5, force_backend="njit_parallel")

    assert np.array_equal(baseline, njit_result)
    assert np.array_equal(baseline, njit_parallel_result)
