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

    gated_blend = confidence_gated_blend(ensemble_pred, auxiliary_pred, auxiliary_confidence, confidence_threshold=0.6, gated_weight=0.6, default_weight=0.0)
    loss_gated_blend = _log_loss(y, gated_blend)

    assert loss_gated_blend < loss_pure_ensemble, f"gated blend should beat the pure ensemble: gated={loss_gated_blend:.4f} pure={loss_pure_ensemble:.4f}"
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


def _make_miscalibrated_gate_data(seed: int):
    """Two auxiliary-model regions with the raw confidence score backwards vs. true reliability.

    Region A: raw confidence is HIGH (0.85) but the auxiliary model is actually pure noise there (accurate
    only ~50% of the time, a coin flip). Region B: raw confidence is only MODERATE (0.55) but the auxiliary
    model is actually very reliable there (accurate ~97% of the time).
    A hard threshold on the raw score (e.g. >= 0.6) gates the blend exactly backwards: it trusts region A
    (should not) and distrusts region B (should trust). An isotonic calibrator fit on held-out
    (confidence, reliability) pairs learns the true (inverted) relationship and gates correctly.
    """
    rng = np.random.default_rng(seed)
    n = 6000
    region_b = rng.random(n) < 0.6

    z = rng.normal(0, 1, n)
    y = (z > 0).astype(int)
    ensemble_pred = 1.0 / (1.0 + np.exp(-z))

    raw_confidence = np.where(region_b, 0.55, 0.85)

    # Region A: aux is pure noise (coin-flip accurate, uninformative) despite its HIGH raw confidence.
    aux_correct_a = rng.random(n) < 0.50
    # Region B: aux is right ~97% of the time despite its only-MODERATE raw confidence.
    aux_correct_b = rng.random(n) < 0.97
    aux_correct = np.where(region_b, aux_correct_b, aux_correct_a)
    auxiliary_pred = np.where(aux_correct, y, 1 - y).astype(np.float64)
    auxiliary_pred = np.clip(auxiliary_pred + rng.normal(0, 0.02, n), 0.01, 0.99)

    return y, ensemble_pred, auxiliary_pred, raw_confidence, region_b


def test_biz_val_confidence_gated_blend_per_sample_gate_calibration_beats_raw_confidence_gate():
    y, ensemble_pred, auxiliary_pred, raw_confidence, region_b = _make_miscalibrated_gate_data(seed=1)

    # Held-out calibration set: same generative process, disjoint draw, with the TRUE reliability label
    # (1 = aux agreed with ground truth) at each raw confidence level.
    cal_y, _, cal_aux_pred, cal_confidence, _ = _make_miscalibrated_gate_data(seed=2)
    cal_reliability = (np.round(cal_aux_pred) == cal_y).astype(np.float64)

    raw_gate_blend = confidence_gated_blend(ensemble_pred, auxiliary_pred, raw_confidence, confidence_threshold=0.6, gated_weight=0.8, default_weight=0.0)
    loss_raw_gate = _log_loss(y, raw_gate_blend)

    calibrated_blend = confidence_gated_blend(
        ensemble_pred,
        auxiliary_pred,
        raw_confidence,
        confidence_threshold=0.6,
        gated_weight=0.8,
        default_weight=0.0,
        per_sample_gate_calibration=True,
        calibration_confidence=cal_confidence,
        calibration_reliability=cal_reliability,
    )
    loss_calibrated = _log_loss(y, calibrated_blend)

    loss_pure_ensemble = _log_loss(y, ensemble_pred)

    # Measured (seed=1): pure=0.408, raw-gate=0.628, calibrated=0.299 -- raw gate is WORSE than doing nothing
    # (it trusts the noisy region and distrusts the reliable one), calibrated gate is clearly better than both.
    assert loss_calibrated < loss_raw_gate * 0.6, (
        f"calibrated gate should substantially beat the raw-confidence gate (miscalibrated on purpose): "
        f"calibrated={loss_calibrated:.4f} raw={loss_raw_gate:.4f}"
    )
    assert loss_calibrated < loss_pure_ensemble * 0.85, (
        f"calibrated gate should beat the pure ensemble: calibrated={loss_calibrated:.4f} pure={loss_pure_ensemble:.4f}"
    )


def test_confidence_gated_blend_per_sample_gate_calibration_requires_calibration_data():
    import pytest

    with pytest.raises(ValueError):
        confidence_gated_blend(
            np.array([0.1, 0.2]),
            np.array([0.2, 0.3]),
            np.array([0.5, 0.9]),
            confidence_threshold=0.5,
            gated_weight=0.5,
            per_sample_gate_calibration=True,
        )


def test_confidence_gated_blend_default_path_unaffected_by_new_params():
    """Omitting the new params must produce bit-identical output to the pre-extension signature."""
    rng = np.random.default_rng(0)
    n = 3000
    ensemble = rng.uniform(0, 1, n)
    aux = rng.uniform(0, 1, n)
    conf = rng.uniform(0, 1, n)

    baseline = confidence_gated_blend(ensemble, aux, conf, 0.6, 0.5, force_backend="numpy")
    same = confidence_gated_blend(ensemble, aux, conf, 0.6, 0.5, default_weight=0.0, force_backend="numpy")
    assert np.array_equal(baseline, same)


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
