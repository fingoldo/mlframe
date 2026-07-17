"""biz_value test for ``calibration.smoothed_override.apply_smoothed_override``.

The win (4th_mechanisms-of-action-moa-prediction.md, Theo Viel): a confident-override rule is usually right
but OCCASIONALLY wrong (a mislabeled ground truth, a stale lookup, an edge case). A hard override (a=1.0)
gets the common "rule is right" case perfectly but is catastrophically wrong whenever the rule itself is
wrong (the model's own prediction is discarded entirely). A convex blend (a=0.9) captures nearly all the
benefit on the common case while bounding the damage when the rule is wrong -- net lower overall error.
"""

from __future__ import annotations

import numpy as np

from mlframe.calibration.smoothed_override import apply_smoothed_override


def _make_override_scenario(n: int, rule_error_rate: float, seed: int):
    """Helper that make override scenario."""
    rng = np.random.default_rng(seed)
    y_true = rng.uniform(0, 1, n)

    # model's own prediction: noisy but unbiased.
    model_pred = y_true + rng.normal(scale=0.15, size=n)

    # the override rule fires on every row; it's usually near-exact but sometimes wrong by a material
    # margin (a mislabeled/stale case), independent of the model's own error.
    rule_is_wrong = rng.random(n) < rule_error_rate
    known_label = np.where(rule_is_wrong, y_true + rng.normal(loc=1.5, scale=0.5, size=n), y_true + rng.normal(scale=0.01, size=n))
    override_mask = np.ones(n, dtype=bool)

    return y_true, model_pred, known_label, override_mask


def test_biz_val_smoothed_override_beats_hard_override_when_rule_occasionally_wrong():
    """Smoothed override beats hard override when rule occasionally wrong."""
    y_true, model_pred, known_label, override_mask = _make_override_scenario(n=3000, rule_error_rate=0.08, seed=0)

    hard_overridden = apply_smoothed_override(model_pred, known_label, override_mask, a=1.0)
    smoothed_overridden = apply_smoothed_override(model_pred, known_label, override_mask, a=0.9)

    mae_hard = float(np.mean(np.abs(y_true - hard_overridden)))
    mae_smoothed = float(np.mean(np.abs(y_true - smoothed_overridden)))
    mae_model_alone = float(np.mean(np.abs(y_true - model_pred)))

    assert mae_smoothed < mae_hard, (
        f"expected the smoothed (a=0.9) override to beat the hard (a=1.0) override when the rule is occasionally wrong, got smoothed={mae_smoothed:.4f} hard={mae_hard:.4f}"
    )
    assert mae_smoothed < mae_model_alone, (
        f"expected the smoothed override to still beat the model alone (capturing most of the rule's benefit), got smoothed={mae_smoothed:.4f} model_alone={mae_model_alone:.4f}"
    )


def test_apply_smoothed_override_exact_blend_and_mask_respected():
    """Apply smoothed override exact blend and mask respected."""
    prediction = np.array([1.0, 2.0, 3.0, 4.0])
    known_label = np.array([10.0, 20.0, 30.0, 40.0])
    mask = np.array([True, False, True, False])

    out = apply_smoothed_override(prediction, known_label, mask, a=0.9)
    np.testing.assert_allclose(out, [0.9 * 10.0 + 0.1 * 1.0, 2.0, 0.9 * 30.0 + 0.1 * 3.0, 4.0])


def test_apply_smoothed_override_a_zero_is_noop_a_one_is_hard_replace():
    """Apply smoothed override a zero is noop a one is hard replace."""
    prediction = np.array([1.0, 2.0])
    known_label = np.array([100.0, 200.0])
    mask = np.array([True, True])

    noop = apply_smoothed_override(prediction, known_label, mask, a=0.0)
    np.testing.assert_allclose(noop, prediction)

    hard = apply_smoothed_override(prediction, known_label, mask, a=1.0)
    np.testing.assert_allclose(hard, known_label)


def test_apply_smoothed_override_invalid_a_raises():
    """Apply smoothed override invalid a raises."""
    import pytest

    with pytest.raises(ValueError):
        apply_smoothed_override(np.array([1.0]), np.array([2.0]), np.array([True]), a=1.5)
