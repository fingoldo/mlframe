"""biz_value test for ``mlframe.competition.quantization_recovery``.

COMPETITION/EXPLORATORY USE ONLY -- see ``mlframe.competition`` package docstring.

The win: a competition anonymizes an originally-integer field by scaling it by an
unknown small denominator and adding tiny noise (``x = true_int * step + noise``).
``detect_quantization_step`` should recover ``step`` tightly, and de-rounding
(``round(x / step) * step``) should recover the true integer values with high exact-match
accuracy -- letting downstream feature engineering treat the recovered values as clean
integers instead of noisy scaled floats.
"""

from __future__ import annotations

import numpy as np

from mlframe.competition.quantization_recovery import (
    QuantizationRankResult,
    derounded_feature,
    detect_quantization_step,
    rank_features_by_quantization_confidence,
)


def _make_scaled_noised_integer_feature(rng: np.random.Generator, n: int, true_step: float, noise_scale: float, int_high: int = 500):
    true_int = rng.integers(0, int_high, size=n)
    noise = rng.normal(0.0, noise_scale, size=n)
    x = true_int * true_step + noise
    return x, true_int


def test_biz_val_quantization_recovery_step_and_derounding_beat_raw_noise():
    rng = np.random.default_rng(0)
    true_step = 0.037
    x, true_int = _make_scaled_noised_integer_feature(rng, n=4000, true_step=true_step, noise_scale=true_step * 0.03)

    recovered_step = detect_quantization_step(x)
    assert recovered_step == recovered_step  # not NaN
    rel_err = abs(recovered_step - true_step) / true_step
    assert rel_err < 0.05, f"recovered step {recovered_step} should be within 5% of true step {true_step} (rel_err={rel_err:.4f})"

    x_derounded = derounded_feature(x, recovered_step)
    recovered_int = np.round(x_derounded / true_step).astype(int)
    exact_match_rate = float(np.mean(recovered_int == true_int))
    assert exact_match_rate > 0.95, f"de-rounding should recover >95% of true integers exactly, got {exact_match_rate:.4f}"

    # de-rounding is a real win over using the raw noised value directly: MAE against the
    # true (unscaled-back) target drops sharply once de-rounded.
    raw_mae = float(np.mean(np.abs(x / true_step - true_int)))
    derounded_mae = float(np.mean(np.abs(recovered_int - true_int)))
    assert derounded_mae < raw_mae * 0.1, f"de-rounded MAE ({derounded_mae:.4f}) should be far below raw MAE ({raw_mae:.4f})"


def test_biz_val_quantization_recovery_rejects_non_quantized_continuous_feature():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, size=4000)  # genuinely continuous, no hidden integer grid
    result = rank_features_by_quantization_confidence({"continuous": x, "quantized": rng.integers(0, 200, size=4000) * 0.05})
    assert isinstance(result, QuantizationRankResult)
    assert result.feature_names[0] == "quantized"
    assert result.confidences[0] > 0.8
    assert result.confidences[result.feature_names.index("continuous")] < 0.5


def test_detect_quantization_step_degenerate_inputs_return_nan():
    assert detect_quantization_step(np.array([1.0])) != detect_quantization_step(np.array([1.0]))  # NaN != NaN
    assert detect_quantization_step(np.array([])) != detect_quantization_step(np.array([]))
    assert detect_quantization_step(np.array([5.0, 5.0, 5.0])) != detect_quantization_step(np.array([5.0, 5.0, 5.0]))


def test_derounded_feature_passthrough_on_invalid_step():
    x = np.array([1.1, 2.2, 3.3])
    out_nan = derounded_feature(x, float("nan"))
    out_zero = derounded_feature(x, 0.0)
    assert np.array_equal(out_nan, x)
    assert np.array_equal(out_zero, x)


def test_biz_val_quantization_recovery_robust_to_higher_noise():
    rng = np.random.default_rng(2)
    true_step = 0.1
    x, true_int = _make_scaled_noised_integer_feature(rng, n=6000, true_step=true_step, noise_scale=true_step * 0.08, int_high=100)
    recovered_step = detect_quantization_step(x)
    rel_err = abs(recovered_step - true_step) / true_step
    assert rel_err < 0.05

    recovered_int = np.round(derounded_feature(x, recovered_step) / true_step).astype(int)
    exact_match_rate = float(np.mean(recovered_int == true_int))
    assert exact_match_rate > 0.9, f"even at higher noise, de-rounding should recover >90% exactly, got {exact_match_rate:.4f}"
