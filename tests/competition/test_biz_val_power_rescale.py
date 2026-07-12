"""biz_value tests for ``mlframe.competition.power_rescale``.

Covers both competition-only tricks:

* ``power_rescale_to_target_sum`` — must hit the target sum to a tight tolerance while
  exactly preserving rank order (Spearman rho == 1.0 vs the original probabilities).
* ``asymmetric_scale_by_sign`` — on a synthetic signed-target scenario where the model
  systematically over-scales negative predictions relative to positive ones, the 1-D CV
  sweep must recover a best scale meaningfully different from 1.0 and improve the metric.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from mlframe.competition.power_rescale import asymmetric_scale_by_sign, power_rescale_to_target_sum


def test_biz_val_power_rescale_hits_target_sum_and_preserves_rank_order():
    rng = np.random.default_rng(0)
    n = 5000
    probs = rng.beta(a=0.5, b=8.0, size=n)  # right-skewed, many low probabilities, like a rare-event model
    true_sum = float(np.sum(probs))
    # extrapolated leaderboard count is materially higher than the raw model's implied sum
    target_sum = true_sum * 1.8

    rescaled = power_rescale_to_target_sum(probs, target_sum)

    achieved_sum = float(np.sum(rescaled))
    assert abs(achieved_sum - target_sum) < 1e-6, f"expected sum~={target_sum}, got {achieved_sum}"

    rho, _ = spearmanr(probs, rescaled)
    assert rho > 1.0 - 1e-9, f"expected (near-)exact rank preservation, got spearman rho={rho}"
    # argsort-based check is immune to spearmanr's own tie-handling floating-point noise.
    assert np.array_equal(np.argsort(probs, kind="stable"), np.argsort(rescaled, kind="stable"))

    # sanity: exponent found must be < 1 (raising sum requires p < 1 since 0 < probs < 1)
    assert np.all(rescaled >= probs) or np.all(rescaled <= probs)


def test_biz_val_power_rescale_hits_target_sum_when_lowering():
    rng = np.random.default_rng(1)
    n = 3000
    probs = rng.beta(a=2.0, b=2.0, size=n)
    true_sum = float(np.sum(probs))
    target_sum = true_sum * 0.4  # extrapolated count lower than raw model implies

    rescaled = power_rescale_to_target_sum(probs, target_sum)
    achieved_sum = float(np.sum(rescaled))
    assert abs(achieved_sum - target_sum) < 1e-6, f"expected sum~={target_sum}, got {achieved_sum}"

    rho, _ = spearmanr(probs, rescaled)
    assert rho > 1.0 - 1e-9, f"expected (near-)exact rank preservation, got spearman rho={rho}"
    assert np.array_equal(np.argsort(probs, kind="stable"), np.argsort(rescaled, kind="stable"))


def _make_asymmetric_signed_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    y_true = rng.normal(size=n)
    # model consistently over-scales POSITIVE predictions by 1.6x relative to true signal, while
    # negative predictions are well calibrated -- asymmetric_scale_by_sign divides positive preds
    # by the swept scale (shrinking them back down) and multiplies negative preds by the same scale,
    # so this asymmetry direction is exactly what the fitted scale factor should correct for.
    y_pred = np.where(y_true > 0, y_true * 1.6, y_true * 1.0) + rng.normal(scale=0.05, size=n)
    return y_true, y_pred


def test_biz_val_asymmetric_scale_by_sign_recovers_nontrivial_scale_and_improves_metric():
    y_true, y_pred = _make_asymmetric_signed_dataset(n=4000, seed=42)

    def neg_mse(candidate: np.ndarray) -> float:
        return -float(np.mean((y_true - candidate) ** 2))

    baseline_score = neg_mse(y_pred)

    rescaled, best_scale = asymmetric_scale_by_sign(y_pred, neg_mse, scale_range=(1.0, 2.0), n_steps=101)

    rescaled_score = neg_mse(rescaled)

    # the fitted scale must genuinely differ from the no-op (1.0); the true positive-side asymmetry is
    # 1.6x but the optimal joint scale trades off against the (correctly calibrated) negative side, so
    # the recovered optimum lands below 1.6 while still being clearly non-trivial.
    assert best_scale > 1.15, f"expected best_scale meaningfully > 1.0 (positive-side asymmetry 1.6x), got {best_scale}"

    # the sweep must materially improve the metric over doing nothing
    assert rescaled_score > baseline_score, f"expected improvement, baseline={baseline_score}, rescaled={rescaled_score}"
    improvement_ratio = (rescaled_score - baseline_score) / abs(baseline_score)
    assert improvement_ratio > 0.10, f"expected >10% relative improvement in neg-MSE, got {improvement_ratio:.4f}"


def test_biz_val_asymmetric_scale_by_sign_noop_when_no_asymmetry():
    rng = np.random.default_rng(7)
    n = 2000
    y_true = rng.normal(size=n)
    y_pred = y_true + rng.normal(scale=0.05, size=n)  # symmetric, well-calibrated

    def neg_mse(candidate: np.ndarray) -> float:
        return -float(np.mean((y_true - candidate) ** 2))

    baseline_score = neg_mse(y_pred)
    rescaled, best_scale = asymmetric_scale_by_sign(y_pred, neg_mse, scale_range=(1.0, 2.0), n_steps=101)

    assert abs(best_scale - 1.0) < 0.05, f"expected near-identity scale on symmetric data, got {best_scale}"
    assert neg_mse(rescaled) <= baseline_score + 1e-3
