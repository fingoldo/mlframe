"""biz_value test for ``votenrank.adversarial_stochastic_blend``.

The win: when train is dominated by one region (A) and test is dominated by a DIFFERENT region (B), an
ordinary blend-weight fit on the full train set is dominated by region A rows and favors whichever model is
best in region A -- even if that model is poor in region B, where the actual test set lives. Weighting the
Monte-Carlo resamples by adversarial-validation test-likeness (over-sampling the sparse-in-train,
common-in-test region B rows) should shift the fitted blend weights toward the model that's actually good in
region B, giving materially better held-out test performance than the naive full-train-unweighted fit.
"""
from __future__ import annotations

import numpy as np

from mlframe.votenrank.adversarial_stochastic_blend import adversarial_stochastic_blend, compute_test_likeness
from mlframe.votenrank.constrained_weight_blend import constrained_weight_blend


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _make_drifted_dataset(seed: int):
    rng = np.random.default_rng(seed)
    n_region_a_train, n_region_b_train = 900, 100  # train dominated by region A
    n_region_b_test = 400  # test is ENTIRELY region B -- the drift scenario

    x_a_train = rng.uniform(0, 1, n_region_a_train)
    x_b_train = rng.uniform(2, 3, n_region_b_train)
    x_train = np.concatenate([x_a_train, x_b_train])
    y_train = np.sin(x_train * 3.0)

    x_test = rng.uniform(2, 3, n_region_b_test)
    y_test = np.sin(x_test * 3.0)

    def model_1(x):  # good in region A [0,1], poor (noisy) in region B [2,3]
        is_a = x < 1.5
        return np.where(is_a, np.sin(x * 3.0), np.sin(x * 3.0) + 1.5 * rng.standard_normal(len(x)))

    def model_2(x):  # good in region B [2,3], poor (noisy) in region A
        is_a = x < 1.5
        return np.where(is_a, np.sin(x * 3.0) + 1.5 * rng.standard_normal(len(x)), np.sin(x * 3.0))

    train_preds = [model_1(x_train), model_2(x_train)]
    test_preds = [model_1(x_test), model_2(x_test)]
    return x_train, y_train, train_preds, x_test, y_test, test_preds


def test_biz_val_adversarial_stochastic_blend_beats_naive_full_train_fit_under_drift():
    x_train, y_train, train_preds, x_test, y_test, test_preds = _make_drifted_dataset(seed=0)

    naive_result = constrained_weight_blend(train_preds, y_train, _rmse, n_restarts=5, random_state=0)
    naive_test_pred = np.tensordot(naive_result["weights"], np.stack(test_preds, axis=0), axes=(0, 0))
    naive_test_rmse = _rmse(y_test, naive_test_pred)

    test_likeness = compute_test_likeness(x_train.reshape(-1, 1), x_test.reshape(-1, 1), cv=5, random_state=0)
    adv_result = adversarial_stochastic_blend(train_preds, y_train, test_likeness, _rmse, n_iterations=100, sample_frac=0.6, n_restarts=2, random_state=0)
    adv_test_pred = np.tensordot(adv_result["weights"], np.stack(test_preds, axis=0), axes=(0, 0))
    adv_test_rmse = _rmse(y_test, adv_test_pred)

    improvement = (naive_test_rmse - adv_test_rmse) / naive_test_rmse
    assert improvement > 0.2, f"expected adversarial stochastic blend to beat naive full-train fit by >20% RMSE on the drifted test set, got naive={naive_test_rmse:.4f} adv={adv_test_rmse:.4f} (improvement={improvement:.4f})"
    assert adv_result["weights"][1] > naive_result["weights"][1], f"expected the adversarial blend to shift weight toward model_2 (good in the test-like region), got adv={adv_result['weights']} naive={naive_result['weights']}"


def test_compute_test_likeness_flags_drifted_region_correctly():
    x_train, _, _, x_test, _, _ = _make_drifted_dataset(seed=1)
    test_likeness = compute_test_likeness(x_train.reshape(-1, 1), x_test.reshape(-1, 1), cv=5, random_state=1)

    is_region_a = x_train < 1.5
    mean_likeness_a = test_likeness[is_region_a].mean()
    mean_likeness_b = test_likeness[~is_region_a].mean()
    assert mean_likeness_b > mean_likeness_a, f"expected region B (test-like) rows to score higher test-likeness, got a={mean_likeness_a:.4f} b={mean_likeness_b:.4f}"


def test_adversarial_stochastic_blend_weights_sum_to_one():
    rng = np.random.default_rng(2)
    y_true = rng.normal(size=200)
    preds = [y_true + 0.3 * rng.standard_normal(200), y_true + 0.5 * rng.standard_normal(200)]
    test_likeness = rng.uniform(size=200)
    result = adversarial_stochastic_blend(preds, y_true, test_likeness, _rmse, n_iterations=20, sample_frac=0.7, n_restarts=1, random_state=2)
    assert np.isclose(result["weights"].sum(), 1.0)
    assert result["weight_std"].shape == (2,)
