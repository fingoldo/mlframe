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
    assert improvement > 0.2, (
        f"expected adversarial stochastic blend to beat naive full-train fit by >20% RMSE on the drifted test set, got naive={naive_test_rmse:.4f} adv={adv_test_rmse:.4f} (improvement={improvement:.4f})"
    )
    assert adv_result["weights"][1] > naive_result["weights"][1], (
        f"expected the adversarial blend to shift weight toward model_2 (good in the test-like region), got adv={adv_result['weights']} naive={naive_result['weights']}"
    )


def test_compute_test_likeness_flags_drifted_region_correctly():
    x_train, _, _, x_test, _, _ = _make_drifted_dataset(seed=1)
    test_likeness = compute_test_likeness(x_train.reshape(-1, 1), x_test.reshape(-1, 1), cv=5, random_state=1)

    is_region_a = x_train < 1.5
    mean_likeness_a = test_likeness[is_region_a].mean()
    mean_likeness_b = test_likeness[~is_region_a].mean()
    assert mean_likeness_b > mean_likeness_a, (
        f"expected region B (test-like) rows to score higher test-likeness, got a={mean_likeness_a:.4f} b={mean_likeness_b:.4f}"
    )


def test_adversarial_stochastic_blend_weights_sum_to_one():
    rng = np.random.default_rng(2)
    y_true = rng.normal(size=200)
    preds = [y_true + 0.3 * rng.standard_normal(200), y_true + 0.5 * rng.standard_normal(200)]
    test_likeness = rng.uniform(size=200)
    result = adversarial_stochastic_blend(preds, y_true, test_likeness, _rmse, n_iterations=20, sample_frac=0.7, n_restarts=1, random_state=2)
    assert np.isclose(result["weights"].sum(), 1.0)
    assert result["weight_std"].shape == (2,)


def test_adversarial_stochastic_blend_default_output_unchanged_by_diagnostics_opt_in():
    """The new diagnostic params must be strictly opt-in -- omitting them must not change any existing key."""
    rng = np.random.default_rng(3)
    y_true = rng.normal(size=150)
    preds = [y_true + 0.3 * rng.standard_normal(150), y_true + 0.5 * rng.standard_normal(150)]
    test_likeness = rng.uniform(size=150)

    baseline = adversarial_stochastic_blend(preds, y_true, test_likeness, _rmse, n_iterations=15, sample_frac=0.7, n_restarts=1, random_state=3)
    with_flags_off = adversarial_stochastic_blend(
        preds, y_true, test_likeness, _rmse, n_iterations=15, sample_frac=0.7, n_restarts=1, random_state=3, track_convergence=False, discriminator_auc=None
    )

    assert set(baseline.keys()) == {"weights", "ensemble_pred", "loss", "weight_std"}
    assert set(with_flags_off.keys()) == set(baseline.keys())
    np.testing.assert_array_equal(baseline["weights"], with_flags_off["weights"])
    np.testing.assert_array_equal(baseline["ensemble_pred"], with_flags_off["ensemble_pred"])
    assert baseline["loss"] == with_flags_off["loss"]
    np.testing.assert_array_equal(baseline["weight_std"], with_flags_off["weight_std"])


def test_biz_val_adversarial_stochastic_blend_convergence_diagnostic_genuine_drift_is_stable_and_trusted():
    """Genuine, strong train/test drift -> discriminator AUC well above chance -> is_trustworthy True and a
    high stability_score (weight estimates should have converged, not still be flailing between resamples)."""
    x_train, y_train, train_preds, x_test, _y_test, _test_preds = _make_drifted_dataset(seed=10)

    test_likeness, likeness_diag = compute_test_likeness(x_train.reshape(-1, 1), x_test.reshape(-1, 1), cv=5, random_state=10, return_diagnostics=True)
    assert likeness_diag["auc"] > 0.85, f"expected the discriminator to reliably separate the drifted regions, got auc={likeness_diag['auc']:.4f}"

    result = adversarial_stochastic_blend(
        train_preds,
        y_train,
        test_likeness,
        _rmse,
        n_iterations=100,
        sample_frac=0.6,
        n_restarts=2,
        random_state=10,
        track_convergence=True,
        discriminator_auc=likeness_diag["auc"],
        fallback_to_uniform_if_untrustworthy=True,
    )

    assert result["is_trustworthy"] is True
    assert result["fallback_applied"] is False
    assert result["stability_score"] > 0.5, f"expected a high stability score under genuine drift, got {result['stability_score']:.4f}"
    assert result["convergence_curve"].shape == (100,)
    # the expanding-window coefficient of variation should trend down as more MC iterations accumulate.
    assert result["convergence_curve"][-1] < result["convergence_curve"][4], "expected convergence curve to decrease as iterations accumulate"


def test_biz_val_adversarial_stochastic_blend_convergence_diagnostic_no_drift_flags_untrustworthy():
    """No real train/test drift -> discriminator AUC near chance -> is_trustworthy False, and the
    opt-in fallback correctly discards the noisy test-likeness weighting in favor of a uniform blend."""
    rng = np.random.default_rng(11)
    n = 600
    x_train = rng.uniform(0, 1, n)  # train and test drawn from the SAME distribution -- no drift.
    x_test = rng.uniform(0, 1, n)
    y_train = np.sin(x_train * 3.0)

    def model_1(x):
        return np.sin(x * 3.0) + 0.4 * rng.standard_normal(len(x))

    def model_2(x):
        return np.sin(x * 3.0) + 0.1 * rng.standard_normal(len(x))

    train_preds = [model_1(x_train), model_2(x_train)]

    test_likeness, likeness_diag = compute_test_likeness(x_train.reshape(-1, 1), x_test.reshape(-1, 1), cv=5, random_state=11, return_diagnostics=True)
    assert likeness_diag["auc_margin_from_chance"] < 0.1, f"expected a near-chance discriminator AUC with no real drift, got auc={likeness_diag['auc']:.4f}"

    result = adversarial_stochastic_blend(
        train_preds,
        y_train,
        test_likeness,
        _rmse,
        n_iterations=100,
        sample_frac=0.6,
        n_restarts=2,
        random_state=11,
        track_convergence=True,
        discriminator_auc=likeness_diag["auc"],
        fallback_to_uniform_if_untrustworthy=True,
    )

    assert result["is_trustworthy"] is False
    assert result["fallback_applied"] is True
    assert np.allclose(result["weights"], 0.5), f"expected uniform fallback weights, got {result['weights']}"
