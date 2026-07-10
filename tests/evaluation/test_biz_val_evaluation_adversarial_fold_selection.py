"""biz_value test for ``evaluation.build_test_like_validation_fold``.

The win: when train is a mixture of two regimes (80% majority regime A, 20% minority regime B) and test is
drawn ENTIRELY from regime B, a model's validation score on a RANDOM fold (mostly regime-A rows) is a
misleading proxy for true test performance -- but validating on the adversarially-selected "most test-like"
fold (which should end up dominated by regime-B rows) gives a validation score that closely tracks the true
test score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from mlframe.evaluation.adversarial_fold_selection import build_test_like_validation_fold


def _make_shifted_scenario(seed: int):
    rng = np.random.default_rng(seed)
    n_train, n_test = 3000, 600

    regime = rng.random(n_train) < 0.2  # True = minority regime B, False = majority regime A
    x_train = rng.normal(0, 1, n_train)
    segment_train = np.where(regime, rng.normal(3, 0.5, n_train), rng.normal(0, 0.5, n_train))
    w_A, w_B = 2.0, -2.0
    y_train = np.where(regime, w_B * x_train, w_A * x_train) + rng.normal(0, 0.3, n_train)
    X_train = pd.DataFrame({"x": x_train, "segment": segment_train})

    x_test = rng.normal(0, 1, n_test)
    segment_test = rng.normal(3, 0.5, n_test)
    y_test = w_B * x_test + rng.normal(0, 0.3, n_test)
    X_test = pd.DataFrame({"x": x_test, "segment": segment_test})

    return X_train, y_train, regime, X_test, y_test


def test_biz_val_adversarial_fold_gives_a_validation_score_closer_to_true_test_score():
    X_train, y_train, regime, X_test, y_test = _make_shifted_scenario(seed=0)
    n_train = len(y_train)

    val_idx_adv, remainder_idx_adv = build_test_like_validation_fold(X_train, X_test, val_fraction=0.2, seed=0)

    rng = np.random.default_rng(1)
    perm = rng.permutation(n_train)
    n_val = len(val_idx_adv)
    val_idx_random, remainder_idx_random = perm[:n_val], perm[n_val:]

    def _fit_eval(remainder_idx, val_idx):
        model = LinearRegression().fit(X_train.iloc[remainder_idx][["x"]], y_train[remainder_idx])
        val_mae = mean_absolute_error(y_train[val_idx], model.predict(X_train.iloc[val_idx][["x"]]))
        test_mae = mean_absolute_error(y_test, model.predict(X_test[["x"]]))
        return val_mae, test_mae

    val_mae_adv, test_mae_adv = _fit_eval(remainder_idx_adv, val_idx_adv)
    val_mae_random, test_mae_random = _fit_eval(remainder_idx_random, val_idx_random)

    gap_adv = abs(val_mae_adv - test_mae_adv)
    gap_random = abs(val_mae_random - test_mae_random)

    assert gap_adv < gap_random * 0.2, (
        f"the adversarially-selected fold should track true test performance far more closely than a random fold: "
        f"gap_adv={gap_adv:.4f} gap_random={gap_random:.4f}"
    )
    # the fold should actually be enriched for the minority regime that matches test.
    assert regime[val_idx_adv].mean() > 0.9


def test_build_test_like_validation_fold_partitions_without_overlap():
    X_train, y_train, regime, X_test, y_test = _make_shifted_scenario(seed=2)
    val_idx, remainder_idx = build_test_like_validation_fold(X_train, X_test, val_fraction=0.25, seed=2)
    assert len(set(val_idx.tolist()) & set(remainder_idx.tolist())) == 0
    assert len(val_idx) + len(remainder_idx) == len(y_train)


def test_build_test_like_validation_fold_empty_train_raises():
    import pytest

    with pytest.raises(ValueError):
        build_test_like_validation_fold(pd.DataFrame({"x": []}), pd.DataFrame({"x": [1.0]}))
