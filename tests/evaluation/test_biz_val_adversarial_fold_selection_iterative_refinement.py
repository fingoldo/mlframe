"""biz_value test for the iterative-refinement (peel-back) mode of ``build_test_like_validation_fold``.

The win: when train/test differ on TWO features -- a dominant but target-IRRELEVANT drift feature (e.g. a
collection-date artifact, here ``leak``, offset by 5 std between train and test but uncorrelated with y) and a
subtler, target-RELEVANT covariate-shift feature (``x``, a mixture-regime shift as in the base biz_value test)
-- a one-shot adversarial classifier fits mostly on the dominant ``leak`` feature and selects a validation fold
that is not actually enriched for the regime that matches test on ``x``. Peeling ``leak`` away after the first
iteration forces the classifier onto ``x``, producing a fold whose validation error tracks true test error far
more closely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from mlframe.evaluation.adversarial_fold_selection import build_test_like_validation_fold


def _make_layered_drift_scenario(seed: int):
    rng = np.random.default_rng(seed)
    n_train, n_test = 3000, 600

    # segment: the REAL distinguishing feature -- its distribution genuinely differs between train's minority
    # regime B and majority regime A, and test is drawn entirely from regime B (mirrors the base biz_value
    # scenario). x: the value-relevant covariate used for the downstream y prediction; x's own MARGINAL
    # distribution does NOT shift between train and test (only how y depends on x, via regime, does) -- so a
    # classifier can only find the real regime-B-like rows through segment, not through x.
    regime = rng.random(n_train) < 0.2  # True = minority regime B, matches test.
    x_train = rng.normal(0, 1, n_train)
    segment_train = np.where(regime, rng.normal(3, 0.5, n_train), rng.normal(0, 0.5, n_train))
    w_A, w_B = 2.0, -2.0
    y_train = np.where(regime, w_B * x_train, w_A * x_train) + rng.normal(0, 0.3, n_train)

    x_test = rng.normal(0, 1, n_test)
    segment_test = rng.normal(3, 0.5, n_test)
    y_test = w_B * x_test + rng.normal(0, 0.3, n_test)

    # leak: dominant train/test separator, pure artifact -- independent of y and of regime in both train/test.
    leak_train = rng.normal(0, 1, n_train)
    leak_test = rng.normal(5, 1, n_test)

    X_train = pd.DataFrame({"leak": leak_train, "segment": segment_train, "x": x_train})
    X_test = pd.DataFrame({"leak": leak_test, "segment": segment_test, "x": x_test})

    return X_train, y_train, regime, X_test, y_test


def test_biz_val_adversarial_fold_iterative_refinement_beats_one_shot_under_dominant_leak():
    X_train, y_train, regime, X_test, y_test = _make_layered_drift_scenario(seed=0)

    val_idx_oneshot, remainder_oneshot = build_test_like_validation_fold(X_train, X_test, val_fraction=0.2, seed=0)
    val_idx_iter, remainder_iter, history = build_test_like_validation_fold(
        X_train,
        X_test,
        val_fraction=0.2,
        seed=0,
        n_iterations=2,
        top_k_drop_per_iteration=1,
        return_history=True,
    )

    def _fit_eval(remainder_idx, val_idx):
        model = LinearRegression().fit(X_train.iloc[remainder_idx][["x"]], y_train[remainder_idx])
        val_mae = mean_absolute_error(y_train[val_idx], model.predict(X_train.iloc[val_idx][["x"]]))
        test_mae = mean_absolute_error(y_test, model.predict(X_test[["x"]]))
        return val_mae, test_mae

    val_mae_oneshot, test_mae_oneshot = _fit_eval(remainder_oneshot, val_idx_oneshot)
    val_mae_iter, test_mae_iter = _fit_eval(remainder_iter, val_idx_iter)

    gap_oneshot = abs(val_mae_oneshot - test_mae_oneshot)
    gap_iter = abs(val_mae_iter - test_mae_iter)

    assert gap_iter < gap_oneshot * 0.5, (
        f"peeling the dominant leak feature should give a fold that tracks true test performance far more "
        f"closely than the one-shot fold dominated by the leak artifact: gap_iter={gap_iter:.4f} "
        f"gap_oneshot={gap_oneshot:.4f}"
    )
    # the iterative fold should be enriched for the minority regime that actually matches test on x.
    assert regime[val_idx_iter].mean() > regime[val_idx_oneshot].mean()

    # AUC-decay curve: 2 iterations recorded, first fit finds the leak trivially separable, dropping it.
    assert len(history) == 2
    assert history[0]["dropped_features"] == ["leak"]
    assert history[1]["dropped_features"] == []
    assert history[0]["auc"] > 0.9  # near-perfect separation on the dominant leak feature.


def test_build_test_like_validation_fold_default_args_unchanged_by_new_params():
    X_train, _y_train, _regime, X_test, _y_test = _make_layered_drift_scenario(seed=1)

    val_idx_default, remainder_default = build_test_like_validation_fold(X_train, X_test, val_fraction=0.2, seed=3)
    val_idx_explicit, remainder_explicit = build_test_like_validation_fold(
        X_train,
        X_test,
        val_fraction=0.2,
        seed=3,
        n_iterations=1,
        top_k_drop_per_iteration=0,
        return_history=False,
    )

    np.testing.assert_array_equal(val_idx_default, val_idx_explicit)
    np.testing.assert_array_equal(remainder_default, remainder_explicit)
