"""biz_value test for ``feature_engineering.control_difference_augment``.

The win: with a tiny real treated-sample training set (overfitting-prone, especially for a tree ensemble in
high dimensions), augmenting with control-difference synthetic rows (which carry a realistic batch-noise
draw and the correct treatment label) should improve held-out AUC over training on the small real set alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from mlframe.feature_engineering.control_difference_augment import control_difference_augment


def _make_treated(n: int, label: int, seed: int, n_features: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal = 1.0 if label == 1 else 0.0
    return signal + rng.normal(0, 3.0, (n, n_features))


def test_biz_val_control_difference_augment_improves_small_sample_generalization():
    n_features = 50
    cols = [f"f{i}" for i in range(n_features)]
    n_train_per_class = 10

    y_train = np.array([0] * n_train_per_class + [1] * n_train_per_class)
    X_train = np.vstack([_make_treated(n_train_per_class, 0, 1, n_features), _make_treated(n_train_per_class, 1, 2, n_features)])
    treated_df = pd.DataFrame(X_train, columns=cols)
    treated_df["y"] = y_train

    rng = np.random.default_rng(0)
    control_df = pd.DataFrame(rng.normal(0, 3.0, (500, n_features)), columns=cols)

    augmented = control_difference_augment(treated_df, control_df, feature_cols=cols, n_augmented_per_treated=15, random_state=0)
    assert len(augmented) == 15 * len(treated_df)
    assert (augmented["y"] == pd.concat([treated_df["y"]] * 15, ignore_index=True)).all()
    combined = pd.concat([treated_df, augmented], ignore_index=True)

    n_test = 300
    X_test = np.vstack([_make_treated(n_test, 0, 100, n_features), _make_treated(n_test, 1, 101, n_features)])
    y_test = np.array([0] * n_test + [1] * n_test)

    model_small = RandomForestClassifier(n_estimators=200, random_state=0).fit(treated_df[cols], treated_df["y"])
    auc_small = roc_auc_score(y_test, model_small.predict_proba(X_test)[:, 1])

    model_augmented = RandomForestClassifier(n_estimators=200, random_state=0).fit(combined[cols], combined["y"])
    auc_augmented = roc_auc_score(y_test, model_augmented.predict_proba(X_test)[:, 1])

    assert auc_augmented > auc_small + 0.02, (
        f"control-difference augmentation should improve small-sample generalization: augmented={auc_augmented:.4f} small={auc_small:.4f}"
    )


def test_biz_val_control_difference_augment_multi_control_pairs_reduces_noise_variance():
    n_features = 30
    cols = [f"f{i}" for i in range(n_features)]
    n_treated = 200

    rng = np.random.default_rng(7)
    treated_df = pd.DataFrame(rng.normal(0, 1.0, (n_treated, n_features)), columns=cols)
    control_df = pd.DataFrame(rng.normal(0, 3.0, (500, n_features)), columns=cols)

    single = control_difference_augment(treated_df, control_df, feature_cols=cols, n_augmented_per_treated=1, random_state=1)
    multi = control_difference_augment(
        treated_df, control_df, feature_cols=cols, n_augmented_per_treated=1, random_state=1, n_control_pairs=20
    )

    # the augmentation noise is `augmented - treated`; averaging over more control pairs should pull it closer
    # to its true zero mean, i.e. shrink its per-row variance relative to the single-pair baseline.
    noise_single = (single[cols].to_numpy() - treated_df[cols].to_numpy()).var()
    noise_multi = (multi[cols].to_numpy() - treated_df[cols].to_numpy()).var()

    assert noise_multi < noise_single * 0.15, (
        f"n_control_pairs=20 should shrink augmentation-noise variance well below the single-pair baseline: "
        f"single={noise_single:.4f} multi={noise_multi:.4f}"
    )


def test_control_difference_augment_multi_control_pairs_default_is_bit_identical():
    n_features = 10
    cols = [f"f{i}" for i in range(n_features)]
    rng = np.random.default_rng(3)
    treated_df = pd.DataFrame(rng.normal(0, 1.0, (20, n_features)), columns=cols)
    control_df = pd.DataFrame(rng.normal(0, 1.0, (50, n_features)), columns=cols)

    baseline = control_difference_augment(treated_df, control_df, feature_cols=cols, n_augmented_per_treated=3, random_state=5)
    explicit_default = control_difference_augment(
        treated_df, control_df, feature_cols=cols, n_augmented_per_treated=3, random_state=5, n_control_pairs=1
    )
    pd.testing.assert_frame_equal(baseline, explicit_default)


def test_control_difference_augment_requires_at_least_two_control_rows():
    import pytest

    treated_df = pd.DataFrame({"f0": [1.0], "y": [1]})
    control_df = pd.DataFrame({"f0": [0.5]})
    with pytest.raises(ValueError):
        control_difference_augment(treated_df, control_df, feature_cols=["f0"])
