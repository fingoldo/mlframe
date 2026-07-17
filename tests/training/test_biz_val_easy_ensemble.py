"""biz_value test for ``training.easy_ensemble_fit_predict``.

The win: under extreme class imbalance, EasyEnsemble bagging over independently-undersampled negative
subsets should beat a single model trained once on the full imbalanced data, since the single model's split
criterion is dominated by the majority class while every bag in the ensemble sees a balanced problem.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from mlframe.training._easy_ensemble import easy_ensemble_fit_predict


def _make_imbalanced_data(seed: int):
    rng = np.random.default_rng(seed)
    n_pos, n_neg = 60, 6000
    pos_x = rng.normal(1.5, 1.0, size=(n_pos, 3))
    neg_x = rng.normal(-0.3, 1.0, size=(n_neg, 3))
    X = np.vstack([pos_x, neg_x])
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    split = int(0.7 * len(y))
    return X[:split], y[:split], X[split:], y[split:]


def test_biz_val_easy_ensemble_beats_single_model_on_extreme_imbalance():
    X_train, y_train, X_test, y_test = _make_imbalanced_data(seed=0)

    result = easy_ensemble_fit_predict(
        X_train, y_train, X_test, model_factory=lambda: LogisticRegression(max_iter=300), n_bags=10, negative_ratio=1.0, random_state=0
    )
    auc_ensemble = roc_auc_score(y_test, result["test_pred"])

    single_model = LogisticRegression(max_iter=300)
    single_model.fit(X_train, y_train)
    auc_single = roc_auc_score(y_test, single_model.predict_proba(X_test)[:, 1])

    assert auc_ensemble > auc_single, f"EasyEnsemble should beat a single model on extreme imbalance: ensemble={auc_ensemble:.4f} single={auc_single:.4f}"
    assert len(result["bag_preds"]) == 10
    assert len(result["models"]) == 10


def test_easy_ensemble_missing_class_raises():
    import pytest

    X = np.zeros((10, 2))
    y = np.zeros(10)
    with pytest.raises(ValueError):
        easy_ensemble_fit_predict(X, y, X, model_factory=lambda: LogisticRegression())


def test_easy_ensemble_negative_ratio_controls_bag_size():
    rng = np.random.default_rng(1)
    n_pos, n_neg = 20, 2000
    X = np.vstack([rng.normal(1, 1, (n_pos, 2)), rng.normal(-1, 1, (n_neg, 2))])
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    result = easy_ensemble_fit_predict(X, y, X[:5], model_factory=lambda: LogisticRegression(), n_bags=3, negative_ratio=2.0, random_state=1)
    assert len(result["test_pred"]) == 5


def test_easy_ensemble_bag_feature_subsample_omitted_is_bit_identical():
    """Default (bag_feature_subsample=None) must reproduce the exact prior full-feature-set behaviour."""
    X_train, y_train, X_test, _ = _make_imbalanced_data(seed=0)

    baseline = easy_ensemble_fit_predict(
        X_train, y_train, X_test, model_factory=lambda: LogisticRegression(max_iter=300), n_bags=5, negative_ratio=1.0, random_state=0
    )
    explicit_none = easy_ensemble_fit_predict(
        X_train,
        y_train,
        X_test,
        model_factory=lambda: LogisticRegression(max_iter=300),
        n_bags=5,
        negative_ratio=1.0,
        random_state=0,
        bag_feature_subsample=None,
    )
    np.testing.assert_array_equal(baseline["test_pred"], explicit_none["test_pred"])
    assert baseline["bag_feature_idx"] is None
    assert explicit_none["bag_feature_idx"] is None


def _make_imbalanced_noisy_data(seed: int, n_informative: int = 10, n_noise: int = 50):
    """Extreme imbalance plus many weak/noisy features, so feature-bagging has something to decorrelate."""
    rng = np.random.default_rng(seed)
    n_pos, n_neg = 50, 5000
    pos_informative = rng.normal(1.5, 1.0, size=(n_pos, n_informative))
    neg_informative = rng.normal(-0.3, 1.0, size=(n_neg, n_informative))
    X_informative = np.vstack([pos_informative, neg_informative])
    n_rows = n_pos + n_neg
    X_noise = rng.normal(0.0, 1.0, size=(n_rows, n_noise))
    X = np.hstack([X_informative, X_noise])
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    idx = rng.permutation(n_rows)
    X, y = X[idx], y[idx]
    split = int(0.7 * n_rows)
    return X[:split], y[:split], X[split:], y[split:]


def test_biz_val_easy_ensemble_bag_feature_subsample_beats_negative_only_on_noisy_features():
    """Combining feature-bagging with negative-undersampling should cut ensemble variance and lift held-out AUC
    versus negative-undersampling alone, when most features are pure noise diluting each bag's split criterion."""
    aucs_negative_only = []
    aucs_combined = []
    for seed in range(5):
        X_train, y_train, X_test, y_test = _make_imbalanced_noisy_data(seed=seed)

        negative_only = easy_ensemble_fit_predict(
            X_train,
            y_train,
            X_test,
            model_factory=lambda: DecisionTreeClassifier(max_depth=4, random_state=0),
            n_bags=15,
            negative_ratio=1.0,
            random_state=seed,
        )
        aucs_negative_only.append(roc_auc_score(y_test, negative_only["test_pred"]))

        combined = easy_ensemble_fit_predict(
            X_train,
            y_train,
            X_test,
            model_factory=lambda: DecisionTreeClassifier(max_depth=4, random_state=0),
            n_bags=15,
            negative_ratio=1.0,
            random_state=seed,
            bag_feature_subsample=0.4,
        )
        aucs_combined.append(roc_auc_score(y_test, combined["test_pred"]))

    mean_negative_only = float(np.mean(aucs_negative_only))
    mean_combined = float(np.mean(aucs_combined))

    assert mean_combined > mean_negative_only, (
        f"row+feature bagging should beat row-only bagging on noisy-feature extreme imbalance: "
        f"combined={mean_combined:.4f} negative_only={mean_negative_only:.4f}"
    )
    assert mean_combined >= 0.80, f"combined bagging AUC too low: {mean_combined:.4f}"
