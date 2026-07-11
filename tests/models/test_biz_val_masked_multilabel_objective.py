"""biz_value test for ``models.masked_multilabel_objective``.

Source: 4th_santander-product-recommendation.md -- flattened row*label labels with a "don't care" sentinel
for already-owned products, avoiding the row-explosion group-ranking API. When ownership correlates with true
propensity (a realistic confound: customers likely to want a product often already own it), naively coding
"already owned" as a negative label systematically teaches the model that propensity-correlated features
predict NON-acquisition -- exactly backwards. Masking those cells out of the loss entirely should recover
materially better held-out discrimination on the genuinely-undetermined (not-yet-owned) cells.
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from mlframe.models.masked_multilabel_objective import flatten_masked_multilabel, masked_multilabel_logloss_objective


def _make_recommendation_data(n_rows: int, n_labels: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features))
    true_logit = X[:, :n_labels] * 1.5
    true_prob = 1.0 / (1.0 + np.exp(-true_logit))
    y_new_acquire = (rng.random((n_rows, n_labels)) < true_prob).astype(np.float64)

    # ownership correlates with propensity: high-propensity customers are likely to already own the product.
    own_prob = np.clip(true_prob * 0.8, 0, 0.9)
    already_owned = rng.random((n_rows, n_labels)) < own_prob

    X_flat = np.repeat(X, n_labels, axis=0)
    label_idx = np.tile(np.arange(n_labels), n_rows)
    X_flat_full = np.hstack([X_flat, label_idx.reshape(-1, 1)])
    return X_flat_full, y_new_acquire, already_owned


def test_biz_val_masked_objective_beats_naive_negative_coding():
    n_rows, n_labels, n_features = 3000, 5, 8
    X_flat, y_new_acquire, already_owned = _make_recommendation_data(n_rows, n_labels, n_features, seed=1)

    y_masked = flatten_masked_multilabel(y_new_acquire, already_owned)
    y_naive = y_new_acquire.copy()
    y_naive[already_owned] = 0.0  # the naive (wrong) convention: code "already owned" as a negative.
    y_naive_flat = y_naive.ravel()

    split = int(0.8 * n_rows) * n_labels
    X_train, X_test = X_flat[:split], X_flat[split:]

    params = {"max_depth": 4, "eta": 0.1, "base_score": 0.5}
    dtrain_masked = xgb.DMatrix(X_train, label=y_masked[:split])
    model_masked = xgb.train(params, dtrain_masked, num_boost_round=100, obj=masked_multilabel_logloss_objective())

    dtrain_naive = xgb.DMatrix(X_train, label=y_naive_flat[:split])
    model_naive = xgb.train({**params, "objective": "binary:logistic"}, dtrain_naive, num_boost_round=100)

    dtest = xgb.DMatrix(X_test)
    pred_masked = 1.0 / (1.0 + np.exp(-model_masked.predict(dtest, output_margin=True)))
    pred_naive = model_naive.predict(dtest)

    test_not_owned = ~already_owned.ravel()[split:]
    y_true_test = y_new_acquire.ravel()[split:]

    auc_masked = float(roc_auc_score(y_true_test[test_not_owned], pred_masked[test_not_owned]))
    auc_naive = float(roc_auc_score(y_true_test[test_not_owned], pred_naive[test_not_owned]))

    assert auc_masked > auc_naive + 0.05, f"expected the masked objective to beat naive negative-coding by >=0.05 AUC on genuinely-undetermined cells, got masked={auc_masked:.4f} naive={auc_naive:.4f}"


def test_flatten_masked_multilabel_sentinel_marks_dont_care_cells():
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    mask = np.array([[True, False], [False, False]])
    flat = flatten_masked_multilabel(y, mask, sentinel=2.0)
    np.testing.assert_array_equal(flat, [2.0, 0.0, 0.0, 1.0])


def test_flatten_masked_multilabel_rejects_invalid_sentinel():
    import pytest

    y = np.array([[1.0, 0.0]])
    mask = np.array([[False, False]])
    with pytest.raises(ValueError):
        flatten_masked_multilabel(y, mask, sentinel=1.0)


def test_masked_objective_zeroes_gradient_and_hessian_at_sentinel():
    class _FakeDMatrix:
        def __init__(self, labels: np.ndarray) -> None:
            self._labels = labels

        def get_label(self) -> np.ndarray:
            return self._labels

    objective = masked_multilabel_logloss_objective(sentinel=2.0)
    y_true = np.array([0.0, 1.0, 2.0])  # last entry is don't-care.
    grad, hess = objective(np.array([0.0, 0.0, 0.0]), _FakeDMatrix(y_true))
    assert grad[2] == 0.0
    assert hess[2] == 0.0
    assert hess[0] > 0.0 and hess[1] > 0.0
