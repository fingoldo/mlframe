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
from sklearn.metrics import average_precision_score, roc_auc_score

from mlframe.models.masked_multilabel_objective import (
    compute_inverse_frequency_class_weights,
    flatten_masked_multilabel,
    flatten_masked_multilabel_class_weights,
    masked_multilabel_logloss_objective,
)


def _make_recommendation_data(n_rows: int, n_labels: int, n_features: int, seed: int):
    """Helper: Make recommendation data."""
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
    """Biz val masked objective beats naive negative coding."""
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

    assert (
        auc_masked > auc_naive + 0.05
    ), f"expected the masked objective to beat naive negative-coding by >=0.05 AUC on genuinely-undetermined cells, got masked={auc_masked:.4f} naive={auc_naive:.4f}"


def test_flatten_masked_multilabel_sentinel_marks_dont_care_cells():
    """Flatten masked multilabel sentinel marks dont care cells."""
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    mask = np.array([[True, False], [False, False]])
    flat = flatten_masked_multilabel(y, mask, sentinel=2.0)
    np.testing.assert_array_equal(flat, [2.0, 0.0, 0.0, 1.0])


def test_flatten_masked_multilabel_rejects_invalid_sentinel():
    """Flatten masked multilabel rejects invalid sentinel."""
    import pytest

    y = np.array([[1.0, 0.0]])
    mask = np.array([[False, False]])
    with pytest.raises(ValueError):
        flatten_masked_multilabel(y, mask, sentinel=1.0)


def test_masked_objective_zeroes_gradient_and_hessian_at_sentinel():
    """Masked objective zeroes gradient and hessian at sentinel."""
    class _FakeDMatrix:
        """Groups tests for: FakeDMatrix."""
        def __init__(self, labels: np.ndarray) -> None:
            """Helper: Init  ."""
            self._labels = labels

        def get_label(self) -> np.ndarray:
            """Get label."""
            return self._labels

    objective = masked_multilabel_logloss_objective(sentinel=2.0)
    y_true = np.array([0.0, 1.0, 2.0])  # last entry is don't-care.
    grad, hess = objective(np.array([0.0, 0.0, 0.0]), _FakeDMatrix(y_true))
    assert grad[2] == 0.0
    assert hess[2] == 0.0
    assert hess[0] > 0.0 and hess[1] > 0.0


def _make_imbalanced_multilabel_data(n_rows: int, n_features: int, seed: int):
    """4 label classes: one common (~40% positive), three genuinely rare (~2% positive)."""
    rng = np.random.default_rng(seed)
    n_labels = 4
    X = rng.normal(size=(n_rows, n_features))
    rates = np.array([0.4, 0.02, 0.02, 0.02])
    bias = np.log(rates / (1.0 - rates))
    coefs = np.array([1.0, 2.0, 2.0, 2.0])
    true_logit = X[:, :n_labels] * coefs + bias
    true_prob = 1.0 / (1.0 + np.exp(-true_logit))
    y = (rng.random((n_rows, n_labels)) < true_prob).astype(np.float64)

    # a modest, label-independent don't-care mask so the flattened weight path is exercised too.
    dont_care_mask = rng.random((n_rows, n_labels)) < 0.1

    X_flat = np.repeat(X, n_labels, axis=0)
    label_idx = np.tile(np.arange(n_labels), n_rows)
    X_flat_full = np.hstack([X_flat, label_idx.reshape(-1, 1)])
    return X_flat_full, y, dont_care_mask, n_labels


def test_biz_val_masked_objective_class_weighting_beats_uniform_on_rare_labels():
    # a single shared model over all flattened (row, label) cells: with max_depth=2 and few boosting rounds,
    # split/round budget is scarce, so which label's loss dominates each round's gain calculation matters --
    # the common class (40% positive, ~10x the other classes' sample-weighted gradient mass) crowds out the
    # rare classes' splits under uniform weighting. This is precisely the shared-capacity effect inverse-
    # frequency weighting is meant to counter.
    """Biz val masked objective class weighting beats uniform on rare labels."""
    n_rows, n_features = 5000, 8
    X_flat, y, dont_care_mask, n_labels = _make_imbalanced_multilabel_data(n_rows, n_features, seed=3)

    y_masked = flatten_masked_multilabel(y, dont_care_mask)
    split = int(0.8 * n_rows) * n_labels
    X_train, X_test = X_flat[:split], X_flat[split:]

    params = {"max_depth": 2, "eta": 0.1, "base_score": 0.5}

    # uniform-weight baseline: default (opt-out) behavior, no weight= on the DMatrix at all.
    dtrain_uniform = xgb.DMatrix(X_train, label=y_masked[:split])
    model_uniform = xgb.train(params, dtrain_uniform, num_boost_round=30, obj=masked_multilabel_logloss_objective())

    # opt-in inverse-frequency class weighting: upweights the rare label classes.
    class_weights = compute_inverse_frequency_class_weights(y[: n_rows * 4 // 5], dont_care_mask[: n_rows * 4 // 5])
    sample_weight = flatten_masked_multilabel_class_weights(y, dont_care_mask, class_weights=class_weights)
    dtrain_weighted = xgb.DMatrix(X_train, label=y_masked[:split], weight=sample_weight[:split])
    model_weighted = xgb.train(params, dtrain_weighted, num_boost_round=30, obj=masked_multilabel_logloss_objective(use_sample_weight=True))

    dtest = xgb.DMatrix(X_test)
    pred_uniform = 1.0 / (1.0 + np.exp(-model_uniform.predict(dtest, output_margin=True)))
    pred_weighted = 1.0 / (1.0 + np.exp(-model_weighted.predict(dtest, output_margin=True)))

    care_test = ~dont_care_mask.ravel()[split:]
    y_true_test = y.ravel()[split:]
    label_idx_test = np.tile(np.arange(n_labels), n_rows)[split:]

    rare_mask = care_test & np.isin(label_idx_test, [1, 2, 3])
    # rare-class positives are ~4% of cells; ROC-AUC is dominated by the easy majority of true negatives and
    # barely moves, while average precision (PR-AUC) is far more sensitive to ranking the few true positives
    # correctly -- the metric that actually matters for "did upweighting the rare class help find it".
    ap_uniform_rare = float(average_precision_score(y_true_test[rare_mask], pred_uniform[rare_mask]))
    ap_weighted_rare = float(average_precision_score(y_true_test[rare_mask], pred_weighted[rare_mask]))

    assert ap_weighted_rare > ap_uniform_rare + 0.04, (
        "expected inverse-frequency class weighting to beat uniform weighting by >=0.04 average precision on the "
        f"rare label classes, got weighted={ap_weighted_rare:.4f} uniform={ap_uniform_rare:.4f}"
    )


def test_masked_objective_use_sample_weight_default_off_is_bit_identical():
    """Omitting use_sample_weight must reproduce the pre-extension objective exactly (no DMatrix.get_weight call)."""

    class _FakeDMatrix:
        """Groups tests for: FakeDMatrix."""
        def __init__(self, labels: np.ndarray) -> None:
            """Helper: Init  ."""
            self._labels = labels

        def get_label(self) -> np.ndarray:
            """Get label."""
            return self._labels

        def get_weight(self) -> np.ndarray:
            """Get weight."""
            raise AssertionError("get_weight() must not be called when use_sample_weight is left at its default (False).")

    y_true = np.array([0.0, 1.0, 2.0])
    pred = np.array([0.1, -0.2, 0.3])
    grad_default, hess_default = masked_multilabel_logloss_objective()(pred, _FakeDMatrix(y_true))
    grad_explicit_off, hess_explicit_off = masked_multilabel_logloss_objective(use_sample_weight=False)(pred, _FakeDMatrix(y_true))
    np.testing.assert_array_equal(grad_default, grad_explicit_off)
    np.testing.assert_array_equal(hess_default, hess_explicit_off)


def test_masked_objective_use_sample_weight_scales_grad_and_hess():
    """Masked objective use sample weight scales grad and hess."""
    class _FakeDMatrix:
        """Groups tests for: FakeDMatrix."""
        def __init__(self, labels: np.ndarray, weight: np.ndarray) -> None:
            """Helper: Init  ."""
            self._labels = labels
            self._weight = weight

        def get_label(self) -> np.ndarray:
            """Get label."""
            return self._labels

        def get_weight(self) -> np.ndarray:
            """Get weight."""
            return self._weight

    y_true = np.array([0.0, 1.0, 2.0])
    weight = np.array([3.0, 5.0, 0.0])
    pred = np.array([0.1, -0.2, 0.3])

    grad_unweighted, hess_unweighted = masked_multilabel_logloss_objective()(pred, _FakeDMatrix(y_true, weight))
    grad_weighted, hess_weighted = masked_multilabel_logloss_objective(use_sample_weight=True)(pred, _FakeDMatrix(y_true, weight))

    np.testing.assert_allclose(grad_weighted, grad_unweighted * weight)
    np.testing.assert_allclose(hess_weighted, hess_unweighted * weight)


def test_masked_objective_use_sample_weight_rejects_missing_weight():
    """Masked objective use sample weight rejects missing weight."""
    import pytest

    class _FakeDMatrix:
        """Groups tests for: FakeDMatrix."""
        def __init__(self, labels: np.ndarray) -> None:
            """Helper: Init  ."""
            self._labels = labels

        def get_label(self) -> np.ndarray:
            """Get label."""
            return self._labels

        def get_weight(self) -> np.ndarray:
            """Get weight."""
            return np.array([])

    y_true = np.array([0.0, 1.0, 2.0])
    pred = np.array([0.1, -0.2, 0.3])
    with pytest.raises(ValueError):
        masked_multilabel_logloss_objective(use_sample_weight=True)(pred, _FakeDMatrix(y_true))


def test_flatten_masked_multilabel_class_weights_uniform_default():
    """Flatten masked multilabel class weights uniform default."""
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    mask = np.array([[False, False], [False, True]])
    weights = flatten_masked_multilabel_class_weights(y, mask)
    np.testing.assert_array_equal(weights, [1.0, 1.0, 1.0, 0.0])


def test_compute_inverse_frequency_class_weights_upweights_rare_class():
    """Compute inverse frequency class weights upweights rare class."""
    y = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    mask = np.zeros_like(y, dtype=bool)
    weights = compute_inverse_frequency_class_weights(y, mask)
    assert weights[1] > weights[0], "the class with fewer positives (col 1: 1/5) should get a larger weight than col 0 (3/5)"
