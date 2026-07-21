"""biz_val for KNN-Shapley across mlframe's full target-type matrix (see ``docs/MULTI_OUTPUT.md`` /
``docs/MULTI_TARGET_REGRESSION.md``): multiclass classification (unchanged ``knn_shapley``), multilabel
classification and multi-target regression (the new per-column-averaged wrappers in
``_knn_shapley_multi_output.py``).

Same three-way train/val/test discipline as the other data_valuation biz_val tests: label/target noise
injected only into TRAIN, valuation computed against a separate VAL split, effect measured on a
held-out TEST split never touched by the valuation step.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_biz_val_multiclass_flags_mislabeled_rows():
    """knn_shapley (unchanged) on a 3-class problem: flipped-label rows flag via low value, same
    mechanism as the binary case -- proves the closed form genuinely generalizes to K>2 classes."""
    from mlframe.data_valuation import knn_shapley
    from sklearn.datasets import make_blobs
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    X, y = make_blobs(n_samples=1500, centers=3, cluster_std=1.5, random_state=0)
    y = y.astype(np.int64)
    n = len(y)
    flip_idx = rng.choice(n, size=int(n * 0.12), replace=False)
    y_noisy = y.copy()
    # Reassign to a DIFFERENT random class (not just the binary complement) -- the genuine multiclass mislabel case.
    other_classes = rng.integers(0, 2, size=len(flip_idx))
    y_noisy[flip_idx] = (y[flip_idx] + 1 + other_classes) % 3

    X_train, y_train_noisy = X[:1000], y_noisy[:1000]
    X_val, y_val = X[1000:], y[1000:]
    values = knn_shapley(X_train, y_train_noisy, X_val, y_val, k=5)

    flip_mask = np.zeros(1000, dtype=bool)
    flip_mask[flip_idx[flip_idx < 1000]] = True
    auroc = float(roc_auc_score(flip_mask.astype(int), -values))
    print(f"\nmulticlass noise-detection AUROC={auroc:.4f}")
    assert auroc >= 0.80, f"multiclass noise-detection AUROC {auroc:.4f} < 0.80"


def _multilabel_noisy_bed(n=2000, k_labels=3, corrupt_frac=0.12, seed=0):
    """K independent binary labels, each a linear-signal threshold; a fraction of TRAIN rows get ALL labels flipped."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    Y_true = np.zeros((n, k_labels), dtype=np.int64)
    for col in range(k_labels):
        w = rng.standard_normal(6)
        Y_true[:, col] = (X @ w + rng.standard_normal(n) * 0.5 > 0).astype(np.int64)
    corrupt_idx = rng.choice(n, size=int(n * corrupt_frac), replace=False)
    Y_noisy = Y_true.copy()
    Y_noisy[corrupt_idx] = 1 - Y_noisy[corrupt_idx]
    return X, Y_true, Y_noisy, corrupt_idx


def test_biz_val_multilabel_flags_mislabeled_rows():
    """knn_shapley_multilabel: rows with ALL K labels flipped get a low averaged value vs clean rows."""
    from mlframe.data_valuation import knn_shapley_multilabel
    from sklearn.metrics import roc_auc_score

    n = 2000
    X, Y_true, Y_noisy, corrupt_idx = _multilabel_noisy_bed(n=n)
    idx = np.arange(n)
    rng = np.random.default_rng(1)
    rng.shuffle(idx)
    n_train, n_val = int(n * 0.6), int(n * 0.2)
    idx_train, idx_val = idx[:n_train], idx[n_train : n_train + n_val]

    X_train, Y_train_noisy = X[idx_train], Y_noisy[idx_train]
    X_val, Y_val_true = X[idx_val], Y_true[idx_val]

    values, info = knn_shapley_multilabel(X_train, Y_train_noisy, X_val, Y_val_true, k=5)
    assert info["n_labels"] == 3
    assert info["per_label_values"].shape == (3, len(idx_train))

    corrupt_mask_train = np.isin(idx_train, corrupt_idx)
    assert int(corrupt_mask_train.sum()) >= 20, "too few corrupted rows landed in train for a meaningful measurement"
    auroc = float(roc_auc_score(corrupt_mask_train.astype(int), -values))
    print(f"\nmultilabel noise-detection AUROC={auroc:.4f}")
    assert auroc >= 0.75, f"multilabel noise-detection AUROC {auroc:.4f} < 0.75"


def _multi_target_regression_bed(n=2000, k_targets=3, corrupt_frac=0.12, seed=0):
    """K independent continuous targets; a fraction of TRAIN rows get ALL target columns reassigned to another row's true values."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    Y_true = np.zeros((n, k_targets), dtype=np.float64)
    for col in range(k_targets):
        w = rng.standard_normal(6)
        Y_true[:, col] = X @ w + rng.standard_normal(n) * 0.5
    corrupt_idx = rng.choice(n, size=int(n * corrupt_frac), replace=False)
    shuffle_source = rng.permutation(corrupt_idx)
    Y_noisy = Y_true.copy()
    Y_noisy[corrupt_idx] = Y_true[shuffle_source]
    return X, Y_true, Y_noisy, corrupt_idx


def test_biz_val_multi_target_regression_flags_mislabeled_rows():
    """knn_shapley_multi_target_regression: rows with ALL K target columns reassigned get a low
    averaged value vs clean rows."""
    from mlframe.data_valuation import knn_shapley_multi_target_regression
    from sklearn.metrics import roc_auc_score

    n = 2000
    X, Y_true, Y_noisy, corrupt_idx = _multi_target_regression_bed(n=n)
    idx = np.arange(n)
    rng = np.random.default_rng(2)
    rng.shuffle(idx)
    n_train, n_val = int(n * 0.6), int(n * 0.2)
    idx_train, idx_val = idx[:n_train], idx[n_train : n_train + n_val]

    X_train, Y_train_noisy = X[idx_train], Y_noisy[idx_train]
    X_val, Y_val_true = X[idx_val], Y_true[idx_val]

    values, info = knn_shapley_multi_target_regression(X_train, Y_train_noisy, X_val, Y_val_true, k=5)
    assert info["n_targets"] == 3
    assert info["per_target_values"].shape == (3, len(idx_train))
    assert info["thresholds"].shape == (3,)

    corrupt_mask_train = np.isin(idx_train, corrupt_idx)
    assert int(corrupt_mask_train.sum()) >= 20, "too few corrupted rows landed in train for a meaningful measurement"
    auroc = float(roc_auc_score(corrupt_mask_train.astype(int), -values))
    print(f"\nmulti-target-regression noise-detection AUROC={auroc:.4f}")
    assert auroc >= 0.60, f"multi-target-regression noise-detection AUROC {auroc:.4f} < 0.60"


def test_knn_shapley_multilabel_rejects_mismatched_label_count():
    """A Y_val with a different number of label columns than Y_train raises ValueError, not a silent shape bug."""
    from mlframe.data_valuation import knn_shapley_multilabel

    rng = np.random.default_rng(3)
    X_train = rng.standard_normal((50, 4))
    Y_train = rng.integers(0, 2, size=(50, 3))
    X_val = rng.standard_normal((20, 4))
    Y_val = rng.integers(0, 2, size=(20, 2))
    with pytest.raises(ValueError):
        knn_shapley_multilabel(X_train, Y_train, X_val, Y_val)


def test_knn_shapley_multi_target_regression_rejects_mismatched_target_count():
    """A Y_val with a different number of target columns than Y_train raises ValueError, not a silent shape bug."""
    from mlframe.data_valuation import knn_shapley_multi_target_regression

    rng = np.random.default_rng(4)
    X_train = rng.standard_normal((50, 4))
    Y_train = rng.standard_normal((50, 3))
    X_val = rng.standard_normal((20, 4))
    Y_val = rng.standard_normal((20, 2))
    with pytest.raises(ValueError):
        knn_shapley_multi_target_regression(X_train, Y_train, X_val, Y_val)
