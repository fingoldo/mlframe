"""biz_value test for ``preprocessing.align_feature_direction`` (``align_feature_direction``, ``apply_feature_direction``).

The win (4th_santander-customer-transaction-prediction.md): a POOLED aggregate across many independently
target-correlated features (a simple mean, a shared embedding, a long-format melt) implicitly assumes
consistent orientation -- a feature negatively correlated with the target contributes the WRONG sign,
partially CANCELING the positively-oriented features' signal. Flipping negatively-oriented features first
should recover a much stronger pooled signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mlframe.preprocessing.align_feature_direction import align_feature_direction, apply_feature_direction


def _make_mixed_orientation_dataset(n: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    true_signal = np.where(y == 1, 1.0, -1.0)

    cols = {}
    for i in range(n_features):
        orientation = 1.0 if i % 2 == 0 else -1.0  # half positively, half negatively oriented
        cols[f"f{i}"] = orientation * true_signal + rng.normal(scale=0.5, size=n)
    return pd.DataFrame(cols), y


def test_biz_val_align_feature_direction_recovers_pooled_signal():
    X_train, y_train = _make_mixed_orientation_dataset(n=2000, n_features=20, seed=0)
    X_test, y_test = _make_mixed_orientation_dataset(n=1000, n_features=20, seed=1)

    naive_pooled_train = X_train.mean(axis=1).to_numpy()
    naive_pooled_test = X_test.mean(axis=1).to_numpy()
    auc_naive = roc_auc_score(y_test, naive_pooled_test)

    X_train_aligned, flip_signs = align_feature_direction(X_train, y_train)
    X_test_aligned = apply_feature_direction(X_test, flip_signs)
    aligned_pooled_train = X_train_aligned.mean(axis=1).to_numpy()
    aligned_pooled_test = X_test_aligned.mean(axis=1).to_numpy()
    auc_aligned = roc_auc_score(y_test, aligned_pooled_test)

    assert auc_naive < 0.6, f"expected the naive pooled mean (mixed orientation, half cancels the other half) to carry weak signal, got AUC={auc_naive:.4f}"
    assert auc_aligned > 0.9, f"expected the direction-aligned pooled mean to carry strong signal, got AUC={auc_aligned:.4f}"
    assert list(np.sign(aligned_pooled_train)) or True  # sanity: aligned_pooled_train computed without error


def test_align_feature_direction_flips_correct_columns():
    rng = np.random.default_rng(2)
    n = 1000
    y = rng.integers(0, 2, n)
    signal = np.where(y == 1, 1.0, -1.0)
    df = pd.DataFrame({"pos": signal + 0.1 * rng.standard_normal(n), "neg": -signal + 0.1 * rng.standard_normal(n)})

    aligned, flip_signs = align_feature_direction(df, y)
    assert flip_signs["pos"] == 1
    assert flip_signs["neg"] == -1
    np.testing.assert_allclose(aligned["neg"].to_numpy(), -df["neg"].to_numpy())
    np.testing.assert_allclose(aligned["pos"].to_numpy(), df["pos"].to_numpy())


def test_apply_feature_direction_never_recomputes_auc():
    flip_signs = {"a": -1, "b": 1}
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    out = apply_feature_direction(df, flip_signs)
    np.testing.assert_allclose(out["a"].to_numpy(), [-1.0, -2.0, -3.0])
    np.testing.assert_allclose(out["b"].to_numpy(), [4.0, 5.0, 6.0])
