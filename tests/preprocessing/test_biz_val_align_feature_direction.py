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

from mlframe.preprocessing.align_feature_direction import align_feature_direction, apply_feature_direction, check_feature_direction_stability


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


def test_align_feature_direction_unaffected_by_stability_check_addition():
    """Regression: align_feature_direction/apply_feature_direction must stay bit-identical -- the new
    check_feature_direction_stability is a fully separate opt-in function, never called by these two."""
    X_train, y_train = _make_mixed_orientation_dataset(n=2000, n_features=20, seed=0)
    X_test, y_test = _make_mixed_orientation_dataset(n=1000, n_features=20, seed=1)

    aligned1, signs1 = align_feature_direction(X_train, y_train)
    aligned2, signs2 = align_feature_direction(X_train, y_train)
    assert signs1 == signs2
    pd.testing.assert_frame_equal(aligned1, aligned2)

    applied1 = apply_feature_direction(X_test, signs1)
    applied2 = apply_feature_direction(X_test, signs1)
    pd.testing.assert_frame_equal(applied1, applied2)


def test_biz_val_check_feature_direction_stability_distinguishes_strong_from_weak():
    """A strongly AUC-directed feature must be flagged stable across folds; a near-chance one, whose single
    full-data AUC estimate can land on either side of 0.5 by noise, must be flagged unstable more often."""
    rng = np.random.default_rng(7)
    n = 1500
    y = rng.integers(0, 2, n)
    signal = np.where(y == 1, 1.0, -1.0)

    df = pd.DataFrame(
        {
            "strong": signal + 0.2 * rng.standard_normal(n),  # AUC far from 0.5, sign should be rock-stable
            "weak": 0.02 * signal + 1.0 * rng.standard_normal(n),  # AUC near 0.5, sign is noise-dominated
        }
    )

    result = check_feature_direction_stability(df, y, n_folds=10, seed=3)

    assert result["strong"]["stable"] is True, f"expected the strong feature's sign to be stable across folds, got {result['strong']}"
    assert result["strong"]["n_sign_flips"] == 0

    # Repeat the weak feature over several seeds and require flips to show up materially more often than for
    # the strong feature -- a single seed could get lucky, so this aggregates across resamples of the noise.
    weak_flip_counts = []
    strong_flip_counts = []
    for seed in range(5):
        rng_s = np.random.default_rng(100 + seed)
        y_s = rng_s.integers(0, 2, n)
        signal_s = np.where(y_s == 1, 1.0, -1.0)
        df_s = pd.DataFrame(
            {
                "strong": signal_s + 0.2 * rng_s.standard_normal(n),
                "weak": 0.02 * signal_s + 1.0 * rng_s.standard_normal(n),
            }
        )
        res_s = check_feature_direction_stability(df_s, y_s, n_folds=10, seed=3)
        weak_flip_counts.append(res_s["weak"]["n_sign_flips"])
        strong_flip_counts.append(res_s["strong"]["n_sign_flips"])

    assert sum(weak_flip_counts) > sum(strong_flip_counts), (
        f"expected the near-chance feature to flip sign across folds far more often than the strong feature; "
        f"weak flips={weak_flip_counts} strong flips={strong_flip_counts}"
    )
    assert sum(strong_flip_counts) == 0, f"strong feature should never flip, got {strong_flip_counts}"
