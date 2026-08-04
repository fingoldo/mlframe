"""Regression test: select_column_transforms's CV probe must fit every transform on the TRAIN
fold only, never on the full column (train+test) before splitting.

Pre-fix, ``_apply_transform`` was called once on the whole column and the resulting array was
then sliced by ``train_idx``/``test_idx`` inside the fold loop, so any transform with fit
statistics (all sklearn scalers, RankGauss) leaked the test fold's own values into the "held-out"
score for every candidate except identity/log1p_signed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mlframe.preprocessing.auto_transform_select import _fit_transform_fold


def test_fit_transform_fold_scaler_uses_train_only_statistics():
    """_fit_transform_fold must fit a scaler on the train slice only, not the whole column."""
    rng = np.random.default_rng(0)
    x = rng.normal(loc=10.0, scale=3.0, size=200)
    train_idx = np.arange(150)
    test_idx = np.arange(150, 200)

    got_train, got_test = _fit_transform_fold(x, "StandardScaler", train_idx, test_idx)

    # Reference: fit the SAME scaler on the train slice only, transform each slice separately.
    ref_scaler = StandardScaler()
    ref_scaler.fit(x[train_idx].reshape(-1, 1))
    ref_train = ref_scaler.transform(x[train_idx].reshape(-1, 1)).ravel()
    ref_test = ref_scaler.transform(x[test_idx].reshape(-1, 1)).ravel()

    np.testing.assert_allclose(got_train, ref_train)
    np.testing.assert_allclose(got_test, ref_test)

    # And explicitly NOT what a whole-column fit would have produced (the pre-fix leak): the
    # whole-column mean/std differ measurably from the train-only mean/std for this fixture.
    leaky_scaler = StandardScaler()
    leaky_scaler.fit(x.reshape(-1, 1))
    leaky_test = leaky_scaler.transform(x[test_idx].reshape(-1, 1)).ravel()
    assert not np.allclose(got_test, leaky_test)


def test_fit_transform_fold_rankgauss_uses_train_only_fit_values():
    """_fit_transform_fold's rankgauss branch must fit on train and replay onto test via apply_rankgauss."""
    rng = np.random.default_rng(1)
    x = rng.exponential(scale=5.0, size=200)
    train_idx = np.arange(150)
    test_idx = np.arange(150, 200)

    got_train, got_test = _fit_transform_fold(x, "rankgauss", train_idx, test_idx)
    assert got_train.shape == (150,)
    assert got_test.shape == (50,)
    assert np.all(np.isfinite(got_train)) and np.all(np.isfinite(got_test))

    # A test-fold value that is a NEW maximum (larger than every train value) must map near the
    # top Gaussian quantile of the TRAIN fit, not silently reuse a whole-column rank.
    x_probe = x.copy()
    x_probe[test_idx[0]] = x[train_idx].max() + 1000.0
    _, probe_test = _fit_transform_fold(x_probe, "rankgauss", train_idx, test_idx)
    assert probe_test[0] > 2.0  # clipped to the extreme rank -> a large positive Gaussian quantile


def test_select_column_transforms_scaler_scores_are_not_leaked():
    """End-to-end: a pure-noise column must not score artificially strong under any transform."""
    # End-to-end: on a column that is PURE NOISE (independent of y), no transform should look
    # artificially strong; a leaked fit would let a scaler's fold-fit boundary overlap the score
    # computation. This mainly guards against a future regression reintroducing the whole-column
    # fit (an exact leak-magnitude assertion would be fixture-fragile; a bounded max-AUC check is
    # a robust proxy for "no fold saw its own test rows' statistics").
    from mlframe.preprocessing.auto_transform_select import select_column_transforms

    rng = np.random.default_rng(2)
    n = 400
    df = pd.DataFrame({"noise": rng.normal(0, 1, n)})
    y = rng.integers(0, 2, n)

    result = select_column_transforms(df, y, task="classification", n_splits=4, random_state=0)
    for score in result["noise"]["all_scores"].values():
        assert 0.3 <= score <= 0.7, f"a pure-noise column scored {score}, suggesting a CV leak inflated it"
