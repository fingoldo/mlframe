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


def _pure_noise_score_ceiling(*, with_gaps: bool, n_draws: int = 8) -> float:
    """The upper end of the empirical null: the highest score any independent pure-noise column reaches.

    A fixed ``[0.3, 0.7]`` band used to stand in for this. Measured over 20 draws, the real null spans
    0.431-0.575 without gaps and 0.450-0.575 with them, so a leak would have to inflate a noise column by
    more than 0.12 AUC before that band noticed -- and this file's own comment concedes a mild leak stays
    inside it. Deriving the ceiling from the same construction makes the bound roughly four times tighter
    and, unlike a literal, it moves with the fixture.

    One-sided on purpose: a leak lets a fold see its own test rows' statistics, which inflates the score.
    A noise column scoring LOW is just noise.
    """
    from mlframe.preprocessing.auto_transform_select import select_column_transforms

    highest = 0.0
    for draw in range(n_draws):
        rng = np.random.default_rng(9000 + draw)
        n = 400
        col = rng.normal(0, 1, n)
        if with_gaps:
            col[rng.random(n) < 0.15] = np.nan
        result = select_column_transforms(pd.DataFrame({"noise": col}), rng.integers(0, 2, n), task="classification", n_splits=4, random_state=0)
        highest = max(highest, max(result["noise"]["all_scores"].values()))
    return highest


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
    ceiling = _pure_noise_score_ceiling(with_gaps=False)
    assert ceiling < 0.7, f"the empirical null already reaches {ceiling:.3f}; a noise column is scoring high on its own and this test cannot separate that from a leak"
    for name, score in result["noise"]["all_scores"].items():
        assert score <= ceiling, f"a pure-noise column scored {score:.4f} under {name}, above the {ceiling:.4f} ceiling of {8} independent noise draws -- a CV leak inflated it"

    # Sensitivity: the score must actually respond to signal, or the bound above holds for a statistic
    # that is pinned near 0.5 whatever happens.
    rng2 = np.random.default_rng(11)
    signal = rng2.normal(0, 1, n)
    y_signal = (signal + rng2.normal(0, 0.3, n) > 0).astype(int)
    informative = select_column_transforms(pd.DataFrame({"signal": signal}), y_signal, task="classification", n_splits=4, random_state=0)
    assert max(informative["signal"]["all_scores"].values()) > ceiling, "a genuinely informative column did not out-score the pure-noise null; the score is not measuring anything"


def test_the_missing_value_fill_is_also_fold_local():
    """This file guarded the transform fit and never reached the IMPUTATION three lines above it.

    Not one fixture here carried a NaN or an inf, so ``select_column_transforms``'s non-finite branch was never
    executed -- and that branch computed its median over the WHOLE column, before the fold split, leaking every
    held-out fold's own values into its own imputation. A regression file for the leakage class that cannot
    reach the leak is the shape this test exists to close.
    """
    from mlframe.preprocessing.auto_transform_select import select_column_transforms

    rng = np.random.default_rng(3)
    n = 400
    noise = rng.normal(0, 1, n)
    # Gaps concentrated in the tail: a whole-column median differs sharply from any single fold's, so a leaked
    # fill is not merely biased but visibly so.
    noise[rng.random(n) < 0.15] = np.nan
    df = pd.DataFrame({"noise_with_gaps": noise})
    y = rng.integers(0, 2, n)

    result = select_column_transforms(df, y, task="classification", n_splits=4, random_state=0)
    assert result["noise_with_gaps"]["all_scores"], "the non-finite branch must still produce scores"
    ceiling = _pure_noise_score_ceiling(with_gaps=True)
    for name, score in result["noise_with_gaps"]["all_scores"].items():
        assert score <= ceiling, f"pure-noise column with gaps scored {score:.4f} under {name}, above the {ceiling:.4f} ceiling of 8 independent gapped noise draws -- the imputation is leaking"
