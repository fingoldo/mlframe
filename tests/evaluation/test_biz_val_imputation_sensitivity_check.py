"""biz_value test for ``evaluation.imputation_sensitivity_check.imputation_sensitivity_check``.

Source: 1st_favorita-grocery-sales-forecasting.md -- CPMP's observation that how different teams filled a
missing "onpromotion" flag correlated with their public/private leaderboard gap. On a synthetic where the
MEANING of missingness itself changes over time (a regime shift: early missing values truly meant "off", late
missing values truly meant "on"), zero-filling gets the early regime right and the late regime badly wrong --
producing wildly unstable CV scores across time-ordered folds -- while filling with the column's own mean
stays consistently (if imperfectly) neutral across both regimes, giving much more stable fold-to-fold scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from mlframe.evaluation.imputation_sensitivity_check import imputation_sensitivity_check


def _make_regime_shift_missingness_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    promo_true = rng.integers(0, 2, n).astype(np.float64)

    missing_mask = rng.random(n) < 0.3
    promo_true[missing_mask & (t < n // 2)] = 0.0  # early missing truly meant "off".
    promo_true[missing_mask & (t >= n // 2)] = 1.0  # late missing truly meant "on" -- the meaning flipped.

    y = 5.0 * promo_true + rng.normal(scale=1.0, size=n) + 0.01 * t

    promo_observed = promo_true.copy()
    promo_observed[missing_mask] = np.nan

    X_zero = pd.DataFrame({"promo": np.nan_to_num(promo_observed, nan=0.0), "t": t.astype(np.float64)})
    X_mean = pd.DataFrame({"promo": pd.Series(promo_observed).fillna(pd.Series(promo_observed).mean()), "t": t.astype(np.float64)})
    X_mode = pd.DataFrame({"promo": pd.Series(promo_observed).fillna(pd.Series(promo_observed).mode()[0]), "t": t.astype(np.float64)})
    return X_zero, X_mean, X_mode, y


def test_biz_val_flags_regime_sensitive_imputation_as_risky():
    X_zero, X_mean, X_mode, y = _make_regime_shift_missingness_data(n=1000, seed=0)
    cv = KFold(n_splits=5, shuffle=False)  # time-ordered blocked folds -- matches the source's own framing.

    result = imputation_sensitivity_check(Ridge(alpha=0.1), {"zero_fill": X_zero, "mean_fill": X_mean, "mode_fill": X_mode}, y, r2_score, cv=cv)

    assert result.loc["zero_fill", "is_risky"], "expected the regime-sensitive zero-fill to be flagged risky"
    assert not result.loc["mean_fill", "is_risky"], "expected the regime-robust mean-fill to NOT be flagged risky"
    assert result.loc["zero_fill", "fold_std"] > result.loc["mean_fill", "fold_std"] * 3, f"expected zero-fill's fold-to-fold variance to be much higher than mean-fill's, got zero={result.loc['zero_fill', 'fold_std']:.4f} mean={result.loc['mean_fill', 'fold_std']:.4f}"


def test_imputation_sensitivity_check_sorted_riskiest_first():
    X_zero, X_mean, X_mode, y = _make_regime_shift_missingness_data(n=800, seed=1)
    cv = KFold(n_splits=5, shuffle=False)
    result = imputation_sensitivity_check(Ridge(alpha=0.1), {"mean_fill": X_mean, "zero_fill": X_zero}, y, r2_score, cv=cv)
    assert list(result.index)[0] == "zero_fill"  # highest fold_std sorted first.


def _make_shift_blind_spot_data(n: int, seed: int):
    """Regime-shift missingness (same generative mechanism as above), but with a SHUFFLED CV splitter every
    fold is a representative blend of both regimes, so a fill choice whose error is merely consistent across
    folds (same average bias each time) reads as "stable" -- the fold-to-fold-only check's blind spot. A real
    chronological holdout (train on the first half only, evaluate on the second half only) is not fooled: it
    exposes the systematic error zero-fill makes on rows whose true missingness meaning it never saw in training.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    promo_true = rng.integers(0, 2, n).astype(np.float64)
    missing_mask = rng.random(n) < 0.3
    promo_true[missing_mask & (t < n // 2)] = 0.0
    promo_true[missing_mask & (t >= n // 2)] = 1.0
    y = 5.0 * promo_true + rng.normal(scale=1.0, size=n)

    promo_observed = promo_true.copy()
    promo_observed[missing_mask] = np.nan

    X_zero = pd.DataFrame({"promo": np.nan_to_num(promo_observed, nan=0.0)})
    X_mean = pd.DataFrame({"promo": pd.Series(promo_observed).fillna(pd.Series(promo_observed).mean())})
    X_mode = pd.DataFrame({"promo": pd.Series(promo_observed).fillna(pd.Series(promo_observed).mode()[0])})
    return X_zero, X_mean, X_mode, y


def test_biz_val_shift_split_catches_instability_fold_cv_misses():
    n = 2000
    X_zero, X_mean, X_mode, y = _make_shift_blind_spot_data(n=n, seed=0)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)  # shuffled -- mixes regimes within every fold.
    shift_split = (np.arange(0, n // 2), np.arange(n // 2, n))  # train on the past, evaluate on the future.

    variants = {"zero_fill": X_zero, "mean_fill": X_mean, "mode_fill": X_mode}
    fold_only = imputation_sensitivity_check(Ridge(alpha=0.1), variants, y, r2_score, cv=cv)
    shift_aware = imputation_sensitivity_check(Ridge(alpha=0.1), variants, y, r2_score, cv=cv, shift_split=shift_split)

    # the fold-to-fold-only check misses zero_fill's real instability under shuffled CV.
    assert not fold_only.loc["zero_fill", "is_risky"], "expected shuffled fold-to-fold CV to miss zero_fill's instability (the blind spot under test)"

    # the shift-aware check catches it via a large train-past/test-future score gap.
    assert shift_aware.loc["zero_fill", "is_shift_risky"], "expected the shift-aware check to flag zero_fill as risky"
    assert not shift_aware.loc["mean_fill", "is_shift_risky"], "expected the regime-robust mean-fill to NOT be flagged shift-risky"
    assert shift_aware.loc["zero_fill", "shift_gap"] > 0.5, f"expected zero_fill's shift_gap to clear 0.5, got {shift_aware.loc['zero_fill', 'shift_gap']:.4f}"
    assert shift_aware.loc["zero_fill", "shift_gap"] > shift_aware.loc["mean_fill", "shift_gap"] * 2, "expected zero_fill's shift_gap to be much larger than mean_fill's"


def test_imputation_sensitivity_check_shift_split_omitted_is_bit_identical():
    X_zero, X_mean, X_mode, y = _make_regime_shift_missingness_data(n=500, seed=2)
    cv = KFold(n_splits=5, shuffle=False)
    variants = {"zero_fill": X_zero, "mean_fill": X_mean, "mode_fill": X_mode}

    baseline = imputation_sensitivity_check(Ridge(alpha=0.1), variants, y, r2_score, cv=cv)
    with_shift_omitted = imputation_sensitivity_check(Ridge(alpha=0.1), variants, y, r2_score, cv=cv, shift_split=None)

    pd.testing.assert_frame_equal(baseline, with_shift_omitted)
