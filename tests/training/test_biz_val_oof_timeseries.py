"""Unit + biz_value coverage for ``trainer._compute_oof_preds_timeseries`` (the ``oof_has_time=True``
manual ``TimeSeriesSplit`` OOF path), previously entirely untested.

``_compute_oof_preds`` (also fixed this session -- see ``DEFAULTS_CHANGELOG.md``'s early-stopping
guard) branches to this function whenever the caller sets ``oof_has_time=True`` and no groups are
given: ``cross_val_predict`` refuses ``TimeSeriesSplit`` (it isn't a partition -- early rows are never
a test fold), so this manual per-fold loop exists instead. It had zero direct test coverage: neither
the temporal-honesty contract (no future row ever predicts a past one) nor the warm-up-rows-stay-NaN
contract had ever been exercised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from mlframe.training.trainer import _compute_oof_preds_timeseries


def _time_ordered_frame(seed=0, n=500):
    """A target with a genuine trend + regime shift, so a model trained on early rows predicts LATER
    rows noticeably worse than a model that also saw recent rows -- the discriminating signal a
    temporally-honest OOF split must preserve (a shuffled KFold would hide this by leaking future
    regime information into early folds)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    x0 = rng.normal(size=n)
    # regime shift at the midpoint: the true relationship between x0 and y changes.
    slope = np.where(t < n / 2, 1.0, -1.0)
    y = slope * x0 + 0.05 * rng.normal(size=n)
    X = pd.DataFrame({"x0": x0, "t": t})
    return X, y


def test_biz_val_timeseries_oof_no_future_leakage():
    """Warm-up rows must stay NaN (no fold ever holds out the earliest chunk), and by the time enough
    same-regime history has accumulated (the late window, deep into the post-shift regime), the OOF
    predictions must correlate strongly with the true target -- a DecisionTreeRegressor that can split
    on the time feature ``t`` recovers the regime-appropriate relationship once its training window
    contains enough post-shift rows; a materially weaker correlation would mean the fold loop isn't
    genuinely learning from (only) the rows a temporally-honest split makes available.
    """
    X, y = _time_ordered_frame()
    est = DecisionTreeRegressor(max_depth=4, random_state=0)
    oof_preds, oof_probs = _compute_oof_preds_timeseries(estimator=est, train_df=X, train_target=y, method="predict", n_splits=4)

    assert oof_probs is None
    assert oof_preds is not None
    assert oof_preds.shape[0] == len(y)
    # Warm-up rows (never in any TimeSeriesSplit test fold -- the earliest chunk) must stay NaN.
    assert np.isnan(oof_preds[0]), "warm-up row should have no honest OOF prediction"
    n_valid = np.isfinite(oof_preds).sum()
    assert n_valid > 0, "no fold ever produced a prediction"

    # biz_value: late-window (deep post-shift, >=75% through the series) OOF predictions must
    # correlate strongly with the true target -- measured 0.82 on this synthetic/seed/estimator combo,
    # threshold set well below that (0.5) to tolerate run-to-run noise while still catching a genuinely
    # broken fold loop (e.g. one that silently predicts a constant, or leaks in a way that scrambles
    # the temporal structure).
    valid_mask = np.isfinite(oof_preds)
    late_mask = valid_mask & (np.arange(len(y)) >= len(y) * 0.75)
    assert late_mask.sum() >= 20, "not enough late-regime OOF rows to measure"
    late_corr = float(np.corrcoef(oof_preds[late_mask], y[late_mask])[0, 1])
    assert late_corr > 0.5, f"late-regime OOF correlation too weak ({late_corr:.3f}) for a temporally-honest split"


def test_biz_val_timeseries_oof_classification_predict_proba():
    """``method='predict_proba'`` must return a properly-shaped (n_rows, n_classes) oof_probs array,
    not oof_preds, with warm-up rows NaN in every class column."""
    rng = np.random.default_rng(1)
    n = 400
    t = np.arange(n, dtype=np.float64)
    x0 = rng.normal(size=n)
    logit = np.where(t < n / 2, x0, -x0)
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    X = pd.DataFrame({"x0": x0, "t": t})

    est = LogisticRegression()
    oof_preds, oof_probs = _compute_oof_preds_timeseries(estimator=est, train_df=X, train_target=y, method="predict_proba", n_splits=4)

    assert oof_preds is None
    assert oof_probs is not None
    assert oof_probs.shape[0] == n
    assert oof_probs.shape[1] == 2
    assert np.isnan(oof_probs[0]).all(), "warm-up row should have no honest OOF prediction in any class"
    valid_rows = np.isfinite(oof_probs).all(axis=1)
    assert valid_rows.sum() > 0
    # Valid rows' predicted probabilities must sum to ~1 per row (a genuine predict_proba output).
    row_sums = oof_probs[valid_rows].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_early_stopping_disabled_estimator_survives_timeseries_path():
    """An estimator that already had early_stopping_rounds cleared upstream (this session's fix in
    _compute_oof_preds, applied before branching here) must clone cleanly through every fold without
    reintroducing the eval_set requirement -- regression guard for the early-stopping OOF bug class."""
    lgb = pytest.importorskip("lightgbm")
    X, y = _time_ordered_frame(n=300)
    est = lgb.LGBMRegressor(n_estimators=20, num_leaves=7, verbose=-1, early_stopping_rounds=None)
    oof_preds, oof_probs = _compute_oof_preds_timeseries(estimator=est, train_df=X, train_target=y, method="predict", n_splits=3)
    assert oof_preds is not None
    assert np.isfinite(oof_preds).sum() > 0
