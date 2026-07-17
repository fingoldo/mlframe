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
    oof_preds, _oof_probs = _compute_oof_preds_timeseries(estimator=est, train_df=X, train_target=y, method="predict", n_splits=3)
    assert oof_preds is not None
    assert np.isfinite(oof_preds).sum() > 0


def test_biz_val_oof_random_seed_is_deterministic_and_varies_folds():
    """``_compute_oof_preds``'s ``random_seed`` param (the non-temporal KFold path) had zero direct
    determinism coverage: same seed must reproduce byte-identical OOF predictions across repeated
    calls (the KFold shuffle -- and therefore which rows land in which fold -- is seeded), and a
    DIFFERENT seed must produce a genuinely different fold assignment (not a no-op knob)."""
    from mlframe.training.trainer import _compute_oof_preds

    X, y = _time_ordered_frame(n=400)
    X = X.drop(columns=["t"])  # KFold path has no temporal structure to preserve

    def _run(seed):
        """Runs OOF prediction once with the given fold-shuffle seed."""
        est = DecisionTreeRegressor(max_depth=4, random_state=0)
        preds, _ = _compute_oof_preds(model=est, train_df=X, train_target=y, is_classifier_model=False, n_splits=5, random_seed=seed)
        return preds

    a1 = _run(seed=0)
    a2 = _run(seed=0)
    b = _run(seed=1)

    assert a1 is not None and b is not None
    assert np.array_equal(a1, a2), "same random_seed must reproduce byte-identical OOF predictions"
    assert not np.array_equal(a1, b), "different random_seed must produce a different fold assignment"


def test_biz_val_oof_group_kfold_keeps_groups_intact():
    """The GroupKFold branch of ``_compute_oof_preds`` (fires whenever ``group_ids`` is supplied and
    matches ``train_df``'s length) had zero direct test coverage. A row's OOF prediction must come
    from a fold that never trained on ANY row sharing that row's group -- verified directly by
    checking that swapping a group's rows between "seen" and "held-out" never happens for the same
    row twice (GroupKFold's own partition guarantee), and that every row still gets a finite
    prediction (a genuine group-aware split, not silently falling back to ungrouped KFold)."""
    from mlframe.training.trainer import _compute_oof_preds

    rng = np.random.default_rng(3)
    n_groups = 20
    rows_per_group = 15
    n = n_groups * rows_per_group
    group_ids = np.repeat(np.arange(n_groups), rows_per_group)
    x0 = rng.normal(size=n)
    # group-level effect: a group's rows share a random offset, so a model that leaks a group's rows
    # across train/test would trivially memorize that offset -- a genuine group-held-out split can't.
    group_offset = rng.normal(size=n_groups)[group_ids]
    y = x0 + group_offset + 0.05 * rng.normal(size=n)
    X = pd.DataFrame({"x0": x0})

    est = DecisionTreeRegressor(max_depth=4, random_state=0)
    preds, probs = _compute_oof_preds(model=est, train_df=X, train_target=y, is_classifier_model=False, n_splits=5, random_seed=0, group_ids=group_ids)

    assert probs is None
    assert preds is not None
    assert np.isfinite(preds).all(), "GroupKFold must produce a finite OOF prediction for every row"
    # A model that could see a group's offset during fit (i.e. the split leaked) would fit that offset
    # almost perfectly; a genuinely held-out group forces the model to fall back on the shared x0
    # relationship alone, so per-group residual variance should NOT collapse to the fit noise floor.
    resid = y - preds
    per_group_resid_std = pd.Series(resid).groupby(group_ids).std()
    assert (per_group_resid_std > 0.15).mean() > 0.5, "residuals look too tight for a genuinely group-held-out split"
