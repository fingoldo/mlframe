"""CatBoost scores a custom eval_metric on the learn set too; ICE must report a real (subsampled) value there, never the 1e6 sentinel.

The learn value used to be the ``ICE_UNCOMPUTABLE`` sentinel on every iteration once an eval set was seen, which drew a flat 1e6 train
line in CatBoost's own plot and in mlframe's training-curve chart.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics._ice_metric import ICE
from mlframe.metrics.calibration import ICE_UNCOMPUTABLE


def _brier(y_true, y_score, sample_weight=None):
    """Weighted Brier score of the positive-class column of y_score."""
    p = np.asarray(y_score)[:, 1]
    return float(np.average((p - np.asarray(y_true)) ** 2, weights=sample_weight))


def _data(n, seed=0):
    """Random logits with labels drawn from their sigmoid probabilities."""
    rng = np.random.default_rng(seed)
    logit = rng.normal(size=n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.float64)
    return logit, y


def test_learn_set_scored_on_subsample_not_sentinel():
    """Learn set scored on subsample not sentinel."""
    m = ICE(metric=_brier, higher_is_better=False, skip_largest_set=True, learn_sample_size=2_000)
    lv, ly = _data(20_000, seed=1)
    vv, vy = _data(3_000, seed=2)
    m.evaluate((lv,), ly, None)
    val = m.evaluate((vv,), vy, None)[0]
    learn = m.evaluate((lv,), ly, None)[0]
    assert val == pytest.approx(_brier(vy, np.column_stack([1 - 1 / (1 + np.exp(-vv)), 1 / (1 + np.exp(-vv))])))
    assert learn != ICE_UNCOMPUTABLE
    full = _brier(ly, np.column_stack([1 - 1 / (1 + np.exp(-lv)), 1 / (1 + np.exp(-lv))]))
    assert abs(learn - full) < 0.01, "a 2k-row subsample of a 20k-row learn set must estimate the full-set Brier closely"
    assert m.evaluate((lv,), ly, None)[0] == learn, "the subsample is fixed, so the learn curve is not re-drawn noise every iteration"


def test_oversized_eval_set_is_subsampled_not_sentinel():
    """Oversized eval set is subsampled not sentinel."""
    m = ICE(metric=_brier, higher_is_better=False, max_arr_size=1_000)
    v, y = _data(5_000, seed=3)
    assert m.evaluate((v,), y, None)[0] < 0.5


def test_sample_weights_follow_the_subsample():
    """Sample weights follow the subsample."""
    m = ICE(metric=_brier, higher_is_better=False, max_arr_size=1_000)
    v, y = _data(5_000, seed=4)
    w = np.where(y > 0, 3.0, 1.0)
    assert np.isfinite(m.evaluate((v,), y, w)[0])


def test_legacy_skip_opt_out_keeps_sentinel():
    """Legacy skip opt out keeps sentinel."""
    m = ICE(metric=_brier, higher_is_better=False, skip_largest_set=True, subsample_skipped_sets=False)
    lv, ly = _data(5_000)
    vv, vy = _data(1_000)
    m.evaluate((lv,), ly, None)
    m.evaluate((vv,), vy, None)
    assert m.evaluate((lv,), ly, None)[0] == ICE_UNCOMPUTABLE


def test_subsample_cache_not_pickled():
    """Subsample cache not pickled."""
    import pickle

    m = ICE(metric=_brier, higher_is_better=False, max_arr_size=1_000)
    v, y = _data(5_000)
    m.evaluate((v,), y, None)
    assert m._sample_idx
    m2 = pickle.loads(pickle.dumps(m))  # nosec B301 - round-trip of an object this code just pickled
    assert m2._sample_idx == {}
    assert m2.evaluate((v,), y, None)[0] == m.evaluate((v,), y, None)[0]


def test_catboost_learn_curve_has_no_sentinel():
    """Catboost learn curve has no sentinel."""
    catboost = pytest.importorskip("catboost")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3_000, 4))
    y = (X[:, 0] + rng.normal(size=3_000) > 0).astype(int)
    m = ICE(metric=_brier, higher_is_better=False, skip_largest_set=True, learn_sample_size=500)
    clf = catboost.CatBoostClassifier(iterations=15, verbose=0, eval_metric=m, thread_count=1)
    clf.fit(X[:2_400], y[:2_400], eval_set=(X[2_400:], y[2_400:]))
    res = clf.get_evals_result()
    learn = res["learn"]["ICE"]
    assert len(learn) == 15
    assert not np.any(np.asarray(learn) == ICE_UNCOMPUTABLE)
    assert not np.any(np.asarray(res["validation"]["ICE"]) == ICE_UNCOMPUTABLE)


def test_eval_set_larger_than_learn_is_scored_in_full():
    """Eval set larger than learn is scored in full."""
    sizes = []

    def _rec(y_true, y_score):
        """Record the size of the set being scored and return a constant."""
        sizes.append(len(y_true))
        return 0.25

    m = ICE(metric=_rec, higher_is_better=False, skip_largest_set=True, learn_sample_size=300)
    lv, ly = _data(1_000)
    vv, vy = _data(4_000)
    for _ in range(2):
        m.evaluate((lv,), ly, None)
        m.evaluate((vv,), vy, None)
    assert sizes == [1_000, 4_000, 300, 4_000], "the early-stopping eval set is never subsampled, even when it outnumbers the learn set"


def test_catboost_scores_learn_before_eval_even_when_eval_is_larger():
    """Catboost scores learn before eval even when eval is larger."""
    catboost = pytest.importorskip("catboost")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3_000, 4))
    y = (X[:, 0] + rng.normal(size=3_000) > 0).astype(int)
    m = ICE(metric=_brier, higher_is_better=False, skip_largest_set=True, learn_sample_size=500)
    clf = catboost.CatBoostClassifier(iterations=10, verbose=0, eval_metric=m, thread_count=1)
    clf.fit(X[:1_000], y[:1_000], eval_set=(X[1_000:], y[1_000:]))
    val = np.asarray(clf.get_evals_result()["validation"]["ICE"])
    p = clf.predict_proba(X[1_000:])
    assert val[-1] == pytest.approx(_brier(y[1_000:], p), rel=1e-9), "the eval set early stopping reads is scored on all its rows"
