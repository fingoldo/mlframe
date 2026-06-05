"""OOF splitter is temporally honest under ``has_time`` and seed-controllable (A7-02, A7-03).

``_compute_oof_preds`` drives ensemble winner selection. On a temporal suite a shuffled ``KFold`` leaks future rows
into the fold that predicts a past row, producing optimistic, selection-biased OOF. With ``has_time=True`` the OOF pass
must use ``TimeSeriesSplit`` (every fold predicts only rows strictly after its training rows). The seed must flow from
the suite master seed, not a hardcoded 42.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _make_dataset(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(4)]), pd.Series(y)


def test_oof_iid_uses_shuffled_seeded_kfold(monkeypatch):
    """The i.i.d. path (has_time=False, no groups) must use a shuffled KFold seeded from the threaded seed."""
    import mlframe.training.trainer as trainer_mod
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LogisticRegression

    captured = {}
    import sklearn.model_selection as skms
    real_cvp = skms.cross_val_predict

    def _spy_cvp(estimator, X, y, *, cv=None, **kw):
        captured["cv"] = cv
        return real_cvp(estimator, X, y, cv=cv, **kw)

    monkeypatch.setattr("sklearn.model_selection.cross_val_predict", _spy_cvp)

    X, y = _make_dataset(n=200, seed=1)
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    trainer_mod._compute_oof_preds(
        model=model, train_df=X, train_target=y.to_numpy(),
        is_classifier_model=True, n_splits=4, random_seed=7, has_time=False,
    )
    assert isinstance(captured["cv"], KFold), f"expected KFold for i.i.d., got {type(captured['cv'])}"
    assert captured["cv"].shuffle is True
    assert captured["cv"].random_state == 7


def test_oof_temporal_does_not_use_cross_val_predict_and_warms_up_with_nan(monkeypatch):
    """has_time=True must take the manual TimeSeriesSplit fold loop (cross_val_predict can't handle non-partition CV)
    and leave the warm-up rows (never held out) as NaN, while later rows receive honest OOF probs."""
    import mlframe.training.trainer as trainer_mod
    from sklearn.linear_model import LogisticRegression

    called = {"cvp": False}

    def _boom_cvp(*a, **k):
        called["cvp"] = True
        raise AssertionError("cross_val_predict must NOT be used on the temporal OOF path")

    monkeypatch.setattr("sklearn.model_selection.cross_val_predict", _boom_cvp)

    X, y = _make_dataset(n=200, seed=2)
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    preds, probs = trainer_mod._compute_oof_preds(
        model=model, train_df=X, train_target=y.to_numpy(),
        is_classifier_model=True, n_splits=4, random_seed=7, has_time=True,
    )
    assert called["cvp"] is False
    assert probs is not None and probs.shape[0] == len(y)
    finite_rows = np.isfinite(probs).all(axis=1)
    assert finite_rows.sum() > 0, "no rows received an OOF prediction"
    assert (~finite_rows).sum() > 0, "expected a NaN warm-up block (the first TimeSeriesSplit train segment)"
    # The warm-up NaN block must be a contiguous prefix (temporal order preserved).
    first_finite = int(np.argmax(finite_rows))
    assert finite_rows[first_finite:].all(), "NaN rows are not a contiguous time-ordered prefix"


def test_oof_timeseries_folds_never_predict_a_past_row():
    """No future training row may produce a held-out prediction for a past row: TimeSeriesSplit guarantees ordered folds."""
    from sklearn.model_selection import TimeSeriesSplit

    n, n_splits = 200, 4
    tss = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tss.split(np.arange(n)):
        # Every test fold index must be strictly greater than every train index (no leak of future into the past).
        assert train_idx.max() < test_idx.min(), "TimeSeriesSplit fold leaks a future row into a past prediction"


def test_oof_shuffled_kfold_seed_is_threaded():
    """Two different seeds must yield different shuffled-KFold OOF on the i.i.d. path (proves seed is honoured, not 42)."""
    from mlframe.training.trainer import _compute_oof_preds
    from sklearn.tree import DecisionTreeClassifier

    X, y = _make_dataset(n=200, seed=3)
    model = DecisionTreeClassifier(max_depth=4, random_state=0)
    model.fit(X, y)

    _, probs_a = _compute_oof_preds(
        model=model, train_df=X, train_target=y.to_numpy(),
        is_classifier_model=True, n_splits=5, random_seed=11, has_time=False,
    )
    _, probs_b = _compute_oof_preds(
        model=model, train_df=X, train_target=y.to_numpy(),
        is_classifier_model=True, n_splits=5, random_seed=999, has_time=False,
    )
    assert probs_a is not None and probs_b is not None
    assert not np.array_equal(probs_a, probs_b), "OOF probs identical across seeds -> seed not threaded into the folds"
