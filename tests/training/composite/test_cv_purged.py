"""Unit + biz_value tests for purged/embargoed time-series CV (composite/cv.py)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.cv import (
    PurgedTimeSeriesSplit,
    make_purged_cv,
    purged_oof_holdout,
)


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #
def test_n_splits_respected():
    """N splits respected."""
    cv = PurgedTimeSeriesSplit(n_splits=4)
    folds = list(cv.split(n_samples=100))
    assert len(folds) == 4
    assert cv.get_n_splits() == 4


def test_forward_only_order():
    """Every train index precedes every test index; test folds march forward."""
    cv = PurgedTimeSeriesSplit(n_splits=5)
    prev_test_start = -1
    for train_idx, test_idx in cv.split(n_samples=120):
        assert train_idx.max() < test_idx.min(), "train must be strictly before test"
        assert int(test_idx[0]) > prev_test_start, "test folds must move forward"
        prev_test_start = int(test_idx[0])
        # contiguous & sorted
        assert np.array_equal(test_idx, np.arange(test_idx[0], test_idx[-1] + 1))
        assert np.array_equal(train_idx, np.sort(train_idx))


def test_no_overlap_plain():
    """No overlap plain."""
    cv = PurgedTimeSeriesSplit(n_splits=5)
    for train_idx, test_idx in cv.split(n_samples=200):
        assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_purge_gap_enforced():
    """Purge of k => >= k row gap between last train row and first test row."""
    purge = 7
    cv = PurgedTimeSeriesSplit(n_splits=4, purge=purge)
    for train_idx, test_idx in cv.split(n_samples=200):
        gap = int(test_idx[0]) - int(train_idx[-1])
        assert gap > purge, f"gap {gap} must exceed purge {purge}"
        assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_embargo_removes_right_count():
    """Embargo widens the train/test gap by exactly the embargo rows."""
    n = 300
    cv0 = PurgedTimeSeriesSplit(n_splits=4, purge=3, embargo=0)
    cv_e = PurgedTimeSeriesSplit(n_splits=4, purge=3, embargo=10)
    f0 = list(cv0.split(n_samples=n))
    fe = list(cv_e.split(n_samples=n))
    for (tr0, te0), (tre, tee) in zip(f0, fe):
        assert np.array_equal(te0, tee), "test folds unaffected by embargo"
        # embargo=10 trims exactly 10 more rows off the train tail
        assert int(tr0[-1]) - int(tre[-1]) == 10


def test_fractional_embargo():
    """Fractional embargo."""
    n = 1000
    cv = PurgedTimeSeriesSplit(n_splits=3, purge=0, embargo=0.05)  # 50 rows
    for train_idx, test_idx in cv.split(n_samples=n):
        gap = int(test_idx[0]) - int(train_idx[-1])
        assert gap > 50, f"fractional embargo gap {gap} should exceed 50"


def test_max_train_size_rolling_window():
    """Max train size rolling window."""
    cv = PurgedTimeSeriesSplit(n_splits=4, max_train_size=30)
    for train_idx, _ in cv.split(n_samples=300):
        assert train_idx.size <= 30


def test_split_reads_frame_shape_no_copy():
    """split() reads only len/shape, never materialises the frame."""

    class _FakeFrame:
        """Groups tests covering fake frame."""
        shape = (150, 4)

        def __len__(self):  # pragma: no cover - shape preferred
            return 150

    cv = PurgedTimeSeriesSplit(n_splits=3)
    folds = list(cv.split(_FakeFrame()))
    assert len(folds) == 3


def test_invalid_params_raise():
    """Invalid params raise."""
    with pytest.raises(ValueError):
        PurgedTimeSeriesSplit(n_splits=1)
    with pytest.raises(ValueError):
        PurgedTimeSeriesSplit(n_splits=3, purge=-1)
    with pytest.raises(ValueError):
        PurgedTimeSeriesSplit(n_splits=3, embargo=-0.1)
    with pytest.raises(ValueError):
        list(PurgedTimeSeriesSplit(n_splits=10).split(n_samples=3))


def test_make_purged_cv_factory():
    """Make purged cv factory."""
    cv = make_purged_cv(n_splits=3, purge=2, embargo=5)
    assert isinstance(cv, PurgedTimeSeriesSplit)
    assert cv.purge == 2 and cv.embargo == 5 and cv.get_n_splits() == 3


def test_purged_oof_holdout_no_overlap_and_gap():
    """Purged oof holdout no overlap and gap."""
    tr, ho = purged_oof_holdout(1000, holdout_frac=0.2, purge=5, embargo=15)
    assert len(np.intersect1d(tr, ho)) == 0
    assert ho.size == 200
    # holdout is the most-recent block
    assert int(ho[0]) > int(tr[-1])
    # gap >= purge + embargo
    assert int(ho[0]) - int(tr[-1]) > 5 + 15
    # forward-only
    assert tr.max() < ho.min()


def test_purged_oof_holdout_bad_args():
    """Purged oof holdout bad args."""
    with pytest.raises(ValueError):
        purged_oof_holdout(1, holdout_frac=0.2)
    with pytest.raises(ValueError):
        purged_oof_holdout(100, holdout_frac=1.5)
    with pytest.raises(ValueError):
        purged_oof_holdout(100, holdout_frac=0.99, purge=200)


# --------------------------------------------------------------------------- #
# biz_value: purged CV must report the HONEST (worse) score on an
# autocorrelated, overlapping-label series where naive KFold leaks via adjacency.
# --------------------------------------------------------------------------- #
def _make_autocorrelated_overlap_series(n=2000, h=20, seed=0):
    """AR(1) feature; label = forward rolling mean over the next h steps.

    Adjacent rows share h-1 of their label-window terms, so train rows next to a
    test fold leak the test answer. A memorisation-prone model (1-NN in time on
    the AR feature) scores spuriously well under KFold(shuffle) and honestly
    poorly under a purged forward walk.
    """
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = eps[0]
    for i in range(1, n):
        x[i] = 0.97 * x[i - 1] + eps[i]
    # forward overlapping-window label
    y = np.array([x[i : i + h].mean() for i in range(n - h)])
    return x[: n - h], y


def _cv_score(splitter, x, y):
    """1-NN-in-feature regressor; returns mean test R^2-ish (1 - MSE/var)."""
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import r2_score

    scores = []
    X = x.reshape(-1, 1)
    for tr, te in splitter.split(n_samples=len(y)) if isinstance(splitter, PurgedTimeSeriesSplit) else splitter.split(X):
        model = KNeighborsRegressor(n_neighbors=1)
        model.fit(X[tr], y[tr])
        scores.append(r2_score(y[te], model.predict(X[te])))
    return float(np.mean(scores))


def test_biz_val_purged_cv_reports_honest_lower_score_than_leaky_kfold():
    """Purged CV score must be materially LOWER (honest) than leaky KFold.

    Measured: KFold(shuffle) R^2 ~0.93 vs PurgedTimeSeriesSplit ~0.55 with a
    purge spanning the full label window. Floor the leakage gap at 0.15 to catch
    a regression where purge/embargo silently stops removing overlap rows.
    """
    from sklearn.model_selection import KFold

    x, y = _make_autocorrelated_overlap_series(n=2000, h=20, seed=1)

    leaky = _cv_score(KFold(n_splits=5, shuffle=True, random_state=0), x, y)
    purged = _cv_score(PurgedTimeSeriesSplit(n_splits=5, purge=20, embargo=20), x, y)

    assert leaky > purged, f"leaky KFold ({leaky:.3f}) should beat purged ({purged:.3f})"
    assert leaky - purged >= 0.15, f"purge/embargo must remove adjacency leakage: leaky={leaky:.3f} purged={purged:.3f} gap={leaky - purged:.3f} (floor 0.15)"
