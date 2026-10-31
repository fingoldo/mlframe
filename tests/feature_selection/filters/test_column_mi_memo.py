"""The batch plug-in MI dispatch scores each distinct (column, target) once: repeats and reordered batches return the stored values, and the values equal the unmemoised batch."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import _hermite_fe_mi as H


def _data(n=4000, k=6, seed=0):
    """A float32 matrix with a signal column and a binary target."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k)).astype(np.float32)
    y = (X[:, 0] + 0.7 * rng.standard_normal(n) > 0).astype(np.int64)
    return X, y


def test_values_equal_the_unmemoised_dispatch_and_repeats_are_not_rescored(monkeypatch):
    """A second call, a reordered batch and a batch of old and new columns give the unmemoised values; only new columns reach the kernel."""
    H._COLUMN_MI_MEMO.clear()
    X, y = _data()
    want = H._plugin_mi_classif_batch_dispatch_uncached(X, y, 10)
    scored = []
    real = H._plugin_mi_classif_batch_dispatch_uncached
    monkeypatch.setattr(H, "_plugin_mi_classif_batch_dispatch_uncached", lambda c, yy, b=20: (scored.append(c.shape[1]), real(c, yy, b))[1])
    assert np.array_equal(H.plugin_mi_classif_batch_dispatch(X, y, 10), want)
    assert np.array_equal(H.plugin_mi_classif_batch_dispatch(X, y, 10), want)
    perm = np.array([3, 1, 5, 0, 2, 4])
    assert np.array_equal(H.plugin_mi_classif_batch_dispatch(np.ascontiguousarray(X[:, perm]), y, 10), want[perm])
    extra = np.random.default_rng(1).standard_normal((len(y), 2)).astype(np.float32)
    mixed = np.ascontiguousarray(np.column_stack([X[:, :2], extra]))
    got = H.plugin_mi_classif_batch_dispatch(mixed, y, 10)
    assert np.array_equal(got[:2], want[:2]) and np.array_equal(got[2:], real(extra, y, 10))
    assert scored == [6, 2]


def test_another_target_or_bin_count_is_a_different_entry():
    """The key holds the target content and the bin count."""
    H._COLUMN_MI_MEMO.clear()
    X, y = _data()
    a = H.plugin_mi_classif_batch_dispatch(X, y, 10)
    b = H.plugin_mi_classif_batch_dispatch(X, 1 - y, 10)
    c = H.plugin_mi_classif_batch_dispatch(X, y, 5)
    assert np.allclose(a, b) and not np.array_equal(a, c)


def test_column_fingerprints_follow_content_and_order_not_layout():
    """Equal columns get equal fingerprints (also from a differently laid out matrix), and a changed value or a changed order changes them."""
    from mlframe.feature_selection.filters._numba_utils import column_fingerprints

    rng = np.random.default_rng(3)
    X = rng.standard_normal((2000, 5)).astype(np.float32)
    X[:, 3] = X[:, 1]
    fp = column_fingerprints(np.ascontiguousarray(X).view(np.uint32))
    assert np.array_equal(fp[1], fp[3]) and len({tuple(r) for r in fp}) == 4
    fp_f = column_fingerprints(np.asfortranarray(X).view(np.uint32))
    assert np.array_equal(fp, fp_f)
    changed = X.copy()
    changed[777, 1] += 1e-3
    assert not np.array_equal(column_fingerprints(changed.view(np.uint32))[1], fp[1])
    swapped = X.copy()
    swapped[[10, 11], 1] = swapped[[11, 10], 1]
    assert not np.array_equal(column_fingerprints(swapped.view(np.uint32))[1], fp[1])
