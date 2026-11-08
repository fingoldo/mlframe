"""partial_fit queues small batches and merges them once: same rows, same refit input, no re-concatenation per batch."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _mrmr_partial_fit as P
from mlframe.feature_selection.filters.mrmr import MRMR


def _batch(n, seed):
    """A small frame and binary target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n), "c": rng.standard_normal(n)})
    return X, pd.Series((X["a"] + 0.3 * rng.standard_normal(n) > 0).astype(int))


def test_merge_pending_equals_the_per_batch_concatenation():
    """Merging the queue once gives the frame that concatenating batch by batch gives."""
    X0, y0 = _batch(50, 0)
    queue = [_batch(30, s) for s in (1, 2, 3)]
    step_X, step_y = X0, y0
    for x, y in queue:
        step_X = pd.concat([step_X, x], axis=0, ignore_index=True)
        step_y = pd.concat([step_y, y], axis=0, ignore_index=True)
    got_X, got_y = P._merge_pending(X0, y0, queue)
    pd.testing.assert_frame_equal(got_X, step_X)
    pd.testing.assert_series_equal(got_y, step_y)
    assert P._merge_pending(X0, y0, []) == (X0, y0) or P._merge_pending(X0, y0, [])[0] is X0


def test_many_small_batches_below_the_threshold_do_not_concatenate(monkeypatch):
    """Ten batches below the recompute threshold call the merge zero times; the refit merges once."""
    calls = []
    real = P._merge_pending
    monkeypatch.setattr(P, "_merge_pending", lambda *a: calls.append(len(a[2])) or real(*a))
    m = MRMR(verbose=0, random_seed=0, n_jobs=1, fe_max_steps=0, partial_fit_min_recompute=1000)
    X, y = _batch(200, 5)
    m.partial_fit(X, y)
    for s in range(10):
        Xb, yb = _batch(20, 10 + s)
        m.partial_fit(Xb, yb)
    assert calls == []
    assert len(m._partial_fit_X_buffer_) + sum(len(x) for x, _ in m._partial_fit_pending_) == 400
