"""The KSG gate and the ensemble scorer share their per-column KSG values within a fit: the second asks for columns the first has already scored and gets them from
the memo; a float32 and a float64 copy of the same values share an entry; the values are those of the per-column kernel, and within rounding of the block kernel."""

import numpy as np

from mlframe.feature_selection.filters import _ksg_cd_fast as K
from mlframe.feature_selection.filters._orthogonal_ksg_mi_fe import _ksg_mi_batch
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_scoring_memo as M


def _data(n=3000, p=12, seed=0):
    """Continuous columns, two of them informative, and a binary target."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = ((X[:, 0] + X[:, 1] ** 2 + 0.5 * rng.standard_normal(n)) > 0.8).astype(np.int64)
    return X, y


def test_the_second_caller_reads_the_memo_and_gets_the_same_values(monkeypatch):
    """Inside a fit scope the gate's batch (float64) scores the columns; the ensemble's call on the float32 copy computes none of them again and returns the same numbers."""
    X, y = _data()
    calls = []
    real = K.ksg_mi_discrete_target_columns

    def spy(Xs, ys, **kw):
        """Count the columns actually computed."""
        calls.append(Xs.shape[1])
        return real(Xs, ys, **kw)

    monkeypatch.setattr(K, "ksg_mi_discrete_target_columns", spy)
    with M.orth_scoring_memo_scope():
        first = _ksg_mi_batch(X.astype(np.float64), y, n_neighbors=3, random_state=0)
        second = M.cached_ksg_columns(X, y, n_neighbors=3, random_state=0)
    assert calls == [X.shape[1]]
    assert np.array_equal(first, second)


def test_outside_a_scope_nothing_is_remembered(monkeypatch):
    """No scope: every call computes."""
    X, y = _data(seed=1)
    calls = []
    real = K.ksg_mi_discrete_target_columns
    monkeypatch.setattr(K, "ksg_mi_discrete_target_columns", lambda Xs, ys, **kw: (calls.append(Xs.shape[1]), real(Xs, ys, **kw))[1])
    M.cached_ksg_columns(X, y, n_neighbors=3, random_state=0)
    M.cached_ksg_columns(X, y, n_neighbors=3, random_state=0)
    assert calls == [X.shape[1], X.shape[1]]


def test_the_values_equal_the_per_column_kernel_and_the_block_kernel_within_its_jitter():
    """The gate now scores with the per-column kernel: equal to it bit for bit, and to the block kernel it used before within the 1e-10 tie-breaking jitter's effect."""
    for seed in (0, 1):
        X, y = _data(seed=seed)
        new = _ksg_mi_batch(X.astype(np.float64), y, n_neighbors=3, random_state=0)
        assert np.array_equal(new, K.ksg_mi_discrete_target_columns(X.astype(np.float64), y, n_neighbors=3, random_state=0))
        old = K.ksg_mi_discrete_target(X.astype(np.float64), y, n_neighbors=3, random_state=0)
        assert np.allclose(new, old, rtol=0, atol=1e-3) and np.array_equal(np.argsort(-new)[:3], np.argsort(-old)[:3])
