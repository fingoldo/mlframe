"""The packed-key sort of the bootstrap rank walk gives the MI of the argsort path, on columns with and without tied values."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _bootstrap_rank_mi_gpu as B


def _block(n=3000, width=12, seed=0):
    """A float32 block with a few tied columns (values from a small grid) and distinct ones."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, width)).astype(np.float32)
    x[:, 3] = rng.integers(0, 40, n).astype(np.float32)
    x[:, 7] = np.round(x[:, 7], 1)
    x[:, 9] = 0.0
    x[::5, 9] = -0.0
    return x


def _mono(x):
    """The order-preserving 32-bit keys of the transposed block, as the walk builds them."""
    xd = cp.ascontiguousarray(cp.asarray(x).T)
    bits = xd.view(cp.uint32)
    return cp.where(bits >> 31 == 1, ~bits, bits | np.uint32(0x80000000)).astype(cp.uint64)


def test_packed_sort_orders_every_column_and_flags_ties_like_argsort():
    """Each column's permutation sorts its values, and the tie flags equal those of the argsort of (column | value) keys."""
    x = _block()
    width, n = x.shape[1], x.shape[0]
    mono = _mono(x)
    ord32, tie = B._sorted_block_packed(cp, mono, width, n)
    keys = ((cp.arange(width, dtype=cp.uint64)[:, None] << np.uint64(32)) | mono).ravel()
    order = cp.argsort(keys)
    tie_ref = cp.zeros(width * n, dtype=cp.uint8)
    tie_ref[:-1] = (keys[order][1:] == keys[order][:-1]).astype(cp.uint8)
    assert np.array_equal(cp.asnumpy(tie), cp.asnumpy(tie_ref.reshape(width, n)))
    xs = cp.asarray(x).T
    ordered = cp.take_along_axis(xs, ord32.astype(cp.int64), axis=1)
    assert bool(cp.all(ordered[:, 1:] >= ordered[:, :-1]))


def test_walk_gives_the_same_mi_with_the_packed_sort_and_with_the_argsort(monkeypatch):
    """distinct_walk_gpu on a block with ties: MI and done flags equal between the two sorts."""
    x = _block(seed=1)
    n = x.shape[0]
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, n)
    prep = B_prep(x, y)
    assert prep is not None, "the resample tables are always built; a None here is a regression"
    with_packed = B.distinct_walk_gpu(x, prep, 10)
    monkeypatch.setattr(B, "_sorted_block_packed", lambda *a, **k: None)
    with_argsort = B.distinct_walk_gpu(x, prep, 10)
    assert with_packed is not None and with_argsort is not None
    assert np.array_equal(with_packed[1], with_argsort[1])
    assert np.array_equal(with_packed[0], with_argsort[0])


def B_prep(x, y):
    """The resample tables the walk reads, built as the walk's own tests build them (10 bootstrap resamples)."""
    from mlframe.feature_selection.filters import _bootstrap_rank_mi_njit as N

    rng = np.random.default_rng(0)
    reps = [np.sort(rng.integers(0, x.shape[0], x.shape[0])) for _ in range(10)]
    return N._resample_tables(reps, np.asarray(y), x.shape[0], 10, True)
