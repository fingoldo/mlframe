"""The resample index matrix must never exist whole.

At the 1000 resamples this bundle is called with, an int64 ``(n_bootstrap, n)`` matrix is 8 GB at n=1M and
56 GB at the 7M rows this package's own predict guards cite as production -- while the live working set is
one ``chunk_size`` block, because the only consumer already walked the matrix in blocks.

Chunking is only allowed to change PEAK MEMORY. The RNG stream must be untouched, since these resamples are
pinned bit-identical to the generic serial path for a given ``random_state``, so the tests below assert both
halves: the chunks are bounded, and their concatenation is exactly what one whole-matrix build would have
produced from the same seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.evaluation._bootstrap_fused_binary_bundle import _iter_resample_idx_chunks


def _whole_matrix(seed: int, n_bootstrap: int, n: int, stratify=None) -> np.ndarray:
    """What the pre-chunking builder produced: every row drawn in order from one Generator."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_bootstrap, n), dtype=np.int64)
    if stratify is None:
        for i in range(n_bootstrap):
            out[i] = rng.integers(0, n, size=n, dtype=np.int64)
        return out
    groups = [np.flatnonzero(stratify == c) for c in np.unique(stratify)]
    sizes = np.array([g.shape[0] for g in groups], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    for i in range(n_bootstrap):
        for c, g in enumerate(groups):
            sz = int(sizes[c])
            out[i, offsets[c] : offsets[c + 1]] = g[rng.integers(0, sz, size=sz, dtype=np.int64)]
    return out


def _collect(seed: int, n_bootstrap: int, n: int, chunk_size: int, stratify=None):
    """Drive the generator and return (chunk shapes, concatenated rows)."""
    rng = np.random.default_rng(seed)
    shapes, rows = [], []
    for lo, hi, chunk in _iter_resample_idx_chunks(rng, n_bootstrap, n, stratify, chunk_size):
        assert chunk.shape[0] == hi - lo, f"chunk rows {chunk.shape[0]} do not match its own span {lo}:{hi}"
        shapes.append(chunk.shape[0])
        rows.append(chunk)
    return shapes, np.concatenate(rows, axis=0)


@pytest.mark.parametrize("chunk_size", [1, 7, 200])
def test_no_chunk_exceeds_the_requested_block(chunk_size: int):
    """Peak memory is the point: a chunk larger than the block would defeat it."""
    shapes, _ = _collect(seed=0, n_bootstrap=23, n=64, chunk_size=chunk_size)
    assert shapes, "the generator yielded nothing"
    assert max(shapes) <= chunk_size, f"a chunk of {max(shapes)} rows exceeds chunk_size={chunk_size}"
    assert sum(shapes) == 23, f"the chunks cover {sum(shapes)} resamples, not 23"


@pytest.mark.parametrize("chunk_size", [1, 7, 200])
def test_the_rng_stream_is_untouched_by_chunking(chunk_size: int):
    """Bit-identical to a single whole-matrix build from the same seed -- not merely close."""
    _, got = _collect(seed=11, n_bootstrap=23, n=64, chunk_size=chunk_size)
    expected = _whole_matrix(seed=11, n_bootstrap=23, n=64)
    assert np.array_equal(got, expected), "chunking changed the resample draw order"


@pytest.mark.parametrize("chunk_size", [1, 5, 200])
def test_the_stratified_path_is_also_bit_identical(chunk_size: int):
    """The stratified branch draws per class per row; its call order must survive chunking too."""
    stratify = np.repeat([0, 1, 2], 8)
    _, got = _collect(seed=3, n_bootstrap=11, n=stratify.shape[0], chunk_size=chunk_size, stratify=stratify)
    expected = _whole_matrix(seed=3, n_bootstrap=11, n=stratify.shape[0], stratify=stratify)
    assert np.array_equal(got, expected), "chunking changed the stratified resample draw order"


def test_a_stratified_resample_keeps_every_class_count():
    """Guards the branch itself: stratified resampling must preserve each class's row count."""
    stratify = np.repeat([0, 1, 2], 8)
    _, rows = _collect(seed=5, n_bootstrap=4, n=stratify.shape[0], chunk_size=2, stratify=stratify)
    for row in rows:
        counts = np.bincount(stratify[row], minlength=3)
        assert list(counts) == [8, 8, 8], f"class counts drifted to {list(counts)}"
