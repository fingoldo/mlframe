"""CPX32 regression: streamed row-chunk X-content hashing is bit-identical to whole-frame ``.tobytes()``.

Pins the RAM-saving change in ``_fit_init._stream_hash_array`` (RFECV skip-retrain signature). The
streamed hash MUST equal ``blake2b(np.ascontiguousarray(arr).tobytes())`` across dtypes, shapes, and
both C- and F-order inputs, and the pandas/polars numeric-block fingerprints must match each other.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.wrappers.rfecv._fit_init import _stream_hash_array


def _streamed(arr: np.ndarray) -> str:
    """Returns ``h.hexdigest()`` (after 2 setup steps)."""
    h = hashlib.blake2b(digest_size=12)
    _stream_hash_array(h, arr)
    return h.hexdigest()


def _whole(arr: np.ndarray) -> str:
    """Returns ``hashlib.blake2b(np.ascontiguousarray(arr).tobytes(), digest_size=12).hexdigest()``."""
    return hashlib.blake2b(np.ascontiguousarray(arr).tobytes(), digest_size=12).hexdigest()


@pytest.mark.parametrize(
    "shape",
    [(0, 5), (1, 1), (7, 1), (1, 9), (333, 17), (10_000, 13), (5, 7)],
)
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int32, np.bool_])
def test_stream_hash_matches_whole_tobytes(shape, dtype):
    """Stream hash matches whole tobytes."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 5, size=shape)
    arr = (base if dtype != np.bool_ else base % 2).astype(dtype)
    assert _streamed(arr) == _whole(arr)


@pytest.mark.parametrize("order", ["C", "F"])
def test_stream_hash_matches_on_non_contiguous(order):
    """Stream hash matches on non contiguous."""
    rng = np.random.default_rng(1)
    arr = np.asarray(rng.random((201, 11)), order=order)
    assert _streamed(arr) == _whole(arr)
    # A genuinely strided view must also match the contiguous-buffer bytes.
    view = arr[::2, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    assert _streamed(view) == _whole(view)


def test_stream_hash_forces_small_chunks(monkeypatch):
    """Tiny chunk budget exercises the multi-update path; result is unchanged."""
    import mlframe.feature_selection.wrappers.rfecv._fit_init as fi

    monkeypatch.setattr(fi, "_HASH_CHUNK_BYTES", 64)
    arr = np.random.default_rng(2).random((500, 9))
    assert _streamed(arr) == _whole(arr)


def test_pandas_polars_numeric_block_fingerprint_match():
    """Pandas polars numeric block fingerprint match."""
    rng = np.random.default_rng(3)
    data = rng.random((400, 6))
    pdf = pd.DataFrame(data, columns=[f"c{i}" for i in range(6)])
    p_pd = _streamed(pdf.to_numpy())
    try:
        import polars as pl
    except ImportError:  # pragma: no cover
        pytest.skip("polars not installed")
    pldf = pl.DataFrame(data, schema=[f"c{i}" for i in range(6)])
    p_pl = _streamed(np.asarray(pldf))
    assert p_pd == p_pl == _whole(data)
