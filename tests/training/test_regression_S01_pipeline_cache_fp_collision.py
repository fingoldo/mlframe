"""Regression sensor for S01: ``_pre_pipeline_cache_key`` fingerprints
``train_target`` via a 4-cell point-sample. Two distinct targets whose
values at the four sampled positions (first, near-head, midpoint, last)
happen to coincide collide on the cache key, so the second target
silently consumes the first target's fit-transform output.

The fix is to fold a full blake2b content hash of ``train_target`` into
the cache key (mirroring ``_full_y_content_hash`` in
``_mrmr_fingerprints.py``). With the fix, two Series that share their
four boundary cells but differ in the middle produce DIFFERENT cache
keys.

This sensor MUST FAIL on the pre-fix code (collision) and PASS on the
post-fix code (distinct cache keys).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training._pipeline_cache import _pre_pipeline_cache_key


def _make_collision_pair_series(n: int = 1000) -> tuple[pd.Series, pd.Series]:
    """Return two Series whose four sampled positions coincide but
    whose middle differs in many rows.

    The point-sample positions used by ``_content_fingerprint_for_cache``
    on a pandas Series are ``(0, min(8, n-1), n // 2, n - 1)``. We pin
    those four cells to identical values on both series and diverge
    everywhere else.
    """
    base = np.zeros(n, dtype=np.float64)
    base[: n // 2] = 1.0
    other = base.copy()
    mid_lo = (n // 2) + 1
    mid_hi = n - 1
    other[mid_lo:mid_hi] = 1.0 - other[mid_lo:mid_hi]
    sample_positions = (0, min(8, n - 1), n // 2, n - 1)
    for pos in sample_positions:
        assert base[pos] == other[pos], "fixture broken: sampled cells must coincide"
    assert not np.array_equal(base, other), "fixture broken: full arrays must differ"
    return pd.Series(base, name="y"), pd.Series(other, name="y")


def test_pre_pipeline_cache_key_distinguishes_targets_with_shared_boundary_cells():
    """Two pandas Series targets that share boundary samples but differ
    in the middle MUST produce DIFFERENT pre-pipeline cache keys. Pre-fix
    the cell-sample fingerprint collides on this case (the suite would
    silently serve target-1's fit output for target-2); post-fix the
    full blake2b content hash separates them.
    """
    y1, y2 = _make_collision_pair_series(n=1000)
    # Same train_df / val_df / pipeline / target_name across both calls so the
    # ONLY thing that varies is the target content. Pipeline=None is acceptable
    # for the key builder; the suite passes a real pipeline but it's identical
    # across per-target loop iterations on the same model.
    train_df = pd.DataFrame({"x": np.arange(1000, dtype=np.float64)})
    val_df = pd.DataFrame({"x": np.arange(200, dtype=np.float64)})
    key1 = _pre_pipeline_cache_key(
        train_df, val_df, pipeline=None,
        train_target=y1, target_name="y",
    )
    key2 = _pre_pipeline_cache_key(
        train_df, val_df, pipeline=None,
        train_target=y2, target_name="y",
    )
    assert key1 != key2, (
        "S01: _pre_pipeline_cache_key collides on two distinct targets "
        "sharing only their boundary cells; the cache would silently "
        "serve target-1's fit output for target-2. Expected DIFFERENT keys."
    )


def test_pre_pipeline_cache_key_distinguishes_numpy_targets_with_shared_boundary_cells():
    """Same collision shape on the raw numpy target path. The numpy
    fingerprint samples 10 evenly-spaced positions; the constructed pair
    coincides on every sampled cell but diverges in the middle.
    """
    n = 1000
    base = np.zeros(n, dtype=np.float64)
    base[: n // 2] = 1.0
    other = base.copy()
    np_positions = [int(i * (n - 1) / 9) for i in range(10)]
    sampled = set(np_positions)
    diverge_idx = [i for i in range(n) if i not in sampled]
    flip_slice = diverge_idx[len(diverge_idx) // 3: 2 * len(diverge_idx) // 3]
    other[flip_slice] = 1.0 - other[flip_slice]
    for pos in np_positions:
        assert base[pos] == other[pos]
    assert not np.array_equal(base, other)
    train_df = pd.DataFrame({"x": np.arange(n, dtype=np.float64)})
    val_df = pd.DataFrame({"x": np.arange(200, dtype=np.float64)})
    key1 = _pre_pipeline_cache_key(
        train_df, val_df, pipeline=None,
        train_target=base, target_name="y",
    )
    key2 = _pre_pipeline_cache_key(
        train_df, val_df, pipeline=None,
        train_target=other, target_name="y",
    )
    assert key1 != key2, (
        "S01: _pre_pipeline_cache_key collides on two distinct numpy "
        "targets sharing all 10 sampled positions; full-content hash "
        "required."
    )


def test_full_target_content_hash_bit_identical_to_tobytes_reference():
    """``_full_target_content_hash`` hashes the contiguous array via the buffer
    protocol directly (no ``.tobytes()`` copy). Pin that the digest is
    bit-identical to the explicit ``blake2b(ascontiguousarray(arr).tobytes())``
    + shape + dtype reference, across dtypes / shapes / a non-contiguous view, so
    the copy-elision can never silently drift the cache key.
    """
    import hashlib
    from mlframe.training._pipeline_cache import _full_target_content_hash

    def _reference(np_arr: np.ndarray) -> str:
        h = hashlib.blake2b(np.ascontiguousarray(np_arr).tobytes(), digest_size=16)
        h.update(str(np_arr.shape).encode())
        h.update(str(np_arr.dtype).encode())
        return h.hexdigest()

    rng = np.random.default_rng(0)
    cases = [
        rng.standard_normal(10_000),
        rng.integers(0, 7, 10_000),
        rng.standard_normal((3_000, 4)),
        rng.integers(0, 2, 257).astype(np.int8),
        rng.standard_normal((1_000, 6))[:, ::2],  # non-contiguous view
    ]
    for arr in cases:
        assert _full_target_content_hash(arr) == _reference(arr), (
            f"digest drift on shape={arr.shape} dtype={arr.dtype}"
        )
    # Content sensitivity is preserved: a single flipped cell changes the digest.
    a = rng.standard_normal(5_000)
    b = a.copy(); b[2_500] += 1.0
    assert _full_target_content_hash(a) != _full_target_content_hash(b)
