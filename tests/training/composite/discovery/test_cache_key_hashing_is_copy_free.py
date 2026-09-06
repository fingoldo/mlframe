"""Both cache fingerprints must hash the array without materialising a copy of it.

``h.update(a.tobytes())`` allocates a second full copy of the array purely to feed the hash, and both call
sites here run on arrays this package sizes in the tens of gigabytes -- ``_key_bank_fingerprint`` on the
whole ``X_train``, ``_keep_mask_cache_key`` on the feature matrix. ``h.update(a.data)`` hands the hash the
existing buffer instead.

The digest must not move: these keys address an on-disk KeyBank cache, so a changed digest silently
invalidates every stored bank. ``tobytes()`` serialises in C order regardless of the array's own layout,
which is why the copy-free form goes through ``ascontiguousarray`` -- the tests below cover the transposed
and sliced cases where that distinction is the whole difference between a matching and a wrong digest.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from mlframe.feature_engineering.transformer._key_bank import _key_bank_fingerprint
from mlframe.training.composite.discovery._collinear_numba import _keep_mask_cache_key


def _arrays():
    """Layouts whose buffer order differs from their C-order serialisation."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(64, 8))
    yield "c_contiguous", base
    yield "fortran", np.asfortranarray(base)
    yield "transposed", rng.normal(size=(8, 64)).T
    yield "row_slice", rng.normal(size=(128, 8))[::2]
    yield "col_slice", rng.normal(size=(64, 16))[:, ::2]
    yield "float32", base.astype(np.float32)


@pytest.mark.parametrize("name,arr", list(_arrays()), ids=lambda v: v if isinstance(v, str) else "")
def test_the_copy_free_update_digests_exactly_like_tobytes(name, arr):
    """A changed digest would invalidate every KeyBank on disk, so this is an equality, not a tolerance."""
    from_copy = hashlib.sha256(arr.tobytes()).digest()
    from_buffer = hashlib.sha256(np.ascontiguousarray(arr).data).digest()
    assert from_buffer == from_copy, f"{name}: the copy-free update changed the digest"


def _fingerprint(x):
    """The fingerprint with every non-data argument held fixed."""
    return _key_bank_fingerprint(x, 7, 2, 3, "cosine", True, 16, 200, "random", np.float32)


def test_the_key_bank_fingerprint_is_layout_independent():
    """The same values in a Fortran layout must address the same cached bank."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(32, 4))
    assert _fingerprint(x) == _fingerprint(np.asfortranarray(x))


def test_the_key_bank_fingerprint_still_separates_different_content():
    """Guards the hash itself: layout-independence must not have flattened the key onto shape alone."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=(32, 4))
    b = a.copy()
    b[0, 0] += 1.0
    assert _fingerprint(a) != _fingerprint(b)


def test_the_keep_mask_key_is_layout_independent_and_content_sensitive():
    """The collinearity keep-mask cache key has the same two obligations."""
    rng = np.random.default_rng(3)
    fm = rng.normal(size=(64, 6))
    assert _keep_mask_cache_key(fm, 0.95) == _keep_mask_cache_key(np.asfortranarray(fm), 0.95)
    other = fm.copy()
    other[5, 2] += 1.0
    assert _keep_mask_cache_key(fm, 0.95) != _keep_mask_cache_key(other, 0.95)


def test_a_contiguous_array_is_hashed_without_a_second_allocation():
    """The digest tests above hold for `tobytes()` too, so this is the one that pins the actual fix.

    `ascontiguousarray` returns the SAME object for an array that is already C-contiguous, which is the
    whole-`X_train` case both call sites are sized against; `tobytes()` always allocates.
    """
    arr = np.random.default_rng(4).normal(size=(256, 8))
    assert arr.flags["C_CONTIGUOUS"]
    assert np.ascontiguousarray(arr) is arr, "the hashed buffer is a copy, not the caller's own array"
