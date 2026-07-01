"""Regression: the single-slot X-content-hash memo must not serve a stale digest when an
``id(X)`` is recycled across fits. The old key was ``(id(X), shape)``; a freed frame A and a
new frame B allocated at the same address with the same shape collided, so B got A's full
digest -> a wrong _FIT_CACHE key (could serve A's selection for B). The fix folds the cheap
strided content signature into the memo key.
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import _mrmr_fingerprints as fp


def test_recycled_id_same_shape_does_not_serve_stale_hash():
    A = np.arange(100, dtype=np.float64).reshape(10, 10)
    B = (np.arange(100, dtype=np.float64) + 1000.0).reshape(10, 10)  # same shape, different content
    hA = fp._full_x_content_hash(A)
    hB = fp._full_x_content_hash(B)
    assert hA and hB and hA != hB

    # Simulate id(B) recycled from a prior frame whose digest was hA by poisoning the single-slot
    # memo with B's (id, shape) but the WRONG hash. Pre-fix the (id, shape)-only key matched and
    # returned hA; post-fix the content-signature in the key makes it a miss -> recompute -> hB.
    fp._MRMR_LAST_X_HASH_CACHE["id_shape"] = (id(B), B.shape)
    fp._MRMR_LAST_X_HASH_CACHE["hash"] = hA
    got = fp._full_x_content_hash(B)
    assert got == hB, "stale digest served for a recycled id+shape -- content discriminator missing from memo key"


def test_intra_fit_same_object_still_hits_memo():
    """The memo must still short-circuit the second call on the SAME object (the perf win it exists for)."""
    X = np.arange(64, dtype=np.float64).reshape(8, 8)
    h1 = fp._full_x_content_hash(X)
    # The memo now holds X's 3-tuple key; a second call returns the cached hash (same value).
    assert fp._full_x_content_hash(X) == h1
    assert fp._MRMR_LAST_X_HASH_CACHE["id_shape"][0] == id(X)
