"""Audit 2026-11-20 finding 34-F27: the shared-fold cache keys on a collision-safe digest of the row indices.

A 64-bit ``hash()`` of the row bytes could map two different folds to one key and return the wrong factorisation or dataset.
The test checks that the digest separates different row sets and is stable for equal ones, in both modules.
"""

import numpy as np

from mlframe.training.composite.discovery import _lgb_shared_fold, _ridge_shared_fold


def test_row_digest_separates_rows_and_is_stable():
    """Different row sets give different digests; equal row sets give the same digest, for both modules."""
    a = np.array([0, 1, 2, 5], dtype=np.int64)
    b = np.array([0, 1, 2, 6], dtype=np.int64)
    for mod in (_ridge_shared_fold, _lgb_shared_fold):
        assert mod._rows_digest(a) == mod._rows_digest(a.copy())
        assert mod._rows_digest(a) != mod._rows_digest(b)
        assert len(mod._rows_digest(a)) == 16
