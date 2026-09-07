"""One way to hand an array's bytes to a hash, shared by every cache key that does it.

`h.update(a.tobytes())` allocates a second full copy of the array purely to feed the hash. These are
cache-KEY computations -- a KeyBank fingerprint over X_train, a collinearity cache key over the
feature matrix, an RFECV signature over X and y -- so the copy is paid on every lookup including the
hits, on frames this package sizes in the tens of gigabytes. The sites were rewritten to feed the
existing buffer instead, which produces a byte-identical digest because `tobytes()` serialises in C
order and that is exactly what `ascontiguousarray` guarantees.

WHY THIS IS A MODULE AND NOT AN IDIOM. The rewrite landed as `np.ascontiguousarray(a).data`, spelled
out at roughly twenty call sites, and it is wrong: `.data` raises
`ValueError: cannot include dtype 'M' in a buffer` on datetime64 and timedelta64, which have no
buffer-protocol format. A datetime column in a training frame is not an edge case, and
`data_signature` crashed on any pandas frame containing one -- the guard above it excluded object and
string dtypes and had no reason to think of `M`. Twenty copies of a cache-key rule is the kind of
duplication `_dtype_canon` was extracted to end: the copies do not fail loudly when one is fixed and
the others are not.

Viewing the contiguous buffer as raw bytes first works for every dtype, including 0-d, empty,
structured and fixed-width string arrays, and leaves the digest unchanged.

Imports nothing from mlframe, so it is a leaf for any caller.
"""

from __future__ import annotations

import numpy as np

__all__ = ["array_buffer"]


def array_buffer(arr: np.ndarray) -> memoryview:
    """The bytes of *arr* in C order, without copying them.

    Equal byte for byte to `np.ascontiguousarray(arr).tobytes()`, so it is a drop-in replacement
    inside a hash and every existing digest is preserved. A non-contiguous input (a column view of a
    wider frame, an F-ordered block) is made contiguous first, which is the copy `tobytes()` would
    have made anyway -- the saving is on the contiguous case, which is the common one.
    """
    return np.ascontiguousarray(arr).view(np.uint8).data
