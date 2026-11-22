"""Audit 2026-11-20 finding 29-F12: the incremental xxhash of an array equals the one-shot digest of head plus bytes.

The incremental form avoids two full copies of the array per lookup. Existing cache keys must not change, so the digests are compared
with the one-shot form the code used before. The test skips without xxhash.
"""

import numpy as np
import pytest

xxhash = pytest.importorskip("xxhash")

from mlframe.utils.disk_cache import hash_array_content


def test_incremental_digest_equals_the_one_shot_digest():
    """The array digest equals xxh3_128 of its shape/dtype header plus its bytes, computed in one call."""
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    head = f"{arr.shape}|{arr.dtype.str}|".encode("ascii")
    expected = str(xxhash.xxh3_128_hexdigest(head + arr.tobytes()))
    assert hash_array_content(arr) == expected
