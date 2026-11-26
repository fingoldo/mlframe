"""The trend renderer's content digest hashes the array without copying it, and still accepts datetime64 arrays."""

import hashlib

import numpy as np

from mlframe.reporting.renderers._trend import _content_digest


def test_content_digest_matches_the_tobytes_digest_for_numeric_arrays():
    """Feeding the uint8 view must give the same digest the old tobytes() form gave, so no stored key moves."""
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2]
    expected = hashlib.blake2b(np.ascontiguousarray(arr).tobytes(), digest_size=16).digest()
    assert _content_digest(arr) == expected


def test_content_digest_accepts_datetime64_arrays():
    """datetime64 has no buffer-protocol format, so a plain .data feed raises here; the uint8 view must not."""
    arr = np.array([1, 2, 3], dtype="datetime64[ns]")
    assert len(_content_digest(arr)) == 16
