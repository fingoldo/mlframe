"""Wave 77 (2026-05-21): bytes-vs-str confusion in legacy npz cache decode.

Audit class: Python 3 strictly separates b"foo" from "foo". Audit found 1 P2:
legacy-format npz reader's `str(kind_arr[0])` branch in
training/feature_handling/cache.py:514 would misdecode if any historical
writer ever serialised `kind` as a bytes-typed object array
(`str(b"ndarray") == "b'ndarray'"` -- does NOT equal "ndarray").

Fix: handle bytes/bytearray separately via .decode("ascii") before falling
back to str().

Everywhere else (cache key construction, fingerprint digests, recurrent
prediction cache, RFECV signatures, kernel_tuning_cache) bytes are confined
to hashlib.update()/int.from_bytes() and exit to str via .hexdigest()
consistently on both write and read sides.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_cache_legacy_kind_decode_handles_bytes(tmp_path) -> None:
    """A legacy npz whose ``kind`` is a bytes-typed object array is read back as the stored ndarray, and an unknown bytes kind is named without the ``b'..'`` repr."""
    from mlframe.training.feature_handling.cache import _deserialize

    value = np.arange(12, dtype=np.float32).reshape(3, 4)
    for label, kind in (("bytes", b"ndarray"), ("str", "ndarray")):
        path = tmp_path / f"legacy_{label}.npz"
        np.savez(path, kind=np.array([kind], dtype=object), value=value)
        loaded = _deserialize(str(path), allow_pickle=True)
        np.testing.assert_array_equal(loaded, value)
        assert loaded.dtype == np.float32

    unknown = tmp_path / "legacy_unknown.npz"
    np.savez(unknown, kind=np.array([b"mystery"], dtype=object), value=value)
    with pytest.raises(ValueError, match=r"unknown serialised kind 'mystery'"):
        _deserialize(str(unknown), allow_pickle=True)


def test_legacy_bytes_kind_decode_path_returns_correct_string() -> None:
    """Document the invariant: bytes-kind in legacy npz must decode to 'ndarray',
    not to the repr 'b\\'ndarray\\''."""
    # Simulate the legacy object-dtype branch.
    kind_arr = np.array([b"ndarray"], dtype=object)
    raw = kind_arr[0]
    # Pre-fix: str(raw) == "b'ndarray'"
    assert str(raw) == "b'ndarray'"  # documents the bug
    # Post-fix: bytes branch decodes correctly.
    decoded = raw.decode("ascii") if isinstance(raw, (bytes, bytearray)) else str(raw)
    assert decoded == "ndarray"


def test_str_kind_path_still_works_for_legacy_str_object_arrays() -> None:
    """The post-fix must not regress the str-typed legacy object-array path."""
    kind_arr = np.array(["ndarray"], dtype=object)
    raw = kind_arr[0]
    decoded = raw.decode("ascii") if isinstance(raw, (bytes, bytearray)) else str(raw)
    assert decoded == "ndarray"
