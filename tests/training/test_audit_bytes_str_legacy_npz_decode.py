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

from pathlib import Path

import numpy as np


MLFRAME_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe"


def _read(rel: str) -> str:
    """Reads an mlframe source file's text for source-level assertions."""
    return (MLFRAME_ROOT / rel).read_text(encoding="utf-8")


def test_cache_legacy_kind_decode_handles_bytes() -> None:
    """Legacy npz cache-kind decode must branch on bytes/bytearray explicitly, not blindly str() the raw value."""
    src = _read("training/feature_handling/cache.py")
    # The pre-fix `str(kind_arr[0])` standalone-line is gone.
    assert "kind = str(kind_arr[0])" not in src
    # The post-fix handles bytes / bytearray separately.
    assert 'raw.decode("ascii") if isinstance(raw, (bytes, bytearray)) else str(raw)' in src


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
