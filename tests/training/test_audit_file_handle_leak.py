"""Wave 42 (2026-05-20): file-handle / OS-resource leak audit.

Result: 1 P1 fix. Codebase otherwise CLEAN -- the broader `with open(...)`
discipline is already followed throughout (~30 sites verified), and the
mkstemp+fdopen ownership case from wave 36 stays fixed across all 3 sites.

P1: training/feature_handling/cache.py:474 (_deserialize)
    np.load(path, mmap_mode="r") returns an NpzFile wrapping a zipfile + a
    file handle + mmap views. The prior code never called .close() on the
    NpzFile -- every successful FeatureCache read leaked one OS handle. On
    Windows this blocks later overwrite/eviction with PermissionError; on
    Linux long CV/RFECV loops eventually hit EMFILE ("too many open files").

Fix: use `with np_load_result as npz:` and materialise arrays via np.array(...)
before exiting so the caller gets owned buffers (mmap views would go invalid
on close).
"""

from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path

import numpy as np

MLFRAME_ROOT = Path(importlib.import_module("mlframe").__file__).parent


def _read(rel: str) -> str:
    """Reads an mlframe source file's text for source-level assertions."""
    return (MLFRAME_ROOT / rel).read_text(encoding="utf-8")


def test_deserialize_uses_with_block_on_npzfile() -> None:
    """Source-level: NpzFile path must close via a with-block, not leak."""
    src = _read("training/feature_handling/cache.py")
    # The fix replaces unrestricted `npz = np.load(...)` with a with-block
    # for the NpzFile branch.
    assert "with loaded as npz:" in src, "training/feature_handling/cache.py: NpzFile must be closed via a with-block."


def test_deserialize_materialises_arrays_before_close() -> None:
    """Arrays returned to the caller must NOT be mmap views (which die on close)."""
    src = _read("training/feature_handling/cache.py")
    # Three callsites: ndarray "value", legacy "value"/first-key fallback, csr.
    assert src.count('np.array(npz["value"])') >= 1
    assert 'np.array(npz["data"])' in src
    assert 'np.array(npz["indices"])' in src
    assert 'np.array(npz["indptr"])' in src


def test_deserialize_roundtrip_does_not_leak_handles_on_windows() -> None:
    """Behavioural: read the same cache file 100 times; final overwrite must succeed.

    The pre-fix bug: on Windows, each leaked NpzFile pins the underlying file.
    A later os.replace / write to the same path raises PermissionError because
    the file is still mapped. Post-fix, the with-block releases the handle so
    the overwrite path works.
    """
    from mlframe.training.feature_handling.cache import _serialize, _deserialize

    arr = np.arange(64, dtype=np.float64)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test_payload.npz")
        with open(path, "wb") as f:
            _serialize(value=arr, fileobj=f, allow_pickle=False)
        # Read many times.
        for _ in range(100):
            out = _deserialize(path, allow_pickle=False)
            np.testing.assert_array_equal(out, arr)
        # The handle-leak symptom: now overwrite the file. Pre-fix on Windows
        # this raised PermissionError because the NpzFile zipfile + mmaps were
        # still open. Post-fix: the with-block released the handle each iteration.
        with open(path, "wb") as f:
            _serialize(value=arr * 2, fileobj=f, allow_pickle=False)
        out2 = _deserialize(path, allow_pickle=False)
        np.testing.assert_array_equal(out2, arr * 2)


def test_deserialize_returned_array_is_owned_not_mmap_view() -> None:
    """The post-fix np.array(...) materialisation produces an OWNED buffer."""
    from mlframe.training.feature_handling.cache import _serialize, _deserialize

    arr = np.arange(32, dtype=np.float32)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "owned.npz")
        with open(path, "wb") as f:
            _serialize(value=arr, fileobj=f, allow_pickle=False)
        out = _deserialize(path, allow_pickle=False)
        # Owned means it's safe to mutate without touching the source file.
        # (mmap views would raise on write under "r" mode.)
        out[0] = 999.0  # Must not raise.
        # And the source file content is unchanged.
        out2 = _deserialize(path, allow_pickle=False)
        assert out2[0] == 0.0, "Mutation of caller's array must NOT touch the source file."
