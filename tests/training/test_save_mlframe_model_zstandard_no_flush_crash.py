"""Regression: save_mlframe_model used to crash on Windows with zstandard threads=-1.

zstandard.ZstdCompressor(threads=-1).stream_writer(f) as a context manager closes
the wrapped file on __exit__ (background flush thread hands the descriptor back
in a closed state). The previous atomic_write_bytes implementation called
f.flush() + os.fsync(f.fileno()) AFTER the inner with-block exited, which raised
``ValueError: I/O operation on closed file`` on every save.

The fix passes ``closefd=False`` to stream_writer so the outer atomic_write_bytes
keeps ownership of the descriptor through the fsync-before-rename window.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np


def test_save_load_no_closed_file_crash(tmp_path):
    """save_mlframe_model must round-trip without raising on closed fd."""
    from mlframe.training.io import save_mlframe_model, load_mlframe_model

    obj = SimpleNamespace(
        arr=np.arange(32, dtype=np.float64),
        meta={"k": 1, "list": [1, 2, 3]},
        label="zstd-threads-minusone",
    )
    path = str(tmp_path / "regression.zst")

    # Pre-fix: ZstdCompressor(threads=-1).stream_writer(f).__exit__ closed f, then
    # atomic_write_bytes(f).flush() raised ValueError("I/O operation on closed file").
    # save_mlframe_model swallows the exception and returns False — assert True.
    ok = save_mlframe_model(obj, path, verbose=0)
    assert ok is True, "save_mlframe_model returned False — flush-after-close regression"
    assert os.path.exists(path), "atomic_write_bytes did not produce target file"
    assert os.path.getsize(path) > 0, "target file is empty"

    loaded = load_mlframe_model(path, safe=False)
    assert loaded is not None
    np.testing.assert_array_equal(loaded.arr, obj.arr)
    assert loaded.meta == obj.meta
    assert loaded.label == obj.label


def test_save_explicit_threads_minus_one(tmp_path):
    """Belt-and-suspenders: pin threads=-1 in zstd_kwargs explicitly."""
    from mlframe.training.io import save_mlframe_model, load_mlframe_model

    obj = SimpleNamespace(payload=np.zeros((128,), dtype=np.float32))
    path = str(tmp_path / "explicit_threads.zst")
    ok = save_mlframe_model(
        obj,
        path,
        zstd_kwargs=dict(level=3, write_checksum=True, write_content_size=True, threads=-1),
        verbose=0,
    )
    assert ok is True
    loaded = load_mlframe_model(path, safe=False)
    assert loaded is not None
    np.testing.assert_array_equal(loaded.payload, obj.payload)


def test_save_single_thread_still_works(tmp_path):
    """Non-threaded path must keep working after the closefd=False fix."""
    from mlframe.training.io import save_mlframe_model, load_mlframe_model

    obj = SimpleNamespace(s="single-thread")
    path = str(tmp_path / "single.zst")
    ok = save_mlframe_model(
        obj,
        path,
        zstd_kwargs=dict(level=3, write_checksum=True, write_content_size=True, threads=0),
        verbose=0,
    )
    assert ok is True
    loaded = load_mlframe_model(path, safe=False)
    assert loaded is not None
    assert loaded.s == "single-thread"
