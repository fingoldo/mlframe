"""Regression sensor for A5 P2 #10.

Pre-fix ``load_mlframe_model`` re-ran ``_SafeUnpickler(zstd.stream_reader(...))`` on every call. In a warm-prediction service (one process, many requests) this is the dominant per-request cost. The fix adds an mtime-keyed module-level cache so repeat loads of the same path return the cached object directly while a file overwrite (new mtime) invalidates automatically. Size-gated to 2 GB on-disk per CLAUDE.md (CatBoost on rich suites can exceed this).
"""

from __future__ import annotations

import io as _stdio
import os
from types import SimpleNamespace

import pytest

dill = pytest.importorskip("dill")
zstd = pytest.importorskip("zstandard")


def _write_dummy_bundle(path: str, payload) -> None:
    cctx = zstd.ZstdCompressor()
    with open(path, "wb") as f:
        with cctx.stream_writer(f) as zw:
            dill.dump(payload, zw)


def test_load_mlframe_model_caches_by_path_and_mtime(tmp_path, monkeypatch):
    from mlframe.training import io as io_mod
    io_mod._load_model_cache_clear()
    bundle = tmp_path / "m.dump"
    obj = SimpleNamespace(score=1.5, fname="m")
    _write_dummy_bundle(str(bundle), obj)
    first = io_mod.load_mlframe_model(str(bundle), safe=False)
    assert first is not None
    second = io_mod.load_mlframe_model(str(bundle), safe=False)
    assert second is first, "warm cache must return the SAME object (id-equal) on repeat load"


def test_load_mlframe_model_invalidates_on_mtime_change(tmp_path):
    from mlframe.training import io as io_mod
    io_mod._load_model_cache_clear()
    bundle = tmp_path / "m.dump"
    _write_dummy_bundle(str(bundle), SimpleNamespace(version=1))
    first = io_mod.load_mlframe_model(str(bundle), safe=False)
    assert first is not None and first.version == 1
    new_mtime = os.stat(str(bundle)).st_mtime_ns + 1_000_000_000
    _write_dummy_bundle(str(bundle), SimpleNamespace(version=2))
    os.utime(str(bundle), ns=(new_mtime, new_mtime))
    second = io_mod.load_mlframe_model(str(bundle), safe=False)
    assert second is not first, "mtime change must invalidate the cached entry"
    assert second.version == 2


def test_load_mlframe_model_size_gate_skips_cache(tmp_path, monkeypatch):
    from mlframe.training import io as io_mod
    io_mod._load_model_cache_clear()
    bundle = tmp_path / "m.dump"
    _write_dummy_bundle(str(bundle), SimpleNamespace(big=False))
    monkeypatch.setenv("MLFRAME_LOAD_MODEL_CACHE_MAX_MB", "0")
    first = io_mod.load_mlframe_model(str(bundle), safe=False)
    second = io_mod.load_mlframe_model(str(bundle), safe=False)
    assert first is not None and second is not None
    assert second is not first, "MAX_MB=0 must disable the cache so each load returns a fresh object"
