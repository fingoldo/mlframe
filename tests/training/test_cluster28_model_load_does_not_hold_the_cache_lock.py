"""Cluster 28 regression: a slow model load does not block cache hits for other models (audit 2026-11-20, finding 28-F27)."""

import os
import threading

import mlframe.training.io as tio


def test_a_cache_hit_is_served_while_another_model_is_still_loading(tmp_path, monkeypatch):
    """Thread A blocks inside the bundle read of file A; the main thread still gets the cached model of file B immediately."""
    file_a = tmp_path / "a.bundle"
    file_b = tmp_path / "b.bundle"
    file_a.write_bytes(b"x")
    file_b.write_bytes(b"y")
    cached_b = object()
    key_b = (os.path.abspath(file_b), os.stat(file_b).st_mtime_ns, True, False)
    tio._load_model_cache_clear()
    tio._LOAD_MODEL_CACHE[key_b] = cached_b

    entered, release = threading.Event(), threading.Event()

    def slow_open(file):
        """Block like a long read until the test releases it."""
        entered.set()
        release.wait(timeout=30)
        return None

    monkeypatch.setattr(tio, "_open_verified_bundle", slow_open)
    worker = threading.Thread(target=lambda: tio.load_mlframe_model(str(file_a)), daemon=True)
    worker.start()
    try:
        assert entered.wait(timeout=30), "the slow load did not start"
        result = {}
        getter = threading.Thread(target=lambda: result.update(model=tio.load_mlframe_model(str(file_b))), daemon=True)
        getter.start()
        getter.join(timeout=10)
        assert not getter.is_alive(), "the cache hit was blocked behind the slow load"
        assert result["model"] is cached_b
    finally:
        release.set()
        worker.join(timeout=30)
        tio._load_model_cache_clear()
