"""SEC3 regression: DiskCache.get must fail CLOSED on a cache file with no .sha256 sidecar.

Previously DiskCache.get called safe_load(..., allow_unverified=True), forcing the fail-OPEN path: a payload planted in the cache dir with
no sidecar was unpickled silently. The fix removes allow_unverified=True so a missing sidecar is refused by default; the only opt-in is the
MLFRAME_ALLOW_UNVERIFIED_PICKLE env var.
"""

import pickle


from mlframe.utils.disk_cache import DiskCache


def _plant_unverified_entry(cache: DiskCache, key: str, value) -> None:
    """Write a cache payload WITHOUT its .sha256 sidecar (simulates a planted/untrusted file)."""
    path = cache._key_path(key)
    with open(path, "wb") as f:
        pickle.dump(value, f)
    sidecar = path.parent / (path.name + ".sha256")
    if sidecar.exists():
        sidecar.unlink()


def test_missing_sidecar_refused_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    cache = DiskCache(tmp_path)
    _plant_unverified_entry(cache, "k", {"planted": 1})

    # Fail-closed: refused, reported as a miss, no unpickling of the un-sidecar'd payload.
    assert cache.get("k") is None


def test_missing_sidecar_allowed_with_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", "1")
    cache = DiskCache(tmp_path)
    _plant_unverified_entry(cache, "k", {"planted": 1})

    assert cache.get("k") == {"planted": 1}


def test_legit_put_roundtrips(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    cache = DiskCache(tmp_path)
    cache.put("k", {"ok": 2})
    assert cache.get("k") == {"ok": 2}
