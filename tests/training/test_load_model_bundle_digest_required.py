"""A model bundle is unpickled only when its bytes match the sha256 recorded in its save sidecar (DESER-02).

Before the fix ``load_mlframe_model`` checked only library-version drift in the sidecar, so a tampered bundle was unpickled.
"""

import pickle

import pytest
import zstandard

from mlframe.training import io as io_mod


@pytest.fixture(autouse=True)
def _no_unverified_opt_out(monkeypatch):
    """Start from the default: the unverified-load opt-out is unset."""
    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)


def _write_bundle(path, payload):
    """Write ``payload`` as a zstd-compressed pickle, the layout ``load_mlframe_model`` reads."""
    compressor = zstandard.ZstdCompressor()
    path.write_bytes(compressor.compress(pickle.dumps(payload)))


def test_bundle_without_a_recorded_digest_is_refused(tmp_path):
    """A bundle with no save sidecar is not unpickled by default."""
    bundle = tmp_path / "model.zst"
    _write_bundle(bundle, {"a": 1})
    with pytest.raises(RuntimeError, match="no sha256 recorded"):
        io_mod.load_mlframe_model(str(bundle))


def test_bundle_with_a_matching_digest_loads(tmp_path):
    """Once the save sidecar records the bundle's digest, the bundle loads."""
    bundle = tmp_path / "model.zst"
    _write_bundle(bundle, {"a": 1})
    io_mod._write_save_meta_sidecar(str(bundle))
    assert io_mod.load_mlframe_model(str(bundle)) == {"a": 1}


def test_tampered_bundle_is_refused_even_with_a_sidecar(tmp_path):
    """Changing the bundle bytes after the digest was recorded makes the load fail before unpickling."""
    bundle = tmp_path / "model.zst"
    _write_bundle(bundle, {"a": 1})
    io_mod._write_save_meta_sidecar(str(bundle))
    _write_bundle(bundle, {"a": 2})
    with pytest.raises(RuntimeError, match="does not match"):
        io_mod.load_mlframe_model(str(bundle))


def test_opt_out_loads_a_bundle_without_a_digest(tmp_path, monkeypatch):
    """The explicit opt-out keeps the legacy path reachable."""
    bundle = tmp_path / "model.zst"
    _write_bundle(bundle, {"a": 1})
    monkeypatch.setenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", "1")
    assert io_mod.load_mlframe_model(str(bundle)) == {"a": 1}
