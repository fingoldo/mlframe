"""A lenient ``load_mlframe_model`` call must not satisfy a later strict one from the cache (FOP4-1).

A cache hit returns before the version check, and the key did not include ``strict_version``. A strict caller could therefore receive an
object that a lenient caller loaded despite library-version drift.
"""

from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest

from mlframe.training import io as io_mod


def test_strict_load_is_not_served_from_a_lenient_cache_entry(tmp_path, monkeypatch):
    """After a lenient load of a drifted bundle, the strict load still raises the drift error."""
    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    io_mod._load_model_cache_clear()
    bundle = tmp_path / "test.dump"
    io_mod.save_mlframe_model(SimpleNamespace(payload=7), str(bundle), verbose=0)
    sidecar = Path(io_mod._meta_sidecar_path(str(bundle)))
    meta = orjson.loads(sidecar.read_bytes())
    meta["lib_versions"]["mlframe"] = "0.001-FAKE-OLD"
    sidecar.write_bytes(orjson.dumps(meta))

    assert io_mod.load_mlframe_model(str(bundle)).payload == 7
    with pytest.raises(ValueError, match="drift"):
        io_mod.load_mlframe_model(str(bundle), strict_version=True)
    io_mod._load_model_cache_clear()
