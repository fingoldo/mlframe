"""``load_mlframe_model`` opens the bundle once: the bytes it verifies are the bytes it unpickles.

It used to hash the file through one open and then reopen it by path to unpickle, so the file could change between the two. The digest check and
the load now share one handle.
"""

from __future__ import annotations

import builtins
import os
import pickle

import zstandard

from mlframe.training import io as io_mod


def _write_bundle(path, payload):
    """Write ``payload`` as a zstd-compressed pickle, the layout ``load_mlframe_model`` reads."""
    path.write_bytes(zstandard.ZstdCompressor().compress(pickle.dumps(payload)))


def test_a_verified_bundle_is_opened_once(tmp_path, monkeypatch):
    """The digest check and the unpickling read the same open file: one open of the bundle path, not two."""
    bundle = tmp_path / "model.zst"
    _write_bundle(bundle, {"a": 1})
    io_mod._write_save_meta_sidecar(str(bundle))
    target = os.path.abspath(str(bundle))
    real_open = builtins.open
    opened = []

    def _counting_open(file, *args, **kwargs):
        """Record each open of the bundle path."""
        if isinstance(file, (str, os.PathLike)) and os.path.abspath(os.fspath(file)) == target:
            opened.append(file)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _counting_open)
    assert io_mod.load_mlframe_model(str(bundle)) == {"a": 1}
    assert len(opened) == 1
