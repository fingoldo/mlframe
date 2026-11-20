"""The resident device caches are keyed by CUDA device id, so a device switch never returns an array held on another device."""
import sys
import types

import numpy as np

from mlframe.feature_selection.filters.info_theory import _cmi_resident_cache as rc


def _fake_cupy(state):
    """A stand-in cupy whose current device is state['dev'] and whose asarray records uploads."""
    mod = types.ModuleType("cupy")
    mod.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(getDevice=lambda: state["dev"]))

    def asarray(a):
        """Record the upload with the device it landed on."""
        state["uploads"].append(state["dev"])
        return ("dev", state["dev"], id(a))

    mod.asarray = asarray
    return mod


def test_resident_upload_reuploads_after_device_switch(monkeypatch):
    """Same content, different current device: a second upload, not a cache hit."""
    state = {"dev": 0, "uploads": []}
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(state))
    rc.clear_cmi_resident_cache()
    arr = np.arange(10, dtype=np.int32)
    first = rc._resident_upload(arr, ("k", 1))
    assert rc._resident_upload(arr, ("k", 1)) is first
    state["dev"] = 1
    second = rc._resident_upload(arr, ("k", 1))
    assert second is not first
    assert state["uploads"] == [0, 1]
    rc.clear_cmi_resident_cache()
