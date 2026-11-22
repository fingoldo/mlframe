"""Audit 2026-11-20 finding 31-F7: discovery's capture of a guarded module does not import it.

The module's docstring forbids discovery from changing the process; the capture used to import cupy when it was not loaded. The test removes
cupy from sys.modules, captures it, and checks it is still not loaded.
"""

import sys

from mlframe.system.kernel_tuning_cache import _discovery


def test_capture_of_an_unloaded_guarded_module_does_not_import_it(monkeypatch):
    """_capture returns the not-loaded sentinel and leaves the module out of sys.modules."""
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    state = _discovery._capture("cupy")
    assert state is _discovery._NOT_LOADED
    assert "cupy" not in sys.modules
