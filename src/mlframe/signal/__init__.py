"""Signal-processing utilities: DTW alignment + helpers.

Name-shadow guard: this subpackage is named ``signal``. When code runs with
``src/mlframe`` itself on ``sys.path`` (e.g. cwd inside the package), a bare
``import signal`` from *anywhere* in the process -- including stdlib
``multiprocessing.resource_tracker`` (via joblib/loky) -- resolves to THIS
package instead of the standard library ``signal`` module, and then blows up on
``signal.SIGINT``. Renaming the subpackage is not viable (it is widely
imported), so we instead make this package a transparent superset of stdlib
``signal``: we load the real stdlib module by file path and re-export its public
names, so ``signal.SIGINT`` / ``signal.SIGTERM`` / etc. keep resolving even when
this package wins the import.
"""

import importlib.util as _ilu
import os as _os
import sys as _sys


def _load_stdlib_signal():
    """Load the standard-library ``signal`` module directly from its file, bypassing
    the name-resolution that would otherwise return this shadowing package."""
    stdlib_dir = _os.path.dirname(_os.__file__)
    path = _os.path.join(stdlib_dir, "signal.py")
    if not _os.path.exists(path):  # pragma: no cover - defensive; stdlib signal is a .py on CPython
        return None
    modname = "_mlframe_stdlib_signal"
    spec = _ilu.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        return None
    mod = _ilu.module_from_spec(spec)
    # stdlib ``signal`` calls ``enum._convert_`` during exec, which looks the module up in
    # ``sys.modules`` by name -- so it MUST be registered there before ``exec_module``.
    _prev = _sys.modules.get(modname)
    _sys.modules[modname] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:  # pragma: no cover - defensive
        return None
    finally:
        if _prev is None:
            _sys.modules.pop(modname, None)
        else:
            _sys.modules[modname] = _prev
    return mod


_stdlib_signal = _load_stdlib_signal()
if _stdlib_signal is not None:
    # Re-export every public name of stdlib ``signal`` (SIGINT, SIGTERM, signal(), etc.)
    # so this package behaves as a transparent superset when it shadows the stdlib module.
    for _name in dir(_stdlib_signal):
        if not _name.startswith("_") and _name not in globals():
            globals()[_name] = getattr(_stdlib_signal, _name)
    del _name
