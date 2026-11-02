"""Run a function once, right after a named third-party module has been imported (or at once when it already was).

Used to patch a library only for processes that actually load it, instead of importing the library at ``import mlframe`` just to patch it.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import threading
from typing import Callable, Dict, List

_LOCK = threading.Lock()
_PENDING: Dict[str, List[Callable[[], None]]] = {}
_FINDER: "_PostImportFinder | None" = None


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that resolves a watched module through the remaining finders and runs its callbacks after the module executed."""

    def find_spec(self, fullname, path=None, target=None):
        """Return a spec whose loader runs the callbacks once ``fullname`` has executed; ``None`` for everything not watched."""
        if fullname not in _PENDING:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _CallbackLoader(spec.loader, fullname)
                return spec
        return None


class _CallbackLoader(importlib.abc.Loader):
    """Wraps a loader so the watched module's callbacks run after its ``exec_module``."""

    def __init__(self, inner, fullname: str) -> None:
        self._inner = inner
        self._fullname = fullname

    def create_module(self, spec):
        """Delegate module creation to the wrapped loader."""
        return self._inner.create_module(spec) if hasattr(self._inner, "create_module") else None

    def exec_module(self, module) -> None:
        """Execute the module, then run (and clear) the callbacks registered for it."""
        self._inner.exec_module(module)
        with _LOCK:
            callbacks = _PENDING.pop(self._fullname, [])
        for callback in callbacks:
            callback()

    def __getattr__(self, name):
        """Forward anything else (``get_code``, ``is_package`` ...) to the wrapped loader."""
        return getattr(self._inner, name)


def run_after_import(module_name: str, callback: Callable[[], None]) -> None:
    """Call ``callback()`` right after ``module_name`` is imported, or immediately when it is already in ``sys.modules``."""
    global _FINDER
    with _LOCK:
        if module_name not in sys.modules:
            _PENDING.setdefault(module_name, []).append(callback)
            if _FINDER is None:
                _FINDER = _PostImportFinder()
                sys.meta_path.insert(0, _FINDER)
            return
    callback()
