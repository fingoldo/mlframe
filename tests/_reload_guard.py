"""Undo ``importlib.reload`` of mlframe modules at the end of the test that performed it.

A reload re-executes the module and mints NEW class and function objects, while every module that imported the old
ones keeps them. A model trained later in the process then pickles those classes by value (dill ``_create_type``),
which the restricted model loader refuses, and ``isinstance`` and class-level caches disagree far from the cause.
Reloading a second time to "restore" does not help: it mints a third set. Only putting the original bindings back does.
"""

from __future__ import annotations

import contextlib
import importlib
from types import ModuleType
from typing import Callable, Iterator, Sequence


@contextlib.contextmanager
def reload_guard(prefixes: Sequence[str] = ("mlframe",)) -> Iterator["dict[str, tuple[ModuleType, dict]]"]:
    """Record the first-seen bindings of every reloaded module under ``prefixes`` and restore them on exit.

    Yields the ``{module name: (module, original namespace)}`` record, so a caller can assert what was reloaded.
    """
    real_reload: Callable[[ModuleType], ModuleType] = importlib.reload
    saved: dict[str, tuple[ModuleType, dict]] = {}

    def guarded(module: ModuleType) -> ModuleType:
        """Snapshot the module namespace once, then reload."""
        name = getattr(module, "__name__", "")
        if name not in saved and any(name == p or name.startswith(p + ".") for p in prefixes):
            saved[name] = (module, dict(module.__dict__))
        return real_reload(module)

    importlib.reload = guarded  # type: ignore[assignment]
    try:
        yield saved
    finally:
        importlib.reload = real_reload  # type: ignore[assignment]
        for module, namespace in saved.values():
            module.__dict__.clear()
            module.__dict__.update(namespace)
