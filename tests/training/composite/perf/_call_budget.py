"""Count calls to named primitives across every module that imported them, for call-budget tests.

A primitive is usually called through a ``from x import f`` binding in the caller's module, so patching ``x.f`` alone misses
it. :class:`CallBudget` replaces the function object wherever an ``mlframe`` module (or the defining module) holds it, counts
the calls, and restores every binding on exit.
"""

from __future__ import annotations

import functools
import sys
from typing import Any, Callable


class CallBudget:
    """Context manager: ``with CallBudget({"name": func}) as calls: ...`` then ``calls["name"]`` is the call count."""

    def __init__(self, targets: dict[str, Callable[..., Any]], owners: dict[str, tuple[Any, str]] | None = None) -> None:
        self._targets = targets
        self._owners = owners or {}  # name -> (class, attribute) for methods, patched on the class only
        self._patched: list[tuple[Any, str, Any]] = []
        self.calls: dict[str, int] = {k: 0 for k in targets}

    def _counting(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        """``fn`` wrapped to increment ``self.calls[name]``."""
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            """Count, then delegate."""
            self.calls[name] += 1
            return fn(*args, **kwargs)
        return wrapper

    def __enter__(self) -> dict[str, int]:
        for name, fn in self._targets.items():
            spy = self._counting(name, fn)
            if name in self._owners:
                owner, attr = self._owners[name]
                self._patched.append((owner, attr, owner.__dict__[attr]))
                setattr(owner, attr, spy)
                continue
            for mod in list(sys.modules.values()):
                modname = getattr(mod, "__name__", "") or ""
                if not (modname.startswith("mlframe") or modname == getattr(fn, "__module__", None)):
                    continue
                for attr, val in list(vars(mod).items()):
                    if val is fn:
                        self._patched.append((mod, attr, val))
                        setattr(mod, attr, spy)
        return self.calls

    def __exit__(self, *exc: Any) -> None:
        for owner, attr, val in reversed(self._patched):
            setattr(owner, attr, val)
        self._patched.clear()
