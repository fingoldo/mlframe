"""A CuPy ``RawModule`` compiled on first use and shared by every caller of the same holder.

Each GPU kernel file keeps one holder per kernel source, so the source and its compile options stay with the kernel they
belong to. The lock makes concurrent first calls compile once.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any, Optional

__all__ = ["LazyRawModule"]


class LazyRawModule:
    """Compile ``source`` with ``options`` the first time :meth:`get` is called, then return the same module."""

    def __init__(self, source: str, options: Sequence[str]) -> None:
        self._source = source
        self._options = tuple(options)
        self._lock = threading.Lock()
        self._module: Optional[Any] = None

    def get(self, cp: Any) -> Any:
        """The compiled module, compiled with ``cp.RawModule`` on the first call and returned unchanged after that."""
        with self._lock:
            if self._module is None:
                self._module = cp.RawModule(code=self._source, options=self._options)
            return self._module

    def __getstate__(self) -> dict:
        """Drop the lock and the compiled module: a device handle does not survive pickling, and the next ``get`` recompiles."""
        state = self.__dict__.copy()
        del state["_lock"]
        state["_module"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
