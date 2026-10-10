"""Thread-safe, compile-once cache for ``cupy.RawModule`` kernel sources, optionally specialised per design width."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple


class RawModuleCache:
    """Compiles one CUDA source lazily and keeps the module; with ``d`` given, one module per design width with ``-DD``/``-DNACC`` defined.

    The source is only remembered at construction; nothing is compiled until the first ``get``.
    """

    def __init__(self, src: str) -> None:
        self._src = src
        self._lock = threading.Lock()
        self._modules: Dict[Optional[int], Any] = {}

    def __getstate__(self) -> Dict[str, Any]:
        """Pickle only the source: the lock and the compiled modules are process-local and are rebuilt lazily."""
        return {"_src": self._src}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore from the source with a fresh lock and an empty module table."""
        self.__init__(state["_src"])  # type: ignore[misc]

    def get(self, cp: Any, d: Optional[int] = None) -> Any:
        """The compiled module, built on first use (per width ``d`` when given)."""
        with self._lock:
            mod = self._modules.get(d)
            if mod is None:
                options: Tuple[str, ...] = ("-std=c++14",)
                if d is not None:
                    options += (f"-DD={d}", f"-DNACC={d * (d + 1) // 2 + d}")
                mod = cp.RawModule(code=self._src, options=options)
                self._modules[d] = mod
            return mod
