"""The registry entry shared by the CUDA kernels that take byte codes when every cardinality is at most 127 and 32-bit integer codes otherwise."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def registry_entry(compile_for: Callable[[Any, bool], Any]) -> Callable[[Callable[..., Any]], Callable[[Any, Any], Any]]:
    """Decorator for the ``kernel(cp, real)`` entry the register-pressure registry compiles through: the decorated stub keeps its name and docstring, and calls ``compile_for`` with
    ``narrow`` set when ``real`` is the byte-code dtype (uint8), so the byte-or-int32 choice lives here once."""

    def decorate(stub: Callable[..., Any]) -> Callable[[Any, Any], Any]:
        """Replace ``stub`` by the shared entry, keeping its identity for the registry."""

        def kernel(cp: Any, real: Any) -> Any:
            """The compiled kernel for codes of dtype ``real``."""
            return compile_for(cp, np.dtype(real) == np.uint8)

        kernel.__name__ = stub.__name__
        kernel.__qualname__ = stub.__qualname__
        kernel.__doc__ = stub.__doc__
        kernel.__module__ = stub.__module__
        return kernel

    return decorate
