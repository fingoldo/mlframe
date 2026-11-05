"""Element types of a CUDA kernel source (CUDA.md rule 3): ``{REAL}`` for values read from the operands, ``{ACC}`` for sums and statistics, ``{EDGE}`` for the bin edges a value is compared with.

A kernel source never names ``double`` or ``float``; the module that compiles it substitutes the two placeholders. ``{REAL}`` follows the dtype of the arrays the launch hands over
(float32 by default, the crit dtype), so a launch with float32 operands never runs a float64 kernel and never reads past the end of a buffer; ``{ACC}`` is float64 where a statistic
is compared with a threshold or summed over many terms, and ``{REAL}`` where it is not.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

_CTYPE = {np.dtype(np.float32): "float", np.dtype(np.float64): "double"}


def ctype_of(dtype: Any) -> str:
    """The CUDA type name of a float32 or float64 dtype."""
    try:
        return _CTYPE[np.dtype(dtype)]
    except KeyError:
        raise ValueError(f"no CUDA floating type for dtype {np.dtype(dtype)}") from None


def render(src: str, real: Optional[Any] = None, acc: Optional[Any] = np.float64, edge: Optional[Any] = None) -> str:
    """``src`` with ``{REAL}`` / ``{ACC}`` / ``{EDGE}`` replaced by the CUDA types of ``real`` / ``acc`` / ``edge``; a placeholder whose type is ``None`` is left for the caller (a module compiled once per value type).

    ``{EDGE}`` is the type of the bin edges a value is compared with: float64 where the comparison must see the double edge, float32 where the edges were rounded up to float32 for float32 values."""
    if real is not None:
        src = src.replace("{REAL}", ctype_of(real))
    if acc is not None:
        src = src.replace("{ACC}", ctype_of(acc))
    if edge is not None:
        src = src.replace("{EDGE}", ctype_of(edge))
    return src


__all__ = ["ctype_of", "render"]
