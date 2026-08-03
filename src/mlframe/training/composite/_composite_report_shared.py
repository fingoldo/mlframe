"""Shared tiny formatting/coercion helpers for the composite-training report modules
(_value_report.py, _moe_gate.py, _regime_headroom.py): independently duplicated across those
modules, consolidated here so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import pandas as pd

    _HAVE_PANDAS = True
except ImportError:  # pragma: no cover - pandas is a hard dep in practice
    _HAVE_PANDAS = False


def as1d(a: Any) -> np.ndarray:
    """Coerce any array-like (list, pandas Series, polars Series, ndarray) to a flat float64 ndarray for the reduction kernels."""
    return np.asarray(a, dtype=np.float64).reshape(-1)


def to_native(v: Any) -> Any:
    """JSON-native group label (numpy scalars -> python; bytes -> ascii; else str)."""
    if v is None or isinstance(v, (str, bool, int, float)):
        return v
    if isinstance(v, np.generic):
        n = v.item()
        return n if isinstance(n, (str, bool, int, float)) else str(n)
    if isinstance(v, bytes):
        return v.decode("ascii", "replace")
    return str(v)


def ascii_safe(s: Any) -> str:
    """Force ASCII for printed/logged strings (cp1251 crashes on non-ASCII)."""
    return str(s).encode("ascii", "replace").decode("ascii")


def factorize(group_ids: Any) -> tuple[np.ndarray, list]:
    """(codes, unique_labels). NaN / null labels map to code -1 (excluded downstream)."""
    if _HAVE_PANDAS:
        codes, uniq = pd.factorize(np.asarray(group_ids), sort=False)
        return np.asarray(codes, dtype=np.int64), list(uniq)
    arr = np.asarray(group_ids)
    uniq, codes = np.unique(arr, return_inverse=True)
    return np.asarray(codes, dtype=np.int64), list(uniq)


def pct(x: Optional[float]) -> str:
    """Format a fraction as a signed percentage for the rendered text block (``None`` -> ``"n/a"``)."""
    return "n/a" if x is None else f"{100.0 * x:+.2f}%"


def num(x: Optional[float]) -> str:
    """Format a metric (RMSE, gap) at 6 significant digits for the rendered text block (``None`` -> ``"n/a"``)."""
    return "n/a" if x is None else f"{x:.6g}"
