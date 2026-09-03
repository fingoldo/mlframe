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
    # NaN / null labels must map to code -1, matching the pandas branch's documented contract -- plain
    # np.unique(..., return_inverse=True) gives NaN its own real code instead (NaN sorts to the end of a
    # float array and forms its own unique group), which would silently treat a NaN group as real data in
    # this fallback-only (no-pandas) path.
    if arr.dtype.kind == "f":
        nan_mask = np.isnan(arr)
    elif arr.dtype.kind in ("U", "S", "O"):
        nan_mask = np.array([v is None or (isinstance(v, float) and v != v) for v in arr], dtype=bool)
    else:
        nan_mask = np.zeros(arr.shape[0], dtype=bool)
    valid = ~nan_mask
    uniq, valid_codes = np.unique(arr[valid], return_inverse=True)
    codes = np.full(arr.shape[0], -1, dtype=np.int64)
    codes[valid] = valid_codes
    return codes, list(uniq)


def pct(x: Optional[float]) -> str:
    """Format a fraction as a signed percentage for the rendered text block (``None`` -> ``"n/a"``)."""
    return "n/a" if x is None else f"{100.0 * x:+.2f}%"


def num(x: Optional[float]) -> str:
    """Format a metric (RMSE, gap) at 6 significant digits for the rendered text block (``None`` -> ``"n/a"``)."""
    return "n/a" if x is None else f"{x:.6g}"
