"""Canonical JSON bytes for hashing: deterministic, key-order independent, never raises, never merges distinct inputs.

Plain string-keyed finite JSON keeps the exact bytes ``orjson.dumps(obj, default=str, option=OPT_SORT_KEYS)`` produced, so existing digests of
ordinary configs are unchanged. Only the shapes orjson cannot represent faithfully are rewritten into tagged forms:
non-finite floats, integers beyond 64 bits, non-string dict keys, sets and numpy values.
"""

from __future__ import annotations

import json
import math
from typing import Any

try:
    import orjson
except ImportError:  # pragma: no cover - orjson is an optional accelerator
    orjson = None  # type: ignore[assignment]

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_MAX_DEPTH = 64


def _key_text(key: Any) -> str:
    """Stringify a dict key with a type tag for non-str keys so ``1`` and ``"1"`` stay distinct."""
    if isinstance(key, str):
        return key
    return f"\x00{type(key).__name__}:{key!r}"


def _canon(obj: Any, depth: int) -> Any:
    """Rewrite ``obj`` into a JSON-safe tree whose dict keys are all str."""
    if depth > _MAX_DEPTH:
        return {"__deep__": type(obj).__name__}
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, int):
        return obj if _INT64_MIN <= obj <= _INT64_MAX else {"__bigint__": str(obj)}
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else {"__float__": repr(obj)}
    if isinstance(obj, dict):
        return {_key_text(k): _canon(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v, depth + 1) for v in obj]
    if isinstance(obj, (set, frozenset)):
        items = [_canon(v, depth + 1) for v in obj]
        return {"__set__": sorted(items, key=lambda v: _dumps_tree(v))}
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return _canon(obj.tolist(), depth + 1)
        except (TypeError, ValueError):
            return str(obj)
    return str(obj)


def _dumps_tree(tree: Any) -> bytes:
    """Serialise an already canonical tree with sorted keys."""
    if orjson is not None:
        return orjson.dumps(tree, default=str, option=orjson.OPT_SORT_KEYS)
    return json.dumps(tree, sort_keys=True, default=str, allow_nan=False).encode("utf-8")


def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic JSON bytes of ``obj`` for digesting; key order never matters and distinct values keep distinct bytes."""
    try:
        return _dumps_tree(_canon(obj, 0))
    except (RecursionError, ValueError, TypeError, OverflowError):
        return b"\x00repr:" + repr(obj).encode("utf-8", "backslashreplace")
