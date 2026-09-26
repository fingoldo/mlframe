"""What a cached model was trained on, so a rerun with different hyperparameters or data does not reuse it.

The suite's model cache is a ``.dump`` per model name, reloaded whenever it exists. It checked the feature schema and a
composite spec's digest, but not the estimator's hyperparameters, the targets, the row selection or the sample weights: a
rerun into the same directory with a new learning rate, a new label column or a new split silently served the old model.
The fingerprint digests those inputs; it is stamped on the dump when it is saved and compared when it is loaded.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Optional

import numpy as np
import logging

logger = logging.getLogger(__name__)

# The ``common_params`` arrays the fit reads besides the feature frame.
_FIT_ARRAYS = ("target", "train_target", "val_target", "train_idx", "val_idx", "sample_weight")
# Keys of ``model_params`` / ``common_params`` that describe the run's reporting or bookkeeping, not the fit.
_NOT_FIT_INPUTS = frozenset({"model", "plot_file", "model_name", "verbose", "trainset_features_stats", "use_cache", "model_path", "artifact_dir"})
_PRIMITIVES = (type(None), bool, int, float, str, bytes, np.integer, np.floating, np.bool_)


def _stable(value: Any, depth: int = 0) -> str:
    """A repr that is stable across processes: primitives and containers by value, arrays by content, objects by class."""
    if isinstance(value, _PRIMITIVES):
        return repr(value.item() if isinstance(value, np.generic) else value)
    if depth > 4:
        return type(value).__qualname__
    if isinstance(value, Mapping):
        return "{" + ",".join(f"{_stable(k, depth + 1)}:{_stable(v, depth + 1)}" for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))) + "}"
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        return type(value).__name__ + "[" + ",".join(_stable(v, depth + 1) for v in items) + "]"
    if isinstance(value, np.ndarray):
        return f"ndarray{value.shape}:{_array_digest(value)}"
    if hasattr(value, "get_params"):
        try:
            return type(value).__qualname__ + _stable(value.get_params(deep=False), depth + 1)
        except Exception as exc:  # an estimator whose get_params fails is identified by its class alone
            logger.debug("_stable: %s", exc, exc_info=True)
            return type(value).__qualname__
    return type(value).__qualname__


def _array_digest(a: Any) -> str:
    """Content digest of an array-like (numpy, pandas or polars), by dtype, shape and bytes."""
    arr = np.asarray(a.to_numpy() if hasattr(a, "to_numpy") else a)
    if arr.dtype == object:
        arr = arr.astype(str)
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{arr.dtype.str}{arr.shape}".encode())
    h.update(np.ascontiguousarray(arr).view(np.uint8).data)
    return h.hexdigest()


def training_fingerprint(model_obj: Any, model_params: Mapping[str, Any], common_params: Mapping[str, Any]) -> str:
    """Digest of the estimator's hyperparameters, the other fit parameters, the targets, rows and weights, and the frame's columns."""
    h = hashlib.blake2b(digest_size=16)
    h.update(_stable(model_obj).encode())
    for source in (model_params, common_params):
        for key in sorted(k for k in source if k not in _NOT_FIT_INPUTS and k not in _FIT_ARRAYS and not k.endswith("_df")):
            h.update(f"|{key}={_stable(source[key])}".encode())
    for key in _FIT_ARRAYS:
        value = common_params.get(key)
        h.update(f"|{key}:{'none' if value is None else _array_digest(value)}".encode())
    frame = common_params.get("train_df")
    if frame is not None and hasattr(frame, "columns"):
        h.update(f"|train_df{getattr(frame, 'shape', None)}:{list(map(str, frame.columns))}".encode())
    return h.hexdigest()


def training_fingerprint_mismatch(loaded_model: Any, want: Optional[str]) -> Optional[str]:
    """Why a cached dump was trained on something else than ``want`` describes (``None`` when it matches)."""
    if want is None:
        return None
    have = getattr(loaded_model, "training_fingerprint_", None)
    if have is None:
        return "the dump records no training fingerprint, so the hyperparameters and data it was trained on are unknown"
    if have != want:
        return "the hyperparameters, targets, rows or weights changed since the dump was trained"
    return None
