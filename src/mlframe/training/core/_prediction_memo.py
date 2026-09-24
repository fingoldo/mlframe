"""Predictions of one model on one frame, computed once per composite post-processing phase.

The wrap pass predicts every composite on val and test; the cross-target ensemble report then predicts the ensemble on
the same frames, the MoE gate predicts that ensemble and the raw model on val again, and the refit pre-screen predicts
each component on val once more. Each call also re-runs the shim's pre-pipeline over the whole frame. Within the phase
the models are fitted and the frames fixed, so a prediction is memoised on (model, frame) and handed back on the next
request.

Entries hold strong references to the model and the frame, so an ``id()`` cannot be reused by another object while the
memo is alive; the memo is dropped when the phase exits. Outside an active phase every call predicts as before.
"""

from __future__ import annotations

import contextlib
import functools
import threading
from typing import Any, Callable, Iterator, cast

import numpy as np

_LOCK = threading.Lock()
_ACTIVE: dict | None = None


@contextlib.contextmanager
def prediction_memo() -> Iterator[None]:
    """Memoise predictions for the duration of the block; a nested block reuses the outer memo."""
    global _ACTIVE
    with _LOCK:
        owner = _ACTIVE is None
        if owner:
            _ACTIVE = {}
    try:
        yield
    finally:
        if owner:
            with _LOCK:
                _ACTIVE = None


def with_prediction_memo(fn: Callable) -> Callable:
    """Run ``fn`` with the prediction memo active, so every predict it triggers on a (model, frame) pair runs once."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        """Call ``fn`` inside :func:`prediction_memo`."""
        with prediction_memo():
            return fn(*args, **kwargs)

    return wrapper


def memo_predict(model: Any, frame: Any) -> np.ndarray:
    """``model.predict(frame)`` as a flat float64 array, served from the active memo when this pair was predicted before.

    Callers receive a copy, so a caller that edits its predictions cannot change what the next caller sees.
    """
    memo = _ACTIVE
    if memo is None:
        return np.asarray(model.predict(frame), dtype=np.float64).reshape(-1)
    key = (id(model), id(frame), getattr(frame, "shape", None))
    with _LOCK:
        hit = memo.get(key)
    if hit is not None and hit[0] is model and hit[1] is frame:
        return cast(np.ndarray, hit[2].copy())
    preds = np.asarray(model.predict(frame), dtype=np.float64).reshape(-1)
    with _LOCK:
        memo[key] = (model, frame, preds)
    return cast(np.ndarray, preds.copy())


def memo_transform(pipeline: Any, frame: Any, compute: Callable[[], Any], tag: str = "") -> Any:
    """``compute()`` - a fitted pipeline's transform of ``frame`` - served from the active memo for a repeated (pipeline, frame).

    Every wrapper predict re-applied its inner pre-pipeline to the whole frame, and the report, the MoE gate and the refit
    pre-screen predicted the same wrappers on the same frames: one frame went through one fitted pipeline seven times in a
    suite run. ``tag`` separates variants of the same pair (a grouped wrapper drops its group column first). The transformed
    frame is shared, not copied: its readers only select columns from it.
    """
    memo = _ACTIVE
    if memo is None or pipeline is None:
        return compute()
    key = ("transform", id(pipeline), id(frame), tag, getattr(frame, "shape", None))
    with _LOCK:
        hit = memo.get(key)
    if hit is not None and hit[0] is pipeline and hit[1] is frame:
        return hit[2]
    out = compute()
    with _LOCK:
        memo[key] = (pipeline, frame, out)
    return out


def memo_seed(model: Any, frame: Any, preds: Any) -> None:
    """Record ``preds`` as ``model.predict(frame)`` in the active memo, for a caller that computed it another way."""
    memo = _ACTIVE
    if memo is None:
        return
    key = (id(model), id(frame), getattr(frame, "shape", None))
    with _LOCK:
        memo[key] = (model, frame, np.asarray(preds, dtype=np.float64).reshape(-1).copy())
