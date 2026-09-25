"""One host thread at a time inside LightGBM's native code, process-wide.

A production Jupyter kernel died three times out of three with a Windows access violation and a heap corruption
(0xc0000374) while composite discovery trained tiny LightGBM models from 16 threads. Serialising only the dataset
construction did not stop it, and a local run reproduced the crash with 4 threads in the WAIC tie-break: one thread in
``Booster.__init__``, one in ``Booster.update``, one in ``predict``. Discovery runs LightGBM from nine thread pools and
through three different entry points (the native API, two dataset caches, the sklearn wrapper), so the guard sits on
LightGBM's own native entry points rather than on each call site.

The lock is re-entrant (``lgb.train`` constructs its datasets from inside ``Booster.__init__``) and costs nothing when
only one thread uses LightGBM, which is how mlframe trains its main models. Parallel speed comes from worker PROCESSES
instead (the tiny-model rerank), each with its own lock. ``MLFRAME_LGB_SERIALISE=0`` turns it off.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

LGB_NATIVE_LOCK = threading.RLock()
"""Held for the duration of every guarded LightGBM call."""

GUARDED_METHODS: dict[str, tuple[str, ...]] = {
    "Dataset": ("construct",),
    "Booster": ("__init__", "update", "predict"),
}
"""The native entry points a training or scoring call goes through: binning, booster creation, a boosting round and
prediction -- each observed in a crashed thread."""


def serialisation_enabled() -> bool:
    """False when ``MLFRAME_LGB_SERIALISE`` is set to 0 / false / no / off."""
    return os.environ.get("MLFRAME_LGB_SERIALISE", "1").strip().lower() not in ("0", "false", "no", "off")


def _serialised(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with LGB_NATIVE_LOCK:
            return fn(*args, **kwargs)

    wrapper._mlframe_lgb_serialised = True  # type: ignore[attr-defined]
    return wrapper


def patch_lightgbm_basic(basic_module: Any) -> list[str]:
    """Wrap the guarded methods of ``basic_module``'s Dataset / Booster; returns the ``Class.method`` names it wrapped.

    Idempotent: an already wrapped method is left alone, so a second call wraps nothing.
    """
    if not serialisation_enabled():
        return []
    wrapped = []
    for cls_name, methods in GUARDED_METHODS.items():
        cls = getattr(basic_module, cls_name, None)
        if cls is None:
            continue
        for name in methods:
            fn = cls.__dict__.get(name)
            if fn is None or getattr(fn, "_mlframe_lgb_serialised", False):
                continue
            setattr(cls, name, _serialised(fn))
            wrapped.append(f"{cls_name}.{name}")
    return wrapped
