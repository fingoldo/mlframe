"""Training callbacks subsystem.

Groups the two callback families used by the training loop:

- ``_callbacks`` -- the early-stopping callback classes
  (``UniversalCallback`` + the per-library ``LightGBMCallback`` /
  ``XGBoostCallback`` / ``CatBoostCallback`` adapters).
- ``stop_file`` -- the cooperative stop-file callbacks
  (``stop_file`` predicate factory + per-library
  ``*StopFileCallback`` adapters) for out-of-band training shutdown.

The public surface is re-exported here so existing
``from mlframe.training.callbacks import X`` import sites resolve from the
documented package path.
"""
from __future__ import annotations

from ._callbacks import (  # noqa: F401
    UniversalCallback,
    LightGBMCallback,
    XGBoostCallback,
    CatBoostCallback,
)
from .stop_file import (  # noqa: F401
    stop_file,
    CatBoostStopFileCallback,
    LightGBMStopFileCallback,
    XGBoostStopFileCallback,
    LightningStopFileCallback,
)

__all__ = [
    "UniversalCallback",
    "LightGBMCallback",
    "XGBoostCallback",
    "CatBoostCallback",
    "stop_file",
    "CatBoostStopFileCallback",
    "LightGBMStopFileCallback",
    "XGBoostStopFileCallback",
    "LightningStopFileCallback",
]
