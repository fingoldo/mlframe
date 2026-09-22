"""Process-level ledger of regression sensor trips, so a flagged model carries its flags past the log stream.

The collapse sensor and the prediction-envelope clip each emit a WARNING and return. Nothing collects them, so a
model that tripped both on both splits is persisted and reported exactly like a clean one, and the evidence survives
only in whichever log the operator happens to read. A production run flagged ``target_hours_to_hire`` four times --
``group-ood-shift`` on val and test, an envelope clip on each -- and saved the artefact regardless.

The ledger is deliberately a plain process-level dict keyed by model name: the sensors sit deep in the reporting
path and receive the model's NAME, not the estimator, so threading a return value up to the persistence layer would
mean changing every frame in between. Entries are appended as they happen and read at save time.
"""
from __future__ import annotations

import threading
from typing import Any

_TRIPS: dict[str, list[dict[str, Any]]] = {}
_LOCK = threading.Lock()

__all__ = ["record_sensor_trip", "sensor_trips_for", "clear_sensor_trips"]


def record_sensor_trip(model_name: Any, sensor: str, branch: str, **details: Any) -> None:
    """Append one trip for ``model_name``. Never raises -- a diagnostic must not be able to fail a training run."""
    try:
        key = str(model_name)
        entry = {"sensor": sensor, "branch": branch, **details}
        with _LOCK:
            _TRIPS.setdefault(key, []).append(entry)
    except Exception:  # nosec B110 -- a ledger write can never be worth failing a fit over
        pass


def sensor_trips_for(model_name: Any) -> list[dict[str, Any]]:
    """Every trip recorded for ``model_name`` so far, oldest first; empty when it has tripped nothing."""
    with _LOCK:
        return list(_TRIPS.get(str(model_name), ()))


def clear_sensor_trips() -> None:
    """Forget every recorded trip. Called at suite entry so one run's flags cannot leak into the next."""
    with _LOCK:
        _TRIPS.clear()
