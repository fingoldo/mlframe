"""Thread-local row-cap for the adaptive Fourier frequency detector (orthogonal extra-basis FE).

The detector row-subsamples its working set so a wide / 1M-row fit does not OOM building per-column coarse-grid planes; the cap was previously carried through ``os.environ["MLFRAME_FOURIER_DETECT_MAX_N"]``, which MRMR.fit shrank to the fast-search subsample for the fit's duration and restored in a finally. ``os.environ`` is process-global and not thread-safe, so two concurrent fits (multi-target discovery, joblib threading, service workers) raced the write/restore and made the detector's sample size non-deterministic across fits. This module carries the cap as a thread-local instead, so each fit's shrink is isolated to its own thread; the ``MLFRAME_FOURIER_DETECT_MAX_N`` env var is still honoured as the cross-process default when no per-fit cap is set.

The cap is detection-only (the recipe replays ``sin(2*pi*f*x)`` full-n), so a per-thread value cannot change selection across threads — it only bounds the detector's working sample.
"""
from __future__ import annotations

import os
import threading

_DEFAULT_MAX_N = 200_000
_state = threading.local()


def set_fourier_detect_cap(max_n: int | None) -> None:
    """Set (or clear with ``None``) the thread-local Fourier-detector row cap for the current fit."""
    _state.cap = max_n


def clear_fourier_detect_cap() -> None:
    """Reset the thread-local Fourier-detect cap to unset (``None``) so the next fit recomputes it."""
    _state.cap = None


def peek_fourier_detect_cap() -> int | None:
    """Raw thread-local cap (``None`` when unset). For the writer's snapshot/restore so nested fits restore the outer cap rather than clearing it."""
    return getattr(_state, "cap", None)


def get_fourier_detect_max_n() -> int:
    """Effective cap for the detector: the thread-local per-fit value if set, else ``MLFRAME_FOURIER_DETECT_MAX_N`` (``0`` / empty disables the cap), else the 200k default. Faithful to the prior ``int(os.environ.get(..., "200000") or "0")`` reader, but thread-local-first and robust to a non-numeric env value (treated as no cap)."""
    cap = getattr(_state, "cap", None)
    if cap is not None:
        return int(cap)
    raw = os.environ.get("MLFRAME_FOURIER_DETECT_MAX_N", str(_DEFAULT_MAX_N)) or "0"
    return int(raw) if raw.strip().lstrip("-").isdigit() else 0


__all__ = [
    "set_fourier_detect_cap",
    "clear_fourier_detect_cap",
    "peek_fourier_detect_cap",
    "get_fourier_detect_max_n",
]
