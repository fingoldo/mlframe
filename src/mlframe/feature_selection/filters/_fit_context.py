"""Snapshot of the per-fit thread-local settings, to publish to worker threads.

``MRMR.fit`` stores its scoring switches (symmetric uncertainty, Miller-Madow, Chao-Shen, JMIM, BUR, group MI, the research knobs), the AUTO-STRICT fit
size, the dynamic-cluster-discovery state and the Fourier detection cap in ``threading.local`` objects, so a fresh worker thread sees only defaults.
The pools that run FE work on threads capture the snapshot in the calling thread and open :func:`fit_context_scope` inside each worker, which makes the
concurrent run score exactly like the sequential one. The wall-clock deadline has its own scope (``fe_deadline_scope``).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, List, Tuple


def _fit_locals() -> "List[Any]":
    """The thread-local objects that carry per-fit settings (imported lazily: some of these modules import the pools that use this one)."""
    from . import _dynamic_cluster_discovery as dcd
    from . import _fe_gpu_strict, _fourier_detect_cap
    from .info_theory.shared import state_and_dispatch as sd

    names = ("_SU_STATE", "_JMIM_STATE", "_BUR_STATE", "_MM_STATE", "_CS_STATE", "_GROUP_MI_STATE", "_RELAXMRMR_STATE", "_PID_STATE", "_CMI_PERM_STATE", "_CPT_STATE")
    return [getattr(sd, n) for n in names if hasattr(sd, n)] + [dcd._DCD_STATE, _fe_gpu_strict._auto_fit_state, _fourier_detect_cap._state]


def capture_fit_context() -> "List[Tuple[Any, dict]]":
    """``[(thread_local, its attributes in the calling thread), ...]`` for every per-fit thread-local."""
    return [(obj, dict(obj.__dict__)) for obj in _fit_locals()]


@contextmanager
def fit_context_scope(snapshot: "List[Tuple[Any, dict]]") -> Iterator[None]:
    """Apply ``snapshot`` to the CURRENT thread's thread-locals for the duration of the block, then put back what the thread had."""
    previous = [(obj, dict(obj.__dict__)) for obj, _ in snapshot]
    for obj, attrs in snapshot:
        obj.__dict__.clear()
        obj.__dict__.update(attrs)
    try:
        yield
    finally:
        for obj, attrs in previous:
            obj.__dict__.clear()
            obj.__dict__.update(attrs)
