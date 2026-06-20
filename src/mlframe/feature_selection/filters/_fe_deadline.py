"""Thread-local wall-clock deadline for the OPTIONAL pre-FE enrichment generators (orthogonal/extra-basis univariate FE,
pair-cross FE). MRMR.fit honours ``max_runtime_mins`` at the FE-loop level (between steps) and now also gates each
enrichment stage before it starts, but a SINGLE enrichment stage on a wide frame can itself run far past a tiny budget
(measured: the orthogonal extra-basis + pair-cross pass alone is tens of seconds at p>=120). Those stages run BEFORE the
budget is spent, so a before-start gate cannot stop them -- they need an INTERNAL per-column / per-pair deadline check.

This module carries that deadline as a thread-local so the generators (in sibling modules) can consult it without
threading a parameter through every call site. The deadline is advisory and scoped to the ENRICHMENT generators only:
the core screen / greedy-selection MI is never gated here, so an aborted enrichment pass still leaves screen free to
produce a usable partial selection (the budget contract: abort early AND expose a non-empty ``support_``).

``set_fe_deadline`` is called once at the top of MRMR.fit (absolute ``timer()`` value, or None to disable) and cleared
in a finally. ``fe_deadline_passed`` is a cheap monotonic-clock compare the enrichment loops call to decide whether to
``break`` and return whatever they engineered so far."""
from __future__ import annotations

import threading
from timeit import default_timer as timer

_state = threading.local()


def set_fe_deadline(deadline: float | None) -> None:
    """Set (or clear with ``None``) the thread-local absolute deadline -- a ``timer()`` value past which the optional
    enrichment FE generators should stop and return their partial output."""
    _state.deadline = deadline


def clear_fe_deadline() -> None:
    _state.deadline = None


def fe_deadline_passed() -> bool:
    """True iff a deadline is set AND the monotonic clock is past it. False when no deadline is set (the common,
    no-budget path) so the generators run to completion."""
    dl = getattr(_state, "deadline", None)
    return dl is not None and timer() >= dl


def fe_budget_active() -> bool:
    """True iff a wall-clock budget (``max_runtime_mins``) is in effect for the current fit, regardless of how much
    has elapsed. Used to skip one-time-but-BLOCKING kernel-tuning sweeps (the CPU-vs-GPU crossover sweeps that ignore
    ``max_runtime_mins`` and run tens of seconds at large n): under an explicit budget the caller wants speed now, so the
    dispatcher uses its measurement-backed fallback backend instead of paying the sweep. No-op (False) on the common
    no-budget path, so a normal fit still tunes per host as before."""
    return getattr(_state, "deadline", None) is not None


__all__ = ["set_fe_deadline", "clear_fe_deadline", "fe_deadline_passed", "fe_budget_active"]
