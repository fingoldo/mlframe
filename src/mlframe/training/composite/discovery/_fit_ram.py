"""Per-fit RAM telemetry helpers for ``CompositeTargetDiscovery.fit``.

Carved out of ``_fit.py`` to keep that module under the 1k-line monolith
threshold. ``fit`` imports ``_phase_ram_report`` and threads a small ``state``
dict through it at each sub-phase boundary. Pure diagnostics -- no effect on the
discovered specs; opt out via ``MLFRAME_DISCOVERY_RAM_PROFILER=0``.
"""
from __future__ import annotations

import logging

from ..._ram_helpers import get_process_rss_mb as _rss_mb

logger = logging.getLogger(__name__)


def _process_mem_mb() -> tuple[float, float, float]:
    """Return ``(rss_mb, uss_mb, commit_mb)`` for the current process.

    Three Windows memory signals: RSS (working set -- misleading post-EmptyWorkingSet since pages page back in on touch); USS (uniquely-owned pages, immune to EmptyWorkingSet -- the honest "what this process allocated"); commit/private bytes (the CommitCharge Windows gates OutOfMemory on -- commit-limit = physical_RAM + pagefile, so a process at commit near the limit triggers kernel-kill on the next big alloc even when USS fits physical RAM). All floats in MB; falls back through RSS when ``memory_full_info()`` is unavailable (never raises).
    """
    rss = _rss_mb()
    uss = rss
    commit = rss
    try:
        import psutil as _psutil
        full = _psutil.Process().memory_full_info()
        uss = float(getattr(full, "uss", rss * 1024 ** 2)) / 1024 ** 2
        # ``private`` exists on Windows (psutil's MEMORY_PRIVATE_USAGE counter --
        # the actual CommitCharge); on Linux fall back to VMS minus shared.
        priv = getattr(full, "private", None)
        if priv is not None:
            commit = float(priv) / 1024 ** 2
        else:
            vms = float(getattr(full, "vms", rss * 1024 ** 2))
            shared = float(getattr(full, "shared", 0.0))
            commit = (vms - shared) / 1024 ** 2
    except Exception:
        pass
    return rss, uss, commit


def _phase_ram_report(state: dict, phase_name: str) -> None:
    """Emit one INFO log line per discovery sub-phase boundary with delta-vs-prev + cumulative delta vs the fit() baseline.

    Reports RSS and USS (RSS << USS flags page-thrashing the prior version masked); ``state`` is a ``{'baseline_uss_mb', 'prev_uss_mb'}`` dict the caller threads through. No GC is forced here -- ``pyutilz.clean_ram()`` on Windows evicts the working set without freeing real memory (and emits a bogus "reclaimed 57 GB" line); the suite runs gc at its own boundaries.
    """
    # The only observable effect of this function is one INFO log line; when INFO is disabled the line is discarded, so the
    # expensive ``memory_full_info()`` USS/commit walk (tens of ms each on Windows when the working set is dirty) would be pure
    # waste. Short-circuit before touching psutil so the default (WARNING-level) config pays nothing for telemetry it drops.
    if not logger.isEnabledFor(logging.INFO):
        return
    try:
        rss_mb, uss_mb, commit_mb = _process_mem_mb()
    except Exception:
        return
    if state.get("baseline_uss_mb") is None:
        state["baseline_uss_mb"] = uss_mb
        state["prev_uss_mb"] = uss_mb
        state["baseline_commit_mb"] = commit_mb
        state["prev_commit_mb"] = commit_mb
        logger.info(
            "[CompositeTargetDiscovery.RAM] phase=%s start USS=%.0f MB (RSS=%.0f MB, commit=%.0f MB)",
            phase_name, uss_mb, rss_mb, commit_mb,
        )
        return
    prev_uss = state["prev_uss_mb"]
    baseline_uss = state["baseline_uss_mb"]
    prev_commit = state.get("prev_commit_mb", commit_mb)
    # When RSS << USS by a non-trivial margin the process is page-thrashing
    # (working-set evicted by EmptyWorkingSet or external memory pressure).
    # When commit >> USS the process holds committed but rarely-touched
    # memory -- on Windows this consumes the system-wide commit limit and
    # is the proximate cause of OOM-kernel-kill even when USS itself is
    # well under physical RAM. Surface both hints inline.
    _hints = []
    if uss_mb > rss_mb * 2 and uss_mb > 1024:
        _hints.append(f"PAGE_THRASHING(uss/rss={uss_mb/max(rss_mb, 1):.1f}x)")
    if commit_mb > uss_mb * 1.4 and commit_mb > 4096:
        _hints.append(f"COMMIT_PRESSURE(commit/uss={commit_mb/max(uss_mb, 1):.1f}x)")
    _hint_suffix = (" " + " ".join(_hints)) if _hints else ""
    logger.info(
        "[CompositeTargetDiscovery.RAM] phase=%s USS=%.0f MB (RSS=%.0f MB, commit=%.0f MB; delta_uss_vs_prev=%+.0f MB, delta_commit_vs_prev=%+.0f MB, cum_uss=%+.0f MB)%s",
        phase_name,
        uss_mb,
        rss_mb,
        commit_mb,
        uss_mb - prev_uss,
        commit_mb - prev_commit,
        uss_mb - baseline_uss,
        _hint_suffix,
    )
    state["prev_uss_mb"] = uss_mb
    state["prev_commit_mb"] = commit_mb
