"""Whether this process can still commit memory, and whether its own commit charge is worth a kernel restart.

Windows kills a process with ``WinError 1455`` once the system-wide commit charge reaches its limit, and a failed
allocation inside a C library often surfaces as an access violation rather than a ``MemoryError``. Both are fatal and
neither says what to do about it, so the condition is reported while there is still room to act.

Two things go wrong in a long-lived kernel, and they are reported separately:

* the system is nearly out of commit -- a production run started with 1.0 GB available of a 344 GB limit and died
  27 minutes later;
* this interpreter holds a large commit charge that is not resident. The same run held 204.8 GB of private commit
  against an RSS of 0.8 GB. Measured cause: an allocator that does not return freed arenas to the operating system.
  Freeing 1.87 GB of numpy returns all of it; freeing the same amount held by polars returned 0.60 GB, and the rest
  stays committed for the life of the process. A restart is the only way to get it back.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

COMMIT_AVAIL_WARN_GB: float = 16.0
"""Commit headroom below which a run is one large allocation away from a fatal failure."""

COMMIT_AVAIL_WARN_FRACTION: float = 0.05
"""...or below this share of the commit limit, whichever is larger, so a big page file is judged on its own scale."""

RETAINED_COMMIT_WARN_GB: float = 32.0
"""Private commit above which the non-resident share is worth naming; below it a restart buys too little to advise."""

RETAINED_COMMIT_RSS_RATIO: float = 4.0
"""Private commit this many times the resident set means most of the charge is retained, not in use."""


def low_headroom_message(commit_limit_gb: float, commit_avail_gb: float) -> Optional[str]:
    """Message when the SYSTEM is nearly out of commit, else None."""
    if commit_limit_gb <= 0 or commit_avail_gb < 0:
        return None
    bar = max(COMMIT_AVAIL_WARN_GB, COMMIT_AVAIL_WARN_FRACTION * commit_limit_gb)
    if commit_avail_gb >= bar:
        return None
    return (
        f"only {commit_avail_gb:.1f} GB of commit left against a {commit_limit_gb:.1f} GB limit (bar {bar:.1f} GB). "
        f"A process that cannot commit is killed outright -- WinError 1455, or an access violation when the failed "
        f"allocation was made inside a C library. Free memory on this host, grow the paging file, or restart the "
        f"kernel before starting work that allocates."
    )


def retained_commit_message(private_commit_gb: Optional[float], rss_gb: Optional[float]) -> Optional[str]:
    """Message when THIS interpreter holds a large commit charge it is not using, else None."""
    if not private_commit_gb or not rss_gb or private_commit_gb < RETAINED_COMMIT_WARN_GB:
        return None
    if private_commit_gb < RETAINED_COMMIT_RSS_RATIO * rss_gb:
        return None
    return (
        f"this interpreter holds {private_commit_gb:.1f} GB of private commit against {rss_gb:.1f} GB resident, so "
        f"most of it is memory earlier work freed and the allocator kept committed (polars / Rust arenas do not "
        f"return it; numpy does). Nothing in this run can release it -- restart the kernel to get it back."
    )


def warn_on_commit_pressure(commit_limit_gb: float, commit_avail_gb: float, private_commit_gb: Optional[float] = None, rss_gb: Optional[float] = None) -> list[str]:
    """Log every commit-pressure condition that holds at WARNING; returns the messages logged."""
    messages = [m for m in (low_headroom_message(commit_limit_gb, commit_avail_gb), retained_commit_message(private_commit_gb, rss_gb)) if m]
    for message in messages:
        logger.warning("[commit-pressure] %s", message)
    return messages


_RETAINED_REPORT_STEP_GB: float = 8.0
"""The heartbeat repeats the retained-commit warning only after it grew by this much since it was last reported."""

_last_retained_reported_gb: Optional[float] = None


def _retained_worth_repeating(private_gb: Optional[float]) -> bool:
    """Whether the retained-commit condition changed enough since the last report to say it again."""
    global _last_retained_reported_gb
    if private_gb is None:
        return False
    if _last_retained_reported_gb is not None and private_gb < _last_retained_reported_gb + _RETAINED_REPORT_STEP_GB:
        return False
    _last_retained_reported_gb = private_gb
    return True


def check_and_warn(throttle_key: Optional[str] = None) -> list[str]:
    """Probe the system and this process, then warn about whatever commit pressure holds. Never raises.

    ``throttle_key`` marks the heartbeat caller. Low system headroom can develop during a run, so it is repeated there
    through the shared log throttle, keyed by the CONDITION -- keying by the message text failed, because the message
    carries a number that changes every beat, and a production run logged the same warning 21 times. Retained commit
    cannot be released by anything in the run, so the heartbeat repeats it only when it grew materially.
    """
    try:
        from .crash_diagnostics import windows_commit_status

        status = windows_commit_status()
        if not status:
            return []
        private_gb = rss_gb = None
        try:
            import psutil

            mi = psutil.Process().memory_info()
            rss_gb = float(mi.rss) / 1024**3
            private = getattr(mi, "private", None)
            private_gb = float(private) / 1024**3 if private else None
        except Exception as exc:
            logger.debug("process commit probe failed: %s", exc)
        low = low_headroom_message(status["commit_limit_gb"], status["commit_avail_gb"])
        retained = retained_commit_message(private_gb, rss_gb)
        messages = [m for m in (low, retained) if m]
        if throttle_key:
            from mlframe.utils.log_throttle import log_throttle

            if low:
                log_throttle(logger, f"{throttle_key}:low_headroom", logging.WARNING, "[commit-pressure] %s", low)
            if retained and _retained_worth_repeating(private_gb):
                logger.warning("[commit-pressure] %s", retained)
        else:
            for message in messages:
                logger.warning("[commit-pressure] %s", message)
            if retained:
                _retained_worth_repeating(private_gb)  # the startup report counts: the heartbeat continues from it
        return messages
    except Exception as exc:  # pragma: no cover - a diagnostic must never end a run
        logger.debug("commit-pressure check failed: %s", exc)
        return []
