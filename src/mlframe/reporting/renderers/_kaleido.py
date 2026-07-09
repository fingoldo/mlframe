"""Kaleido lifecycle + static-image write plumbing for the plotly renderer.

Carved out of ``plotly.py`` to keep that module under the size limit. Holds the persistent
sync-server state machine (start / restart / burn), the oneshot-fallback stats counters, and
``write_image_via_kaleido`` -- the png/svg/pdf write path with the persistent-fast-path /
oneshot-retry / HTML-fallback recovery ladder. ``plotly.py`` re-exports the public symbols so
``from mlframe.reporting.renderers.plotly import get_kaleido_oneshot_stats`` still resolves.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

logger = logging.getLogger(__name__)

# Process-singleton: track whether the persistent kaleido sync server
# is up so we don't pay the ~10-15s Chromium-spawn cost on every PNG /
# SVG / PDF write. Verified empirically (2026-05-08) that kaleido 1.x
# default ``fig.write_image()`` calls spawn a fresh Chromium process
# per call (~13s each); persistent server reuses one process and drops
# subsequent calls to ~0.13s. On c0114 (lgb+xgb multiclass, 100k rows,
# 32 PNG calls), this saved 32 * ~13s = ~7 minutes of wall time.
_KALEIDO_SERVER_STARTED = False

# Process-wide flag that the persistent path has burned itself on a JS error / hang. Once True,
# all subsequent saves in this process go straight to HTML fallback (skipping kaleido entirely
# because plotly's oneshot also routes through the SAME broken sync server). Reset only on
# interpreter exit.
_KALEIDO_PERSISTENT_BURNED = False

# Count of consecutive timeouts/errors. Burn after this many. Single-burn-on-first-failure paid
# 12 x 30s timeouts in c0031 because each timeout fires once before "burning" via the next-call
# path; cap at 2 consecutive failures (~60s wasted before HTML fallback takes over).
_KALEIDO_PERSISTENT_FAIL_COUNT = 0
_KALEIDO_PERSISTENT_FAIL_THRESHOLD = 2

# Idempotency guard for the "Failed to start kaleido sync server" warning: without it the warn
# fired on EVERY call when the kaleido binary lacked ``start_sync_server`` (kaleido 0.x wheels),
# 32+ times per plot-heavy suite. The suite-end wall-share log still surfaces the cumulative
# oneshot-fallback cost so the user notices the missing fast path.
_KALEIDO_START_WARN_EMITTED = False
# Counter of fallback PNG/SVG/PDF writes that took the slow oneshot path, reported in the
# suite-end summary so the reader sees ROI for upgrading kaleido.
_KALEIDO_ONESHOT_CALL_COUNT = 0
_KALEIDO_ONESHOT_TOTAL_WALL_S = 0.0


def get_kaleido_oneshot_stats() -> Tuple[int, float]:
    """Returns (n_oneshot_calls, total_wall_seconds) so suite-end
    reporting can quote concrete numbers. Cleared via ``reset_kaleido_oneshot_stats``."""
    return _KALEIDO_ONESHOT_CALL_COUNT, _KALEIDO_ONESHOT_TOTAL_WALL_S


def reset_kaleido_oneshot_stats() -> None:
    """Zero the cumulative oneshot-fallback counters (call/wall-time), typically between test runs or suite invocations."""
    global _KALEIDO_ONESHOT_CALL_COUNT, _KALEIDO_ONESHOT_TOTAL_WALL_S
    _KALEIDO_ONESHOT_CALL_COUNT = 0
    _KALEIDO_ONESHOT_TOTAL_WALL_S = 0.0


def record_kaleido_oneshot_call(wall_s: float) -> None:
    """Accumulate one oneshot-fallback write's wall time into the process-global stats used by the suite-end summary."""
    global _KALEIDO_ONESHOT_CALL_COUNT, _KALEIDO_ONESHOT_TOTAL_WALL_S
    _KALEIDO_ONESHOT_CALL_COUNT += 1
    _KALEIDO_ONESHOT_TOTAL_WALL_S += wall_s


# Hard ceiling on a single persistent write_fig_sync call. Beyond this the call is treated as
# hung; we leave the worker thread to die on process exit (the kaleido server holds an asyncio
# loop so we can't safely cancel it from outside). Normal cost after warmup: 0.13s/call; cold
# persistent warmup: ~8s. 30s bounds c0031-style hangs to 30s instead of infinity.
_KALEIDO_PERSISTENT_TIMEOUT_S = 30.0


def _is_kaleido_persistent_burned() -> bool:
    """True once the persistent-server path has been given up on for this process (fail threshold crossed or force-burned)."""
    return _KALEIDO_PERSISTENT_BURNED


def _record_kaleido_persistent_failure() -> bool:
    """Increment failure counter; return True if we just crossed the
    threshold (caller should burn the persistent path)."""
    global _KALEIDO_PERSISTENT_FAIL_COUNT, _KALEIDO_PERSISTENT_BURNED
    _KALEIDO_PERSISTENT_FAIL_COUNT += 1
    if _KALEIDO_PERSISTENT_FAIL_COUNT >= _KALEIDO_PERSISTENT_FAIL_THRESHOLD:
        _KALEIDO_PERSISTENT_BURNED = True
        return True
    return False


def _mark_kaleido_persistent_burned() -> None:
    """Force-burn the persistent path (legacy entry, kept for tests
    and external callers)."""
    global _KALEIDO_PERSISTENT_BURNED
    _KALEIDO_PERSISTENT_BURNED = True


def _restart_kaleido_server() -> bool:
    """Stop + restart the kaleido sync server. Used after a JS error
    poisons the async task chain so subsequent calls don't deadlock.
    Idempotent / no-op when the server isn't started.

    A successful restart clears the persistent-failure counter AND the burned flag. The counter
    exists to catch "this process's kaleido is fundamentally broken" -- a clean restart proves
    otherwise. Without this, prior failures accumulated across two unrelated callsites (e.g. two
    separate test_kaleido_recovery tests) would cross the burned threshold and force HTML fallback
    forever.
    """
    global _KALEIDO_SERVER_STARTED
    global _KALEIDO_PERSISTENT_FAIL_COUNT, _KALEIDO_PERSISTENT_BURNED
    try:
        import kaleido
    except ImportError:
        return False
    if _KALEIDO_SERVER_STARTED:
        try:
            kaleido.stop_sync_server(silence_warnings=True)
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed in _kaleido.py:116: %s", e)
            pass
        _KALEIDO_SERVER_STARTED = False
    started = _ensure_kaleido_server_started()
    if started:
        # Successful restart: the persistent path is usable again; clear cumulative-failure state.
        _KALEIDO_PERSISTENT_FAIL_COUNT = 0
        _KALEIDO_PERSISTENT_BURNED = False
    return started


def _ensure_kaleido_server_started() -> bool:
    """Start the kaleido sync server (idempotent). Returns True if the
    server is up after the call, False if kaleido isn't installed.
    """
    global _KALEIDO_SERVER_STARTED
    if _KALEIDO_SERVER_STARTED:
        return True
    try:
        import kaleido
    except ImportError:
        return False
    try:
        # silence_warnings=True so the "already started" message doesn't spam logs if some other
        # caller already started the server.
        kaleido.start_sync_server(silence_warnings=True)
        _KALEIDO_SERVER_STARTED = True
        # Register cleanup so the Chromium subprocess gets a clean exit rather than the "Resorting
        # to unclean kill browser" warning at interpreter shutdown.
        import atexit
        def _stop():
            """Stop the kaleido sync server at interpreter shutdown so the Chromium subprocess exits cleanly."""
            try:
                kaleido.stop_sync_server(silence_warnings=True)
            except Exception:  # nosec B110 - optional dependency import guard
                pass
        atexit.register(_stop)
        return True
    except Exception as e:
        global _KALEIDO_START_WARN_EMITTED
        if not _KALEIDO_START_WARN_EMITTED:
            logger.warning(
                "[plotly-render] kaleido sync server unavailable (%s); will "
                "use the slower per-call oneshot path. This warning fires "
                "ONCE per process; check the suite-end wall-share summary "
                "for cumulative oneshot cost. To enable the fast path, "
                "upgrade kaleido (>=1.x ships ``start_sync_server``).", e,
            )
            _KALEIDO_START_WARN_EMITTED = True
        return False


def _oneshot_write_static(fig: Any, path: str, fmt: str) -> None:
    """Isolated static-image write for the oneshot fallback, bypassing the persistent sync server.

    Prefers plotly's ``fig.write_image`` (the fast native path when plotly>=6.1 matches kaleido>=1). On the
    version-mismatch ValueError kaleido 1.x raises for older plotly (``fig.write_image`` disabled), fall back to
    kaleido's ``calc_fig_sync`` + a manual byte write -- the API kaleido's own warning recommends. ``calc_fig_sync``
    does not route through the ``write_fig_sync`` queue that may have just failed/poisoned the persistent path, so
    it is a genuinely independent recovery route (verified: writes a real PNG even when write_fig_sync raises).
    """
    try:
        fig.write_image(path, format=fmt)
        return
    except Exception:
        import kaleido
        data = kaleido.calc_fig_sync(fig, opts={"format": fmt})
        with open(path, "wb") as fh:
            fh.write(data)


def write_image_via_kaleido(fig: Any, path: str, fmt: str) -> None:
    """Write ``fig`` to ``path`` as png/svg/pdf via kaleido.

    Persistent-server fast path reuses one Chromium subprocess across all calls in the process,
    dropping per-call cost from ~13s (oneshot) to ~0.13s. Recovery ladder for the c0031-class
    hang (a JS error inside kaleido cancels the persistent server's async task chain and leaves
    ``write_fig_sync`` blocked forever):
      1. Try the persistent server in a worker thread with a hard timeout.
      2. On a single raised exception, restart the sync server and retry via the isolated oneshot.
      3. On a timeout (server hung) or after the burn threshold, write interactive HTML instead --
         a different code path (``write_html`` never touches kaleido) that always works.
    ``server_hung`` (timeout) is distinguished from a single raised exception: a hung server skips
    the oneshot retry (plotly's oneshot would re-enter the same dead queue) and goes straight to
    HTML, while a raised exception restarts the server and retries oneshot.
    """
    persistent_failed = _is_kaleido_persistent_burned()
    server_hung = False
    burned_now = False
    if not persistent_failed and _ensure_kaleido_server_started():
        import kaleido as _kal
        import threading

        _result: list = [None]
        _exc: list = [None]

        def _do_persistent():
            """Run the persistent-server write on a worker thread so the caller can enforce a hard timeout on a hung kaleido server."""
            try:
                _kal.write_fig_sync(fig, path, opts={"format": fmt})
            except Exception as ee:
                _exc[0] = ee

        th = threading.Thread(target=_do_persistent, daemon=True)
        th.start()
        th.join(timeout=_KALEIDO_PERSISTENT_TIMEOUT_S)
        if th.is_alive():
            persistent_failed = True
            server_hung = True
            burned_now = _record_kaleido_persistent_failure()
            logger.warning(
                "kaleido persistent write_fig_sync(%s) did not "
                "return in %.0fs%s.",
                fmt, _KALEIDO_PERSISTENT_TIMEOUT_S,
                "; persistent path BURNED for this process -- subsequent "
                "saves write HTML directly" if burned_now else
                " (will retry persistent up to threshold before burning)",
            )
            # Don't try to restart the server -- a hung server may have stop_sync_server() ALSO
            # blocking. Just leave it behind; it gets cleaned up at process exit.
        elif _exc[0] is not None:
            persistent_failed = True
            burned_now = _record_kaleido_persistent_failure()
            logger.warning(
                "kaleido persistent save(%s) raised %s%s.",
                fmt,
                type(_exc[0]).__name__,
                "; persistent path BURNED" if burned_now else " (will retry up to threshold)",
            )
            # Server is likely still alive but its async task chain may be poisoned -- restart it
            # so the oneshot retry below uses a clean queue. Best-effort; if restart hangs we fall
            # through to the HTML fallback via the burned-or-hung gate below.
            try:
                _restart_kaleido_server()
            except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed in _kaleido.py:248: %s", e)
                pass
        else:
            return  # success

    if persistent_failed and (server_hung or burned_now or _is_kaleido_persistent_burned()):
        from os.path import splitext
        root, _ = splitext(path)
        try:
            fig.write_html(root + ".html", include_plotlyjs="cdn", auto_open=False)
            logger.warning(
                "kaleido burned -- wrote interactive HTML instead of %s "
                "to %s (PNG/SVG/PDF unavailable for the rest of this "
                "process; restart Python to retry persistent kaleido).",
                fmt, root + ".html",
            )
        except Exception as e:
            logger.error(
                "All save paths failed for %s (%s); diagnostic chart "
                "lost but suite continues. Last error: %s",
                path, fmt, e,
            )
        return

    # Persistent path skipped (kaleido never started or unavailable); use plotly oneshot. Catch
    # ALL exceptions for HTML fallback. Oneshot wall-time + call count are instrumented so the
    # suite-end summary surfaces the cumulative cost (e.g. 32 oneshots x ~13s = 7+ minutes that
    # disappear when kaleido is upgraded to a build with ``start_sync_server``).
    import time as _time
    _t0 = _time.time()
    try:
        try:
            _oneshot_write_static(fig, path, fmt)
        except Exception as e:
            logger.warning(
                "plotly write_image(%s) oneshot failed (%s: %s); "
                "falling back to .html.",
                fmt, type(e).__name__, e,
            )
            from os.path import splitext
            root, _ = splitext(path)
            try:
                fig.write_html(root + ".html", include_plotlyjs="cdn", auto_open=False)
            except Exception as e2:
                logger.error(
                    "All save paths failed for %s (%s); diagnostic chart "
                    "lost but suite continues. Last error: %s",
                    path, fmt, e2,
                )
    finally:
        # Record cumulative oneshot wall regardless of success / exception path; the suite-end
        # summary uses this to quote concrete savings of upgrading kaleido.
        record_kaleido_oneshot_call(_time.time() - _t0)


__all__ = [
    "get_kaleido_oneshot_stats",
    "reset_kaleido_oneshot_stats",
    "record_kaleido_oneshot_call",
    "write_image_via_kaleido",
    "_is_kaleido_persistent_burned",
    "_record_kaleido_persistent_failure",
    "_mark_kaleido_persistent_burned",
    "_restart_kaleido_server",
    "_ensure_kaleido_server_started",
]
