"""Per-test stall diagnostics: dumps every thread's native/Python stack if a test runs too long.

CI's macOS shards have repeatedly hung until the 360-minute job cap force-kills them (see the long note
on ``timeout-minutes: 360`` in ``.github/workflows/ci.yml``), with the real blocking call never identified
because pytest-timeout's ``method=thread`` cannot interrupt a test blocked inside native code, and
``method=signal`` (which can) was already tried on macOS and reverted -- SIGALRM delivered into a thread
running numba/multiprocessing native work there raises ``Fatal Python error: Aborted`` instead of a clean
timeout. ``faulthandler.dump_traceback_later`` does not raise into the stalled thread at all -- it only
prints every thread's current frames from a separate watchdog thread -- so it cannot trigger that abort,
and it names the exact blocking call the next time a shard hangs instead of just reporting "cancelled".
"""

import faulthandler
import os
import sys

HANG_WATCHDOG_SECONDS = int(os.environ.get("MLFRAME_HANG_WATCHDOG_SECONDS", "600"))


def arm(node_id: str) -> None:
    """Start the traceback-dump timer for one test. A value of ``0`` disables the watchdog entirely."""
    if HANG_WATCHDOG_SECONDS <= 0:
        return
    print(f"[hang-watchdog] arming {HANG_WATCHDOG_SECONDS}s for {node_id}", file=sys.stderr)
    faulthandler.dump_traceback_later(HANG_WATCHDOG_SECONDS, exit=False, file=sys.stderr)


def disarm() -> None:
    """Cancel the timer after the test finishes normally, so a slow-but-fine later test isn't dumped."""
    if HANG_WATCHDOG_SECONDS <= 0:
        return
    faulthandler.cancel_dump_traceback_later()
