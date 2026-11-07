"""Synchronous pre-compilation of the hot numba ``typed.Dict`` machinery.

``screen_predictors`` builds several ``numba.typed.Dict(unicode -> double precision)``
memoization caches and passes them into njit kernels (``evaluate_gain``, the
entropy kernels, ``fleuret``, ``friend_graph``). Constructing the first such
dict in a process JIT-compiles the whole typed-dict method suite (empty /
setitem / getitem / contains / delitem / len / iter) - ~5s of LLVM codegen
that is NOT numba-disk-cacheable and otherwise lands inside the first fit.

The compile is per-process: once the machinery exists, every subsequent fit in
the same process reuses it. Warming it here, synchronously at import, moves that
one-off cost off the fit's critical path and onto import (which the caller pays
once), so the FIRST fit is as fast as later ones. A background-thread variant was
tried and rejected: numba's global compile lock serialises it against the fit's
own compilation, and this host-serial pipeline has no concurrent GPU work to
overlap it with.

Best-effort and opt-out: set ``MLFRAME_SKIP_NUMBA_WARMUP=1`` to keep import lean
(e.g. for a process that imports mlframe but never fits). Any failure is swallowed
-- correctness never depends on the warm-up.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_warmup_done = False


def warmup_typed_dict() -> None:
    """Force-compile the ``unicode -> double precision`` typed.Dict method suite once."""
    global _warmup_done
    if _warmup_done or os.environ.get("MLFRAME_SKIP_NUMBA_WARMUP") == "1":
        return
    _warmup_done = True
    try:
        import numba
        from numba.core import types

        # Intentional float64: must match the unicode->float64 typed.Dict used by screen_predictors (numba contract)
        d = numba.typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
        # Exercise every operation screen_predictors' caches use so each method
        # lowering is compiled now rather than on first fit.
        d["a,b"] = 1.0
        d["c,d|e,f"] = 2.0
        _ = d["a,b"]
        _ = "a,b" in d
        _ = "zz" in d
        _ = len(d)
        for _k in d:
            _ = d[_k]
        if "a,b" in d:
            d["a,b"] = d["a,b"] + 0.5
        del d["c,d|e,f"]
    except Exception as e:  # nosec B110 - non-trivial body; best-effort/optional path
        # Best-effort only; a warm-up failure must never affect a real fit.
        logger.debug("typed.Dict warm-up failed, first real fit will pay the JIT cost: %s", e)


_BACKGROUND_STARTED = False


def start_background_prewarm() -> None:
    """Run the once-per-process warmups the first fit otherwise pays in the middle of its work (the typed.Dict suite, about 5 s, and the numba.cuda compile probe, about 5 s,
    both seen as ticks with the CPU and the GPU idle in a cold 300k fit) on a daemon thread started at fit entry, so they overlap the fit's first stages. Both are idempotent and the
    probe is locked, so a fit that reaches them first just waits for or repeats nothing. Opt-in: ``MLFRAME_FIT_BACKGROUND_PREWARM=1``.

    bench-attempt-rejected (2026-11-07, 100000 x 100 all families, fresh process per run, alternating): off 77.6 / 77.7 s, on 77.0 / 78.6 s. No gain, so it is not the default."""
    global _BACKGROUND_STARTED
    if _BACKGROUND_STARTED or os.environ.get("MLFRAME_SKIP_NUMBA_WARMUP") == "1" or os.environ.get("MLFRAME_FIT_BACKGROUND_PREWARM", "0").strip() != "1":
        return
    _BACKGROUND_STARTED = True
    import threading

    def _run() -> None:
        """The warmups, each failure logged and ignored (the fit's own call does the same work)."""
        try:
            warmup_typed_dict()
        except Exception as exc:  # nosec B110 - best effort: the fit compiles on first use otherwise
            logger.debug("background typed.Dict warmup failed: %r", exc)
        try:
            from ._internals import numba_cuda_can_compile

            numba_cuda_can_compile()
        except Exception as exc:  # nosec B110 - best effort: the fit probes on first use otherwise
            logger.debug("background numba.cuda probe failed: %r", exc)

    threading.Thread(target=_run, daemon=True, name="mlframe-fit-prewarm").start()
