"""Synchronous pre-compilation of the hot numba ``typed.Dict`` machinery.

``screen_predictors`` builds several ``numba.typed.Dict(unicode -> float64)``
memoization caches and passes them into njit kernels (``evaluate_gain``, the
entropy kernels, ``fleuret``, ``friend_graph``). Constructing the first such
dict in a process JIT-compiles the whole typed-dict method suite (empty /
setitem / getitem / contains / delitem / len / iter) -- ~5s of LLVM codegen
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

import os

_warmup_done = False


def warmup_typed_dict() -> None:
    """Force-compile the ``unicode -> float64`` typed.Dict method suite once."""
    global _warmup_done
    if _warmup_done or os.environ.get("MLFRAME_SKIP_NUMBA_WARMUP") == "1":
        return
    _warmup_done = True
    try:
        import numba
        from numba.core import types

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
    except Exception:  # nosec B110 - non-trivial body; best-effort/optional path, no module logger
        # Best-effort only; a warm-up failure must never affect a real fit.
        pass
