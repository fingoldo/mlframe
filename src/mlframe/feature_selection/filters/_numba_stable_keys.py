"""Deterministic ids for numba dispatchers, so a cached kernel that closes over or calls other device functions hits its cache in the next process.

numba keys the on-disk cache of a ``cache=True`` function by (signature, target, hash of its bytecode, hash of the pickled content of its closure cells). A closure cell that holds a dispatcher (a ``cuda.jit(device=True)`` helper built
in a factory) is pickled together with everything its function reaches, module-level dispatchers included, and every numba dispatcher carries a ``uuid1`` created lazily on the first pickle: a different string in every process. The key
then never matched: the kernel was compiled again in every process (2.2 s for the usability warp kernel) and one more 320 KB entry was written next to the old ones (285 files, 30 MB after a day of runs).

``pin_dispatcher_uuids`` gives each dispatcher reachable from the given ones a uuid derived from its module and qualified name (and a ``tag``, e.g. the float type a factory specialised it for). It must run before the first pickle of the
dispatcher (numba refuses to change an id that was already read), which is the case right after the factory has built them.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

_NAMESPACE = uuid.UUID("5d3c2b1a-0f4e-4a67-9b1c-6d2e8a7f3c10")


def _is_dispatcher(obj: Any) -> bool:
    """Whether ``obj`` is a numba dispatcher with an id that can be pinned."""
    return hasattr(obj, "_set_uuid") and hasattr(obj, "py_func")


def pin_dispatcher_uuids(*dispatchers: Any, tag: str = "") -> int:
    """Pin the uuid of every dispatcher reachable from ``dispatchers`` (their closures and the globals their code names, followed through nested dispatchers); the number pinned.

    A dispatcher whose id was already read keeps it (counted as not pinned): the pin is best effort, a kernel that keeps a random id is only recompiled, as before."""
    seen: "set[int]" = set()
    pinned = 0
    stack = [d for d in dispatchers if _is_dispatcher(d)]
    while stack:
        disp = stack.pop()
        if id(disp) in seen:
            continue
        seen.add(id(disp))
        pf = disp.py_func
        name = f"{getattr(pf, '__module__', '?')}.{getattr(pf, '__qualname__', getattr(pf, '__name__', '?'))}|{tag}"
        try:
            disp._set_uuid(str(uuid.uuid5(_NAMESPACE, name)))
            pinned += 1
        except AssertionError:
            logger.debug("numba dispatcher %s already has an id; left as it is", name)
        for cell in getattr(pf, "__closure__", None) or ():
            try:
                content = cell.cell_contents
            except ValueError:  # an empty cell
                continue
            if _is_dispatcher(content):
                stack.append(content)
        glb = getattr(pf, "__globals__", {})
        for gname in getattr(getattr(pf, "__code__", None), "co_names", ()):
            obj = glb.get(gname)
            if _is_dispatcher(obj):
                stack.append(obj)
    return pinned
