"""Warning suppression that is safe to use from worker threads.

``warnings.catch_warnings()`` snapshots and restores a PROCESS-GLOBAL filter list. Two joblib ``backend="threading"``
workers running the same block overlap: one thread's ``__exit__`` restores the snapshot it took before the other
entered, so the other's suppression disappears mid-computation (warning spam), or an "ignore" leaks past its block and
hides a genuine warning - a numpy ``RuntimeWarning`` over a silent NaN - in unrelated caller code.

The filters here are installed ONCE and never removed, so nothing races a restore. That is only acceptable because each
one is narrow: a specific message, or a category plus the module it may be silenced for. A blanket "ignore this whole
category" must not be installed this way; for numpy's own floating-point warnings use ``np.errstate``, which is
thread-local.
"""

from __future__ import annotations

import threading
import warnings
from typing import Literal

#: The actions `warnings.filterwarnings` accepts; typed so a misspelt action is caught before it is installed.
WarningAction = Literal["default", "error", "ignore", "always", "all", "module", "once"]

_INSTALLED: set = set()
_LOCK = threading.Lock()


def install_filter_once(action: "WarningAction" = "ignore", *, message: str = "", category: type = Warning, module: str = "") -> bool:
    """Install one narrow warning filter, at most once per process; True when this call installed it.

    ``message`` and ``module`` are regexes matched as ``warnings.filterwarnings`` matches them.
    """
    key = (action, message, category, module)
    with _LOCK:
        if key in _INSTALLED:
            return False
        warnings.filterwarnings(action, message=message, category=category, module=module)
        _INSTALLED.add(key)
        return True
