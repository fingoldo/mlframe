"""One reading of a boolean environment variable, for every switch in the package.

Switches were parsed four ways. ``not os.environ.get(NAME)`` treats any non-empty value as on, so
``MLFRAME_KEEP_T_SCALE_COMPOSITE_REPORTS=0`` kept the charts it was meant to drop. ``lower() in {"1", "true", "yes"}``
misses ``on``, so a kill switch set to ``on`` did not kill anything. ``== "1"`` misses everything else. An operator
cannot be expected to remember which switch takes which spelling, so they all take the same one here.
"""

from __future__ import annotations

import os

__all__ = ["env_flag", "TRUE_VALUES", "FALSE_VALUES"]

TRUE_VALUES = frozenset({"1", "true", "yes", "on", "y", "t"})
FALSE_VALUES = frozenset({"0", "false", "no", "off", "n", "f", ""})


def env_flag(name: str, default: bool = False) -> bool:
    """Whether the environment variable ``name`` is set to a true value.

    Parameters
    ----------
    name
        The variable's name, including the project prefix.
    default
        What an unset variable, or a value in neither vocabulary, means.

    Returns
    -------
    bool
        True for ``1 / true / yes / on / y / t``, False for ``0 / false / no / off / n / f`` and the empty string, both
        case-insensitively and ignoring surrounding whitespace; ``default`` for anything else.
    """
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    value = raw.strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    return bool(default)
