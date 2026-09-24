"""One reading of a boolean or numeric environment variable, for every switch in the package.

Switches were parsed four ways. ``not os.environ.get(NAME)`` treats any non-empty value as on, so
``MLFRAME_KEEP_T_SCALE_COMPOSITE_REPORTS=0`` kept the charts it was meant to drop. ``lower() in {"1", "true", "yes"}``
misses ``on``, so a kill switch set to ``on`` did not kill anything. ``== "1"`` misses everything else. An operator
cannot be expected to remember which switch takes which spelling, so they all take the same one here.

The numeric readers exist for knobs that change results: read on every call (a value set after ``import mlframe`` takes
effect) and never raising (an unparseable value is reported once and replaced by the default, instead of an import
error in an unrelated module).
"""

from __future__ import annotations

import logging
import os
from typing import Optional, cast

logger = logging.getLogger(__name__)

__all__ = ["env_flag", "env_float", "env_int", "TRUE_VALUES", "FALSE_VALUES"]

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


def _env_number(name: str, default, parse, minimum, maximum):
    """The environment variable ``name`` parsed with ``parse``, or ``default`` when unset, unparseable, NaN or out of range."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = parse(raw.strip().replace("_", ""))
    except ValueError:
        value = None
    if value is None or value != value or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        from mlframe.utils.log_throttle import log_throttle

        log_throttle(logger, "env_number_invalid:" + name, logging.WARNING, "%s=%r is not a valid value (range [%s, %s]); using %r.", name, raw, minimum, maximum, default)
        return default
    return value


def env_float(name: str, default: float, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    """The environment variable ``name`` as a float, or ``default`` when unset, unparseable, NaN or outside [minimum, maximum]."""
    return cast(float, _env_number(name, default, float, minimum, maximum))


def env_int(name: str, default: int, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    """The environment variable ``name`` as an int, or ``default`` when unset, unparseable or outside [minimum, maximum]."""
    return cast(int, _env_number(name, default, int, minimum, maximum))
