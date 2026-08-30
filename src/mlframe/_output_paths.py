"""Create a figure's parent directory before writing it.

``render_and_save`` has always made its own directories, but thirteen other call sites write a figure straight
to a path -- ``savefig`` / ``write_html`` / ``write_image`` -- with nothing having created the folder. On a
first run into a fresh output tree those raise ``FileNotFoundError``, and most of them sit inside a
best-effort ``except`` that swallows it, so the chart simply never appears and nothing says why.

Every writer routes through :func:`ensure_parent_dir` so a missing directory is created rather than diagnosed
by its absence.

It lives at the top level rather than under ``mlframe.reporting`` because importing any name from that
package runs its ``__init__``, which builds the chart catalogue -- and a chart module imports
``mlframe.metrics``, so a metrics module importing this helper from there closed an import cycle.
"""

from __future__ import annotations

import logging
import os
from typing import TypeVar

PathT = TypeVar("PathT")

logger = logging.getLogger(__name__)


def ensure_parent_dir(path: PathT) -> PathT:
    """Create ``path``'s parent directory if needed and return ``path`` unchanged.

    Returns the path so a caller can wrap an existing expression in place: ``fig.savefig(ensure_parent_dir(p))``.
    A path with no directory component (a bare filename, written into the cwd) needs nothing created. Failure to
    create the directory is logged and swallowed: the write itself is about to raise a far more specific error,
    and masking it with a mkdir traceback helps nobody.
    """
    if not path:
        return path
    directory = os.path.dirname(str(path))
    if not directory:
        return path
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        logger.debug("could not create output directory %r (%s: %s)", directory, type(exc).__name__, exc)
    return path


__all__ = ["ensure_parent_dir"]
