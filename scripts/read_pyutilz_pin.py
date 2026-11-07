"""Print the commit SHA that ``uv.lock`` pins pyutilz to.

``uv.lock`` is the single source for the pyutilz revision: ``uv lock --upgrade-package pyutilz`` moves it, and CI reads it from here instead
of a hand-copied SHA in each workflow. Exits non-zero when the lock has no git-pinned pyutilz entry.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_PYUTILZ_GIT_SOURCE = re.compile(r'name = "pyutilz"\nversion = "[^"]+"\nsource = \{ git = "[^"]*#([0-9a-f]{40})" \}')


def read_pyutilz_sha(lock_text: str) -> str:
    """The 40-hex commit SHA of the git-pinned pyutilz package in ``lock_text``."""
    match = _PYUTILZ_GIT_SOURCE.search(lock_text)
    if match is None:
        raise ValueError("uv.lock has no git-pinned pyutilz entry")
    return match.group(1)


def main(argv: list[str] | None = None) -> int:
    """Print the pinned SHA of the lock file given as the first argument (default ``uv.lock`` in the working directory)."""
    args = sys.argv[1:] if argv is None else argv
    lock = Path(args[0]) if args else Path("uv.lock")
    try:
        print(read_pyutilz_sha(lock.read_text(encoding="utf-8")))
    except ValueError as exc:
        print(f"read_pyutilz_pin: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover -- CLI entry
    raise SystemExit(main())
