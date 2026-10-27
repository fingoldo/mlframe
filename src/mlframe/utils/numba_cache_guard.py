"""Remove numba on-disk caches that a changed callee in ANOTHER file has made stale.

numba's ``cache=True`` invalidates a function when its own source file changes, but not when a function it calls lives in a different file and changed: the cached caller keeps the old
callee compiled in and silently runs the old code. This happened in this repository (a split-search change in ``supervised_binning`` never reached the tree in ``_numba_utils_cross_module``).

For every module of a package that defines a ``cache=True`` kernel, a stamp is kept in the package's ``__pycache__``: the hash of the module's own source plus the sources of the sibling
modules it imports (transitively). When the stamp differs from the stored one, the module's ``.nbi`` / ``.nbc`` files are deleted, so numba recompiles it on the next call. Modules whose stamp is
unchanged keep their caches (an edit costs the recompile of the modules that depend on it, not of the whole package).
"""

from __future__ import annotations

import contextlib
import hashlib
import re
from pathlib import Path

_STAMP_DIR_NAME = "__pycache__"
_STAMP_FILE = "mlframe_numba_dependency_stamps.txt"
_RELATIVE_IMPORT = re.compile(r"^\s*from\s+\.+(\w+)?\s+import\s+([^\n(]+|\([^)]*\))", re.MULTILINE)
_RELATIVE_MODULE_IMPORT = re.compile(r"^\s*from\s+\.+(\w+)\s+import", re.MULTILINE)


def _unlink_quietly(path: Path) -> None:
    """Delete ``path``; a file another process holds or removed first is left alone (the guard is a convenience)."""
    with contextlib.suppress(OSError):
        path.unlink()


def _defines_cached_kernel(source: str) -> bool:
    """True when the source declares an ``njit`` / ``jit`` with ``cache=True``."""
    return "cache=True" in source and "njit" in source


def _sibling_modules(source: str, known: set[str]) -> set[str]:
    """Names of sibling modules of the package imported by relative ``from .x import ...`` statements."""
    return {m for m in _RELATIVE_MODULE_IMPORT.findall(source) if m in known}


def _stamps(package_dir: Path) -> dict[str, str]:
    """``{module: hash of its source and of its transitive sibling imports}`` for the modules that define cached kernels."""
    sources = {p.stem: p.read_text(encoding="utf-8", errors="ignore") for p in package_dir.glob("*.py")}
    known = set(sources)
    deps = {name: _sibling_modules(text, known) for name, text in sources.items()}
    out: dict[str, str] = {}
    for name, text in sources.items():
        if not _defines_cached_kernel(text):
            continue
        seen: set[str] = set()
        stack = [name]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(deps.get(cur, ()))
        digest = hashlib.sha256()
        for dep in sorted(seen):
            digest.update(dep.encode())
            digest.update(sources[dep].encode("utf-8", errors="ignore"))
        out[name] = digest.hexdigest()
    return out


def invalidate_stale_numba_caches(package_dir: "str | Path") -> list[str]:
    """Delete the numba cache files of the modules of ``package_dir`` whose dependency stamp changed since the last call; returns the module names cleared (a first call only records the stamps)."""
    package_dir = Path(package_dir)
    cache_dir = package_dir / _STAMP_DIR_NAME
    stamp_path = cache_dir / _STAMP_FILE
    current = _stamps(package_dir)
    previous: dict[str, str] = {}
    if stamp_path.exists():
        for line in stamp_path.read_text(encoding="utf-8").splitlines():
            name, _, digest = line.partition(" ")
            if digest:
                previous[name] = digest
    cleared: list[str] = []
    if previous:
        for name, digest in current.items():
            if previous.get(name, digest) != digest and cache_dir.is_dir():
                for pattern in (f"{name}.*.nbi", f"{name}.*.nbc"):
                    for stale in cache_dir.glob(pattern):
                        _unlink_quietly(stale)
                cleared.append(name)
    cache_dir.mkdir(exist_ok=True)
    stamp_path.write_text("\n".join(f"{name} {digest}" for name, digest in sorted(current.items())) + "\n", encoding="utf-8")
    return cleared
