"""Makes numba's on-disk cache notice edits to a kernel's callees in OTHER mlframe source files.

numba stamps a ``cache=True`` kernel's cache index with the mtime and size of the kernel's OWN source file only. A kernel that calls another
``@njit`` function defined in a different file keeps serving the machine code compiled against the callee's old body after the callee is edited,
so the edit silently has no effect until the caller's own file is touched or the cache directory is wiped.

A cache locator installed ahead of numba's own widens that stamp: it appends the (mtime, size) of every mlframe file that the kernel's file reaches
through ``from <mlframe module> import <jit function>`` (followed transitively, re-exports included). Editing a callee file then invalidates exactly
the kernels that depend on it, and nothing else. Kernels reaching a callee through a module attribute (``mod.func``) are not tracked.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

# top-level package name -> its source directory; kernels in tracked packages get the dependency-aware stamp.
_PACKAGES: Dict[str, str] = {"mlframe": os.path.dirname(os.path.abspath(__file__))}
_LOCK = threading.RLock()
_INSTALLED = False


_JIT_DECORATOR_RE = re.compile(r"^@(?:\w+\.)*(?:njit|jit|vectorize|guvectorize|stencil|cfunc)\b", re.MULTILINE)
_DEF_RE = re.compile(r"^def (\w+)\(", re.MULTILINE)
_FROM_IMPORT_RE = re.compile(r"^[ \t]*from[ \t]+([\w.]+)[ \t]+import[ \t]+(\([^)]*\)|[^\n]*)", re.MULTILINE)
_COMMENT_RE = re.compile(r"#[^\n]*")


class _Import(NamedTuple):
    """One ``from <module> import <name> as <local>`` of an mlframe module."""

    local: str
    name: str
    module: str


class _FileFacts(NamedTuple):
    """Parsed view of one source file: its top-level jit functions and its from-imports of mlframe names."""

    jit_names: frozenset
    imports: Tuple[_Import, ...]


# path -> (stamp, facts); facts are re-derived when the file's stamp moves.
_FACTS: Dict[str, Tuple[Tuple[float, int], _FileFacts]] = {}


def _stamp(path: str) -> Optional[Tuple[float, int]]:
    """(mtime, size) of ``path`` or None when it cannot be stat'ed."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime, st.st_size)


def _package_of(path: str) -> Optional[str]:
    """Name of the tracked package whose directory contains ``path``."""
    full = os.path.normcase(os.path.abspath(path))
    for name, root in _PACKAGES.items():
        prefix = os.path.normcase(root)
        if full == prefix or full.startswith(prefix + os.sep):
            return name
    return None


def _module_name_for(path: str) -> str:
    """Dotted module name of a file under a tracked package ('' when outside all of them)."""
    pkg = _package_of(path)
    if pkg is None:
        return ""
    rel = os.path.relpath(os.path.abspath(path), os.path.dirname(_PACKAGES[pkg]))
    name = rel[:-3].replace(os.sep, ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def _module_path(module: str) -> Optional[str]:
    """Source file of a tracked-package module, resolved without importing it."""
    root = _PACKAGES.get(module.split(".")[0])
    if root is None:
        return None
    base = os.path.join(os.path.dirname(root), *module.split("."))
    for candidate in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(candidate):
            return candidate
    return None


def _import_target(module_text: str, module_name: str, is_pkg: bool) -> str:
    """Absolute dotted module a ``from <module_text> import ...`` statement refers to."""
    level = len(module_text) - len(module_text.lstrip("."))
    if not level:
        return module_text
    parts = module_name.split(".")
    base = ".".join(parts[: max(len(parts) - (level - (1 if is_pkg else 0)), 0)])
    rest = module_text[level:]
    return f"{base}.{rest}" if rest else base


def _facts(path: str) -> Optional[_FileFacts]:
    """Facts for ``path`` (memoised on the file stamp); read by regex because parsing every kernel file's AST costs seconds at import."""
    stamp = _stamp(path)
    if stamp is None:
        return None
    cached = _FACTS.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    module_name = _module_name_for(path)
    is_pkg = os.path.basename(path) == "__init__.py"
    jit_names = set()
    for deco in _JIT_DECORATOR_RE.finditer(text):
        fn = _DEF_RE.search(text, deco.end())
        if fn is not None:
            jit_names.add(fn.group(1))
    imports: List[_Import] = []
    for m in _FROM_IMPORT_RE.finditer(text):
        target = _import_target(m.group(1), module_name, is_pkg)
        if target.split(".")[0] not in _PACKAGES:
            continue
        for item in _COMMENT_RE.sub("", m.group(2)).strip("() \t\r\n").split(","):
            pieces = item.split()
            if pieces and pieces[0] != "*":
                imports.append(_Import(pieces[-1], pieces[0], target))
    facts = _FileFacts(frozenset(jit_names), tuple(imports))
    _FACTS[path] = (stamp, facts)
    return facts


def _resolve_jit(module: str, name: str, seen: Set[Tuple[str, str]]) -> Optional[str]:
    """Source file defining jit function ``name`` as reached through ``module`` (following re-exports), or None."""
    if (module, name) in seen:
        return None
    seen.add((module, name))
    path = _module_path(module)
    facts = _facts(path) if path is not None else None
    if path is None or facts is None:
        return None
    if name in facts.jit_names:
        return path
    for imp in facts.imports:
        if imp.local == name:
            hit = _resolve_jit(imp.module, imp.name, seen)
            if hit is not None:
                return hit
    return None


def dependency_files(py_file: str) -> List[str]:
    """Other mlframe source files whose jit functions the kernels of ``py_file`` import, transitively (sorted, ``py_file`` excluded)."""
    start = os.path.abspath(py_file)
    found: Set[str] = set()
    queue = [start]
    visited: Set[str] = set()
    while queue:
        current = queue.pop()
        if current in visited:
            continue
        visited.add(current)
        facts = _facts(current)
        if facts is None:
            continue
        for imp in facts.imports:
            dep = _resolve_jit(imp.module, imp.name, set())
            if dep is not None:
                dep = os.path.abspath(dep)
                if dep != start and dep not in found:
                    found.add(dep)
                    queue.append(dep)
    return sorted(found)


def dependency_digest(py_file: str) -> str:
    """Digest of the (path, mtime, size) of every dependency file of ``py_file``; empty string when there are none."""
    deps = dependency_files(py_file)
    if not deps:
        return ""
    h = hashlib.sha1(usedforsecurity=False)
    for dep in deps:
        h.update(f"{_module_name_for(dep)}|{_stamp(dep)}".encode())
    return h.hexdigest()


def track_package(name: str, root: str) -> None:
    """Also give kernels under ``root`` (the directory of top-level package ``name``) the dependency-aware stamp."""
    with _LOCK:
        _PACKAGES[name] = os.path.abspath(root)


def _make_locator_class() -> type:
    """Build the locator class (deferred so importing this module does not import numba)."""
    from numba.core import caching

    class DependencyAwareLocator(caching._CacheLocator):
        """Delegates paths to numba's regular locators but widens the source stamp with the cross-file dependency digest."""

        def __init__(self, inner: Any, py_file: str) -> None:
            self._inner = inner
            self._py_file = py_file

        def get_cache_path(self) -> str:
            """Cache directory chosen by the delegate locator."""
            return str(self._inner.get_cache_path())

        def get_source_stamp(self) -> object:
            """The delegate's (mtime, size) of the kernel's own file plus the digest of its callee files."""
            return (self._inner.get_source_stamp(), dependency_digest(self._py_file))

        def get_disambiguator(self) -> str:
            """Disambiguator chosen by the delegate locator."""
            return str(self._inner.get_disambiguator())

        @classmethod
        def from_function(cls, py_func: Any, py_file: str) -> Optional["DependencyAwareLocator"]:
            """A locator for kernels in mlframe files, wrapping the first regular numba locator that accepts the function."""
            if _package_of(py_file) is None:
                return None
            for base in (caching.UserProvidedCacheLocator, caching.InTreeCacheLocator, caching.UserWideCacheLocator):
                inner = base.from_function(py_func, py_file)
                if inner is not None:
                    return cls(inner, py_file)
            return None

    return DependencyAwareLocator


def install() -> bool:
    """Put the dependency-aware locator ahead of numba's own; idempotent. Returns True once installed, False when numba is unavailable."""
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return True
        try:
            from numba.core import caching
        except ImportError:
            return False
        locators = getattr(caching.CacheImpl, "_locator_classes", None)
        if locators is None:
            return False
        locators.insert(0, _make_locator_class())
        _INSTALLED = True
        return True
