"""Every kernel-tuning cache key that production code reads must be one something writes.

A dispatcher that calls ``cache.lookup("some_key")`` for a key nothing ever writes always gets a miss and silently stays on its conservative default: ``_fe_batch_dispatch`` read ``fe_batch_backend`` for
months while the device was 7 to 11 times faster than the numba path it kept choosing (CT11-122). The scan is static (AST). A key counts as written when it is a literal ``kernel_name=`` of a
``kernel_tuner(...)`` registration, the literal first argument of a cache write (``update`` / ``record`` / ``put`` / ``set_regions`` / ``get_or_tune``), or any string literal of the auto-tune sweep package
``_benchmarks/kernel_tuning_cache``. The keys that are read but written by nothing today are a debt ratchet (``_kernel_tuning_unwritten_keys_baseline.json``, tracker OP11-15): the list may only shrink, a
new unwritten read fails at once, and a key that became written or is no longer read must leave the list.
"""

from __future__ import annotations

import ast
from pathlib import Path

import orjson

SRC = Path(__file__).resolve().parents[2] / "src"
BASELINE = Path(__file__).with_name("_kernel_tuning_unwritten_keys_baseline.json")
_WRITE_METHODS = ("update", "record", "put", "set_regions", "store", "save_regions", "write", "get_or_tune")


def _py_files() -> list[Path]:
    """Production python files (no caches)."""
    return [p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts]


def _literal(node: ast.AST) -> str | None:
    """The string value of a literal node, else None."""
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _scan() -> tuple[dict[str, list[str]], set[str]]:
    """``({lookup key: [file:line, ...]}, {written keys})`` over ``src``."""
    lookups: dict[str, list[str]] = {}
    written: set[str] = set()
    sweep_pkg = SRC / "mlframe" / "feature_selection" / "_benchmarks" / "kernel_tuning_cache"
    for path in _py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        in_sweeps = sweep_pkg in path.parents
        for node in ast.walk(tree):
            if in_sweeps and (lit := _literal(node)):
                written.add(lit)
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg == "kernel_name" and (name := _literal(kw.value)):
                    written.add(name)
            func = node.func
            if not (isinstance(func, ast.Attribute) and node.args and (key := _literal(node.args[0]))):
                continue
            if func.attr == "lookup":
                lookups.setdefault(key, []).append(f"{path.relative_to(SRC)}:{node.lineno}")
            elif func.attr in _WRITE_METHODS:
                written.add(key)
    return lookups, written


def _unwritten() -> dict[str, list[str]]:
    """The keys production code reads that nothing writes, with their read sites."""
    lookups, written = _scan()
    return {key: sites for key, sites in lookups.items() if key not in written}


def test_no_new_cache_key_is_read_without_a_writer():
    """A dispatcher may not read a tuning-cache key nothing writes (a permanent silent miss), except the ratcheted debt."""
    known = set(orjson.loads(BASELINE.read_bytes())["keys"])
    new = {key: sites for key, sites in _unwritten().items() if key not in known}
    assert not new, f"cache keys read by production code that no kernel_tuner / cache write / sweep produces: {new}"


def test_baseline_only_shrinks():
    """A baseline key that is now written, or no longer read, must be removed; the list never grows by hand."""
    known = set(orjson.loads(BASELINE.read_bytes())["keys"])
    still = set(_unwritten())
    stale = sorted(known - still)
    assert not stale, f"baseline keys that are written or no longer read (remove them): {stale}"


def test_known_dead_keys_stay_gone():
    """The key the FE batch dispatcher used to read in vain is not read again."""
    lookups, _written = _scan()
    assert "fe_batch_backend" not in lookups
