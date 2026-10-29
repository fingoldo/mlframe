"""Meta-test: no new ``np.asarray(x).shape[0]`` / ``len(np.asarray(x))``: a copy made only to read a size.

On a plain ndarray ``np.asarray(x)`` is free, so the pattern looks harmless. On a lazy device-backed array (``LazyHostCodes``) it is a full device-to-host copy of the
column, made just to learn how many rows it has and thrown away or repeated by the next reader. Six such sites were found in the feature-engineering stages, each moving a whole
column for a number the array already knew. ``length_of`` and ``size_of`` in ``filters/_lazy_host_codes.py`` read the length without materialising.

Detected, anywhere under ``src/mlframe``: ``np.asarray(...)`` or ``np.array(...)`` (also ``numpy.``) whose result is used only through ``.shape``, ``.size`` or ``.ndim``, and
``len(np.asarray(...))``. A line marked ``# length-ok: <reason>`` is exempt, for an argument that is known to be a host array or a list. Existing hits are frozen in
``_asarray_length_baseline.json`` and may only shrink. Refresh with ``--refresh-asarray-length-baseline``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import orjson
import pytest

import mlframe

from tests.test_meta._shared_ast_cache import parsed_ast, source_text

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_asarray_length_baseline.json"
_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "tests", "legacy", "profiling", "explore", "_benchmarks")
_CONVERTERS = {"asarray", "array"}
_SIZE_ATTRS = {"shape", "size", "ndim"}
_MARKER = "# length-ok:"


def _is_numpy_conversion(node: ast.AST) -> bool:
    """True for a call of ``np.asarray`` / ``np.array`` / ``numpy.asarray`` / ``numpy.array``."""
    if not isinstance(node, ast.Call) or not node.args:
        return False
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr in _CONVERTERS and isinstance(func.value, ast.Name) and func.value.id in {"np", "numpy"}


class _Finder(ast.NodeVisitor):
    """Collects ``(qualified function name, source text, line)`` of every copy-for-size expression in one module."""

    def __init__(self) -> None:
        """Start with an empty scope stack and no findings."""
        self.scope: list[str] = []
        self.found: list[tuple[str, str, int]] = []

    def _scoped(self, node: ast.AST, name: str) -> None:
        """Visit ``node``'s children inside the scope ``name``."""
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track the enclosing function."""
        self._scoped(node, node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track the enclosing class."""
        self._scoped(node, node.name)

    def _record(self, node: ast.AST) -> None:
        """File one hit under the current scope."""
        self.found.append((".".join(self.scope) or "<module>", ast.unparse(node), node.lineno))

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """``np.asarray(x).shape`` / ``.size`` / ``.ndim`` (a subscript of it is reached through this node)."""
        if node.attr in _SIZE_ATTRS and _is_numpy_conversion(node.value):
            self._record(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """``len(np.asarray(x))``."""
        if isinstance(node.func, ast.Name) and node.func.id == "len" and len(node.args) == 1 and _is_numpy_conversion(node.args[0]):
            self._record(node)
        self.generic_visit(node)


def _findings_in_source(src: str, rel: str) -> set[str]:
    """``{rel::function::expression}`` for each unmarked copy-for-size expression in ``src``."""
    lines = src.splitlines()
    finder = _Finder()
    finder.visit(ast.parse(src))
    return {f"{rel}::{scope}::{text}" for scope, text, line in finder.found if _MARKER not in lines[line - 1]}


def _build_offending_set() -> set[str]:
    """Every hit under ``src/mlframe`` outside the exempt folders."""
    out: set[str] = set()
    for py in sorted(MLFRAME_DIR.rglob("*.py")):
        if any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS):
            continue
        src = source_text(py)
        if src is None or parsed_ast(py) is None:
            continue
        out |= _findings_in_source(src, py.relative_to(MLFRAME_DIR).as_posix())
    return out


def _refresh_requested() -> bool:
    """True if ``--refresh-asarray-length-baseline`` was passed on the pytest command line."""
    return "--refresh-asarray-length-baseline" in sys.argv


def test_detector_flags_copies_made_for_a_size_and_nothing_else():
    """The size reads of a fresh conversion are reported; direct attribute reads, plain conversions and marked lines are not."""
    flagged = "import numpy as np\n\ndef f(x):\n    a = np.asarray(x).shape[0]\n    b = len(np.asarray(x))\n    c = np.array(x).size\n    return a + b + c + np.asarray(x).ndim\n"
    assert len(_findings_in_source(flagged, "m.py")) == 4
    clean = "import numpy as np\n\ndef f(x):\n    a = x.shape[0]\n    b = np.asarray(x, dtype=float)\n    return a + b.sum() + np.asarray(x).sum() + len(x)\n"
    assert _findings_in_source(clean, "m.py") == set()
    marked = "import numpy as np\n\ndef f(x):\n    return np.asarray(x).shape[0]  # length-ok: x is a host list\n"
    assert _findings_in_source(marked, "m.py") == set()


def test_no_new_copy_made_only_to_read_a_size():
    """No new ``np.asarray(x).shape[0]``-style expression under ``src/mlframe`` beyond the frozen baseline.

    Use ``length_of(x)`` / ``size_of(x)`` from ``mlframe.feature_selection.filters._lazy_host_codes`` when ``x`` may be a lazy device-backed array,
    read ``x.shape`` directly when it is known to have one, or mark a line whose argument is a host list or array with ``# length-ok: <reason>``.
    """
    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(orjson.dumps(sorted(_build_offending_set()), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"asarray-length baseline refreshed at {_BASELINE_PATH.name}")
    current = _build_offending_set()
    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)
    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_copy_made_only_to_read_a_size] {len(fixed)} site(s) no longer present:\n  "
            + "\n  ".join(fixed[:15])
            + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + "\n  Refresh: pytest tests/test_meta/test_no_asarray_for_length.py --refresh-asarray-length-baseline\n"
        )
    if new:
        pytest.fail(
            f"{len(new)} new expression(s) copy an array only to read its size; on a lazy device-backed array that is a full device-to-host transfer. "
            "Use length_of / size_of (filters/_lazy_host_codes.py), read .shape directly, or mark the line '# length-ok: <reason>':\n  "
            + "\n  ".join(new[:30])
            + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )
