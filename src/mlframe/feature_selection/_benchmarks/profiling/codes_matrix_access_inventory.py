"""Inventory of the readers of the (n_rows, n_columns) codes matrix: how each function walks it.

Run ``python -m mlframe.feature_selection._benchmarks.profiling.codes_matrix_access_inventory [out.md]``.

The matrix is C-ordered, one row per sample. A reader that fixes a column and runs over the rows (``data[:, j]``, or ``data[i, j]`` with ``i`` the innermost loop variable) strides ``n_columns * itemsize`` bytes between
neighbouring reads: on a GPU the lanes of a warp then touch different cache lines, on a CPU every read is a cache miss. Stored feature-major ``(n_columns, n_rows)`` the same reads are contiguous.
The reader that fixes a row and runs over the columns (``data[i, :]``, or ``data[i, j]`` with ``j`` innermost) is the one the row-major layout serves.

Per function with a parameter named like the matrix the inventory counts the subscripts of that parameter by access kind:

* ``column`` - ``name[:, j]`` / ``name[:, a:b]`` (a column or a column block, rows contiguous only in feature-major);
* ``row`` - ``name[i, :]`` (a row, contiguous in row-major);
* ``rows_inner`` - ``name[i, j]`` where the innermost enclosing loop variable is in the ROW position (strided in row-major);
* ``cols_inner`` - ``name[i, j]`` where the innermost loop variable is in the COLUMN position (contiguous in row-major);
* ``other`` - a single-index subscript (a vector element or a matrix row: the rank is not known) or a scalar subscript whose indices are not loop variables of the function.

A function leans feature-major when ``column + rows_inner`` exceeds ``row + cols_inner``. CUDA ``RawKernel`` sources are strings, not code the AST sees: the files that hold them are listed apart for reading by hand.
The result is a count of lines of evidence, not a measurement; a reader the inventory puts on the feature-major side still has to be timed before its layout is changed.
"""

from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

MATRIX_PARAMS = frozenset({"factors_data", "data", "codes", "codes_matrix", "sub", "cm", "factors"})
_SKIP_DIRS = {"_benchmarks", "__pycache__", "tests"}


def _loop_vars(node: ast.AST) -> set[str]:
    """Names bound by a ``for`` target (``range`` / ``prange`` index variables included)."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} if isinstance(node, ast.AST) else set()


def _jit_kind(func: ast.FunctionDef) -> str:
    """``cuda.jit`` / ``njit`` / ``python`` from the decorators."""
    text = " ".join(ast.unparse(d) for d in func.decorator_list)
    if "cuda.jit" in text or ("cuda" in text and "jit" in text):
        return "cuda.jit"
    if "njit" in text or "numba.jit" in text or "jit(" in text:
        return "njit"
    return "python"


class _AccessCounter(ast.NodeVisitor):
    """Counts the subscripts of the parameter ``name`` by access kind, tracking the loop variables that enclose each."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.loops: list[set[str]] = []
        self.counts: Counter = Counter()

    def visit_For(self, node: ast.For) -> None:
        """Enter a loop: its target names are loop variables inside the body."""
        self.visit(node.iter)
        self.loops.append(_loop_vars(node.target))
        for child in node.body + node.orelse:
            self.visit(child)
        self.loops.pop()

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Classify ``name[...]``."""
        if isinstance(node.value, ast.Name) and node.value.id == self.name:
            self.counts[self._kind(node.slice)] += 1
        self.generic_visit(node)

    def _kind(self, sl: ast.AST) -> str:
        """The access kind of one subscript."""
        if not isinstance(sl, ast.Tuple) or len(sl.elts) != 2:
            return "other"  # a single index: an element of a vector or a row of a matrix, the parameter's rank is not known here
        first, second = sl.elts
        if isinstance(first, ast.Slice) and not isinstance(second, ast.Slice):
            return "column"
        if isinstance(first, ast.Slice) and isinstance(second, ast.Slice):
            return "other"
        if isinstance(second, ast.Slice):
            return "row"
        inner = self.loops[-1] if self.loops else set()
        in_first = bool(_loop_vars(first) & inner)
        in_second = bool(_loop_vars(second) & inner)
        if in_first and not in_second:
            return "rows_inner"
        if in_second and not in_first:
            return "cols_inner"
        return "other"


def inventory(root: Path) -> list[dict]:
    """One record per function with a matrix parameter that subscripts it, under ``root``."""
    records: list[dict] = []
    for path in sorted(root.rglob("*.py")):
        if _SKIP_DIRS & set(path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef):
                continue
            params = {a.arg for a in func.args.args + func.args.kwonlyargs} & MATRIX_PARAMS
            for name in sorted(params):
                counter = _AccessCounter(name)
                for stmt in func.body:
                    counter.visit(stmt)
                if not counter.counts:
                    continue
                c = counter.counts
                records.append(
                    {
                        "file": str(path.relative_to(root)).replace("\\", "/"),
                        "function": func.name,
                        "param": name,
                        "kind": _jit_kind(func),
                        "column": c["column"], "row": c["row"], "rows_inner": c["rows_inner"], "cols_inner": c["cols_inner"], "other": c["other"],
                        "leans": "feature-major" if c["column"] + c["rows_inner"] > c["row"] + c["cols_inner"] else ("row-major" if c["row"] + c["cols_inner"] > 0 else "neutral"),
                    }
                )
    return records


def raw_kernel_files(root: Path) -> list[str]:
    """Files that hold CUDA source strings (``__global__``), to be read by hand."""
    out = []
    for path in sorted(root.rglob("*.py")):
        if _SKIP_DIRS & set(path.parts):
            continue
        try:
            if "__global__" in path.read_text(encoding="utf-8"):
                out.append(str(path.relative_to(root)).replace("\\", "/"))
        except UnicodeDecodeError:
            continue
    return out


def render(records: list[dict], kernels: list[str]) -> str:
    """The inventory as a markdown report."""
    by = Counter((r["kind"], r["leans"]) for r in records)
    lines = ["# Readers of the codes matrix", "", f"{len(records)} (function, parameter) pairs subscript a matrix parameter.", "", "| reader kind | leans feature-major | leans row-major | neutral |", "|---|---|---|---|"]
    for kind in ("cuda.jit", "njit", "python"):
        lines.append(f"| {kind} | {by[(kind, 'feature-major')]} | {by[(kind, 'row-major')]} | {by[(kind, 'neutral')]} |")
    lines += ["", "## Functions that lean feature-major", "", "| file | function | param | kind | column | rows_inner | row | cols_inner |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in records if r["leans"] == "feature-major"), key=lambda r: (r["kind"] != "cuda.jit", r["kind"] != "njit", -(r["column"] + r["rows_inner"]))):
        lines.append(f"| {r['file']} | {r['function']} | {r['param']} | {r['kind']} | {r['column']} | {r['rows_inner']} | {r['row']} | {r['cols_inner']} |")
    lines += ["", "## Functions that lean row-major", "", "| file | function | param | kind | row | cols_inner | column | rows_inner |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in records if r["leans"] == "row-major"), key=lambda r: (r["kind"] != "cuda.jit", r["kind"] != "njit", -(r["row"] + r["cols_inner"]))):
        lines.append(f"| {r['file']} | {r['function']} | {r['param']} | {r['kind']} | {r['row']} | {r['cols_inner']} | {r['column']} | {r['rows_inner']} |")
    lines += ["", "## Files with CUDA source strings (read by hand)", ""] + [f"- {k}" for k in kernels]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    """Write the report to ``argv[1]`` (default: print)."""
    root = Path(__file__).resolve().parents[2] / "filters"
    text = render(inventory(root), raw_kernel_files(root))
    if len(argv) > 1:
        Path(argv[1]).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
