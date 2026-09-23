"""Frame copies in the composite and suite paths are recorded, and the record may only shrink.

A copy of the train frame is the pipeline's most expensive line: on the sizes this framework targets it is tens of GB, and
the composite paths run per target, so one careless ``.copy()`` multiplies by K. The copies that exist today are listed in
``_frame_copy_baseline.json``; a new one has to be argued for, and a line that genuinely needs one says so inline with
``# frame-copy: <reason>``, which takes it out of the count. A shallow ``copy(deep=False)`` shares the blocks it copies the
handle of, so it is not a frame copy and is not counted.
"""

from __future__ import annotations

import ast
import orjson
from collections import Counter
from pathlib import Path

import mlframe

from tests.test_meta._shared_ast_cache import parsed_ast

_PKG = Path(mlframe.__file__).resolve().parent
_SCOPE = (_PKG / "training" / "composite", _PKG / "training" / "core")
_BASELINE = Path(__file__).resolve().parent / "_frame_copy_baseline.json"
_MARKER = "# frame-copy:"


def _is_frame_name(node: ast.AST) -> bool:
    """True when ``node`` is a name or attribute that reads as a frame (``train_df``, ``self.df_``, ``disc_frame``)."""
    name = getattr(node, "id", getattr(node, "attr", None))
    if not isinstance(name, str):
        return False
    low = name.lower().rstrip("_")
    return low.endswith(("df", "frame", "data")) or low in {"x", "pool"}


def frame_copies(tree: ast.Module, source: str) -> list[tuple[str, int]]:
    """``(what, line)`` for every unmarked frame copy: ``<frame>.copy()``, ``<frame>.clone()`` or ``concat`` of a column selection."""
    lines = source.split("\n")
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        what = None
        shallow = any(k.arg == "deep" and getattr(k.value, "value", None) is False for k in node.keywords)
        if isinstance(func, ast.Attribute) and func.attr in ("copy", "clone") and _is_frame_name(func.value) and not shallow:
            what = f"{getattr(func.value, 'id', getattr(func.value, 'attr', '?'))}.{func.attr}"
        elif isinstance(func, ast.Attribute) and func.attr == "concat" and node.args:
            first = node.args[0]
            if isinstance(first, (ast.List, ast.Tuple)) and any(
                isinstance(e, ast.Subscript) and _is_frame_name(e.value) and not isinstance(e.slice, ast.Constant) for e in first.elts
            ):
                what = "concat(column selection)"
        if what is not None and _MARKER not in lines[node.lineno - 1]:
            out.append((what, node.lineno))
    return sorted(set(out), key=lambda t: t[1])


def _scan() -> Counter:
    """``path::what`` counts over the composite and core packages."""
    counts: Counter = Counter()
    scanned = 0
    for root in _SCOPE:
        for path in sorted(root.rglob("*.py")):
            if "_benchmarks" in path.parts:
                continue
            tree = parsed_ast(path)
            if tree is None:
                continue
            scanned += 1
            source = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
            for what, _line in frame_copies(tree, source):
                counts[f"{path.relative_to(_PKG).as_posix()}::{what}"] += 1
    assert scanned >= 250, f"scanned only {scanned} modules; the scope no longer matches the tree"
    return counts


def test_no_new_frame_copies():
    """The recorded copies may only shrink; a removed one must leave the baseline with it."""
    got = _scan()
    recorded = Counter(orjson.loads(_BASELINE.read_text(encoding="utf-8")))
    new = {k: v - recorded.get(k, 0) for k, v in got.items() if v > recorded.get(k, 0)}
    gone = {k: v - got.get(k, 0) for k, v in recorded.items() if v > got.get(k, 0)}
    assert not new, f"drop the copy, or justify it inline with '{_MARKER} <reason>': {new}"
    assert not gone, f"these recorded copies are gone; lower them in {_BASELINE.name}: {gone}"


def test_the_scan_sees_a_copy_and_honours_the_marker():
    """Canary: an unmarked frame copy and a column-selection concat are flagged; a marked, a shallow and a series copy are not."""
    src = (
        "def f(train_df, s, pd):\n"
        "    a = train_df.copy()\n"
        "    b = s.copy()\n"
        "    c = train_df.copy()  # frame-copy: the caller mutates it\n"
        "    d = pd.concat([train_df[cols], y], axis=1)\n"
        "    e = train_df.copy(deep=False)\n"
        "    return a, b, c, d, e\n"
    )
    assert frame_copies(ast.parse(src), src) == [("train_df.copy", 2), ("concat(column selection)", 5)]
