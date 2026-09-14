"""No FE module may discretise a float target with a truncating ``astype(np.int64)`` (mrmr_audit_2026-09-14 RO-4/IMPL-4).

The pattern ``if y.dtype.kind in "fc": if <distinct count> <= 32: y = y.astype(np.int64)`` treats "fewer than 33 distinct
float values" as "integral-valued", which it is not: ``{0.0, 0.5, 1.0, 1.5}`` truncates to two classes and a target of
levels inside ``[0, 1)`` collapses to one, so every MI score against it reads ~0 and the family becomes a silent no-op.
It was copy-pasted to 19 live sites, including one outside the MRMR cluster the audit agent covered. The shared fix is
``_y_encoding.encode_y_for_classif_mi``. This gate walks the AST rather than grepping, so it flags the construct itself
wherever it reappears, and it deliberately passes the two legitimate look-alikes: an ``astype(np.int64)`` inside an ``if``
TEST (an integrality check), and a cast applied to codes that ``np.unique(..., return_inverse=True)`` already produced.
"""

import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection"


def _is_int64_cast_of(node: ast.AST, name: str) -> bool:
    """``<name>.astype(np.int64)`` (optionally chained after another call, e.g. ``pd.qcut(...).astype(np.int64)``)."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "astype"):
        return False
    if not (node.args and isinstance(node.args[0], ast.Attribute) and node.args[0].attr == "int64"):
        return False
    return isinstance(node.func.value, ast.Name) and node.func.value.id == name


def _compares_against_32(test: ast.AST) -> bool:
    """True when an ``if`` test compares something against the literal 32."""
    return any(isinstance(n, ast.Compare) and any(isinstance(c, ast.Constant) and c.value == 32 for c in n.comparators) for n in ast.walk(test))


def _densified_earlier(stmts: list, name: str, upto: int) -> bool:
    """An earlier statement in the same block rebinds ``name`` from ``np.unique(..., return_inverse=True)`` -- it is already codes."""
    for st in stmts[:upto]:
        if isinstance(st, ast.Assign) and isinstance(st.value, ast.Call):
            f = st.value.func
            if isinstance(f, ast.Attribute) and f.attr == "unique" and any(k.arg == "return_inverse" for k in st.value.keywords):
                targets = [t for tgt in st.targets for t in (tgt.elts if isinstance(tgt, ast.Tuple) else [tgt])]
                if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                    return True
    return False


def find_truncating_y_casts(root: pathlib.Path = _ROOT) -> list[str]:
    """Every ``<y> = <y>.astype(np.int64)`` assigned in the body of an ``if`` that compares a count against 32."""
    hits: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "_benchmarks" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.If) and _compares_against_32(node.test)):
                continue
            for stmts in (node.body, node.orelse):
                for i, st in enumerate(stmts):
                    if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)):
                        continue
                    name = st.targets[0].id
                    if _is_int64_cast_of(st.value, name) and not _densified_earlier(stmts, name, i):
                        hits.append(f"{path.relative_to(root).as_posix()}:{st.lineno}")
    return hits


def test_no_fe_module_truncates_a_float_target_to_int64():
    """The shared encoder is the only correct path; a truncating cast anywhere is the RO-4 bug back again."""
    hits = find_truncating_y_casts()
    assert hits == [], "truncating float-target discretisation reintroduced -- use encode_y_for_classif_mi:\n  " + "\n  ".join(hits)
