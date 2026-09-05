"""Guard: no Python-3.10-only annotation form may appear in the dataset package's pydantic models.

pydantic v2 EVALUATES annotations at class-creation time, so a PEP-604 ``X | None`` or a bare builtin
generic ``dict[str, X]`` inside a ``BaseModel`` raises ``TypeError`` on IMPORT under Python 3.9 - which this
project's ``requires-python`` still allows and which CI genuinely runs. ``from __future__ import
annotations`` does not save a ``BaseModel`` (it saves plain dataclasses, which is what makes the mistake so
easy to make), and mypy here targets 3.10 and stays silent.

This is an AST guard rather than a real 3.9 import because no 3.9 interpreter in this environment has
pydantic and numpy installed; a real ``py -3.9 -c "import mlframe.data.datasets"`` remains the stronger
check wherever such an interpreter exists.
"""

import ast
from pathlib import Path

import pytest

import mlframe.data.datasets as datasets_pkg

_PACKAGE_DIR = Path(datasets_pkg.__file__).resolve().parent

# Builtin containers whose subscripted form (``dict[str, int]``) is a syntax error to EVALUATE on 3.9,
# even though it parses there.
_BARE_GENERICS = {"dict", "list", "tuple", "set", "frozenset", "type"}


def _annotation_nodes(tree: ast.AST):
    """Yield every AST node that is used as an annotation somewhere in ``tree``.

    Args:
        tree: Parsed module.

    Yields:
        The annotation expression nodes of assignments, function arguments and return types.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and node.annotation is not None:
            yield node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                yield node.returns
            for arg in [*node.args.args, *node.args.posonlyargs, *node.args.kwonlyargs]:
                if arg.annotation is not None:
                    yield arg.annotation


def _offenders(path: Path):
    """Return every 3.10-only annotation form found in one module.

    Args:
        path: Module to scan.

    Returns:
        A list of ``"<form> at line N"`` strings, empty when the module is 3.9-evaluable.
    """
    tree = ast.parse(path.read_bytes().decode("utf-8"))
    found = []
    for annotation in _annotation_nodes(tree):
        for node in ast.walk(annotation):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
                found.append(f"PEP-604 union at line {node.lineno}")
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                if node.value.id in _BARE_GENERICS:
                    found.append(f"bare generic {node.value.id}[...] at line {node.lineno}")
    return found


@pytest.mark.parametrize("module_name", ["spec.py", "ground_truth.py", "_rng.py", "_scm.py", "__init__.py"])
def test_no_py310_only_annotation_forms(module_name):
    """Every core module stays evaluable under Python 3.9's typing rules."""
    path = _PACKAGE_DIR / module_name
    assert not _offenders(path), f"{module_name} uses 3.10-only annotation forms: {_offenders(path)}"


def test_guard_detects_a_planted_offender(tmp_path):
    """The guard fails on code that would actually break 3.9, so a green run means something."""
    planted = tmp_path / "planted.py"
    planted.write_bytes(b"from pydantic import BaseModel\n\n\nclass M(BaseModel):\n    a: int | None = None\n    b: dict[str, int] = {}\n")
    offenders = _offenders(planted)
    assert any("PEP-604" in item for item in offenders)
    assert any("bare generic dict" in item for item in offenders)
