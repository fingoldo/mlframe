"""Meta-test: no ``@njit``/``@torch.jit.script`` function may call ``typing.cast``.

``cast(T, val)`` is a pure static-typing no-op at runtime (``typing.cast`` literally
``return val``), which is exactly why the mypy-clean-code sweep wraps numpy/pandas/torch
arithmetic-chain returns in it -- to satisfy mypy's ``no-any-return`` without changing
behavior. But ``cast`` is a real Python function CALL from a compiler's point of view, and
neither numba nor TorchScript know what to do with it:

- TorchScript (``@torch.jit.script``): ``RuntimeError: builtin cannot be used as a value``
  at compile time -- confirmed on ``_ranker_losses._ranknet_loss_precomputed_core``, added by
  commit f430e085b ("mypy: fix all 10 remaining errors ... closing out training/neural") and
  broke import-time collection of every test that imports ``ranker.py`` (2026-07-10).
- numba ``@njit``: ``TypingError: Untyped global name 'cast': Cannot determine Numba type of
  <class 'function'>`` -- confirmed empirically the same day; no known occurrence in this repo
  yet, but the same mypy-sweep pattern could introduce one in a future pass.

The check is a structural ``ast`` walk over the installed package source, so it matches CALLS
only -- mentions in docstrings/comments are ignored. Fix for any real finding: drop the
``cast(...)`` wrapper inside the compiled function body (it changes nothing at runtime); keep
it in eager/wrapper code around the compiled call if mypy still needs the hint there.
"""

from __future__ import annotations

import ast
from pathlib import Path

import mlframe

from tests.test_meta._shared_ast_cache import parsed_ast

PKG_ROOT = Path(mlframe.__file__).resolve().parent


def _is_compiled_decorator(dec: ast.expr) -> bool:
    """True for ``@njit(...)``/``@numba.njit(...)``/``@jit(nopython=True)`` (CPU nopython) and
    ``@torch.jit.script``. Plain ``@numba.cuda.jit`` is excluded -- CUDA kernels have their own
    typing surface, not exercised by this check."""
    target = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(target, ast.Attribute):
        name = target.attr
        if isinstance(target.value, ast.Attribute) and target.value.attr == "cuda":
            return False
        return name in ("njit", "jit", "script")
    if isinstance(target, ast.Name):
        return target.id in ("njit", "jit")
    return False


def _is_cast_call(node: ast.Call) -> bool:
    """True for a bare ``cast(...)`` or ``<module>.cast(...)`` call."""
    f = node.func
    if isinstance(f, ast.Name):
        return f.id == "cast"
    if isinstance(f, ast.Attribute):
        return f.attr == "cast"
    return False


def _cast_violations(tree: ast.AST, rel: object) -> list[str]:
    """``cast(...)`` calls inside compiled functions of one parsed module, as report lines."""
    violations: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(_is_compiled_decorator(d) for d in fn.decorator_list):
            continue
        for call in ast.walk(fn):
            if isinstance(call, ast.Call) and _is_cast_call(call):
                violations.append(f"{rel}:{call.lineno}  {fn.name}() calls cast(...)")
    return violations


def test_cast_detector_catches_cast_in_njit_and_passes_eager_cast():
    """``cast`` inside an ``@njit`` body is reported; the same call in a plain function is not."""
    bad = ast.parse("@njit\ndef f(x):\n    return cast(int, x)\n")
    clean = ast.parse("def f(x):\n    return typing.cast(int, x)\n")
    assert _cast_violations(bad, "m.py") == ["m.py:3  f() calls cast(...)"]
    assert _cast_violations(clean, "m.py") == []


def test_no_cast_call_in_compiled_functions():
    """No ``cast(...)`` call inside the body of an ``@njit``/``@torch.jit.script`` function (dead at runtime)."""
    violations: list[str] = []
    scanned = 0
    for path in PKG_ROOT.rglob("*.py"):
        tree = parsed_ast(path)
        if tree is None:
            continue
        scanned += 1
        violations.extend(_cast_violations(tree, path.relative_to(PKG_ROOT.parent)))
    assert scanned > 0, "no package sources were scanned"
    assert not violations, (
        "typing.cast() called inside a compiled (@njit / @torch.jit.script) function -- "
        "both TorchScript and numba nopython reject it as an unresolvable builtin call. "
        "cast() is a runtime no-op; drop the wrapper inside the compiled body:\n  " + "\n  ".join(sorted(violations))
    )
