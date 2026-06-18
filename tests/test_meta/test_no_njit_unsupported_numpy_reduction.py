"""Meta-test: no ``@njit`` function may CALL ``np.median`` (use np.sort instead).

``np.median`` inside ``@njit`` compiled fine on numba 0.65.1 in a plain env
(Windows / py3.14, even under xdist+cache) yet raised ``TypingError: Use of
unsupported NumPy function 'numpy.median'`` deterministically on the CI Linux
runners (numba 0.65.1 / numpy 2.4.6), cascading into dozens of MRMR / LtR / FE
failures via ``feature_selection/filters/hermite_fe/_hermite_robust.py``. The
exact mechanism is a CI-Linux numba quirk we could not reproduce locally, but
``np.median`` lives in numba's lazily-registered ``arraymath`` extension while
``np.sort`` is always-core -- so deriving the median from ``np.sort`` (which is
bit-identical: numpy averages the two middle order statistics for even n, as
does a full sort + midpoint) sidesteps the whole failure mode.

Scope is deliberately ``np.median`` ONLY: it is the one with direct CI evidence.
``np.percentile`` / ``np.quantile`` are intentionally used in njit kernels
(e.g. discretization ``get_binning_edges``) and verified working under numba
0.65, so they are NOT forbidden here.

The check is a structural ``ast`` walk over the installed package source, so it
matches CALLS only -- mentions in docstrings/comments are ignored.
"""
from __future__ import annotations

import ast
from pathlib import Path

import mlframe

PKG_ROOT = Path(mlframe.__file__).resolve().parent

# np.median (+ its NaN variant) -- the reduction with direct CI evidence of a numba
# nopython "unsupported function" failure. Derive from np.sort instead.
FORBIDDEN_IN_NJIT = {"median", "nanmedian"}


def _is_njit_decorator(dec: ast.expr) -> bool:
    """True for ``@njit(...)`` / ``@numba.njit(...)`` / ``@jit(nopython=True)`` style decorators
    (CPU nopython compilation). Plain ``@numba.cuda.jit`` is excluded -- CUDA kernels have their
    own (different) typing surface and are not what bit us here."""
    target = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(target, ast.Attribute):
        name = target.attr
        # numba.cuda.jit -> the value is an Attribute ending in `.cuda`; skip those.
        if isinstance(target.value, ast.Attribute) and target.value.attr == "cuda":
            return False
        return name in ("njit", "jit")
    if isinstance(target, ast.Name):
        return target.id in ("njit", "jit")
    return False


def _forbidden_np_call(node: ast.Call) -> str | None:
    """Return ``np.<name>`` if this call is a forbidden NumPy reduction, else None."""
    f = node.func
    if isinstance(f, ast.Attribute) and f.attr in FORBIDDEN_IN_NJIT:
        base = f.value
        if isinstance(base, ast.Name) and base.id in ("np", "numpy"):
            return f"np.{f.attr}"
    return None


def test_no_unsupported_numpy_reduction_in_njit():
    violations: list[str] = []
    for path in PKG_ROOT.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not any(_is_njit_decorator(d) for d in fn.decorator_list):
                continue
            for call in ast.walk(fn):
                if isinstance(call, ast.Call):
                    bad = _forbidden_np_call(call)
                    if bad:
                        rel = path.relative_to(PKG_ROOT.parent)
                        violations.append(f"{rel}:{call.lineno}  {fn.name}() calls {bad}")

    if violations:
        raise AssertionError(
            "Forbidden NumPy reduction(s) called inside @njit functions (numba nopython support "
            "is version-fragile -- derive from np.sort instead, see _hermite_robust._median_sorted_njit):\n  "
            + "\n  ".join(sorted(violations))
        )
