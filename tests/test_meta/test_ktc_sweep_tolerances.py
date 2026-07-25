"""Meta-test: a kernel-tuning equivalence sweep must not use a tolerance far looser than its own claim.

``sweep_backend_grid(..., equiv_rtol=, equiv_atol=)`` decides, on each host, whether a candidate backend is
"equivalent enough" to be crowned the fastest. When the two variants are the SAME arithmetic differing only in
float64 reduction order, their real divergence is ~1e-12; a 5e-2 tolerance is then thirteen orders too loose and
can crown a genuinely divergent kernel - on MI values in the 0.001-0.5 range an ABSOLUTE 5e-2 can exceed the
value being compared.

Sweeps that compare genuinely DIFFERENT algorithms (a rank binner vs a percentile-edge binner, DTW variants)
legitimately need a loose tolerance; those are listed in ``_USER_DEFERRED_LOOSE_TOLERANCE`` with the reason, so
the gate fires only where the module itself claims bit-identity or FP-reorder-only divergence.
"""

from __future__ import annotations

import ast
from pathlib import Path

import mlframe

from tests.test_meta._shared_ast_cache import parsed_ast

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent

_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "tests", "legacy", "profiling", "explore")

# An FP-reorder twin diverges by a few ULP x O(cells); 1e-6 is already generous for that class.
_MAX_TOLERANCE = 1e-6

# "relpath::funcname" -> why a genuinely-different-algorithm comparison needs a loose tolerance.
_PREEXISTING = (
    "pre-existing sweep comparing a GPU kernel against a CPU reference with a genuinely different reduction "
    "structure (not an FP-reorder twin). Tightening it would change which backend a host selects, so it needs "
    "its own numerical validation first; deferred rather than retightened blind."
)
_USER_DEFERRED_LOOSE_TOLERANCE: dict[str, str] = {
    "feature_selection/filters/_resident_candidate_mi_ktc.py::_run_rescand_sweep": (
        "rank binning vs percentile-edge binning are DIFFERENT binning schemes that disagree at ties; the "
        "module documents this as the approved FE-PAIR selection-equivalence trade, not an FP-reorder twin."
    ),
    "data_valuation/_propagate_gpu_ktc.py::_run_propagate_sweep": _PREEXISTING,
    "feature_selection/filters/_cat_confirm_permutation_tuning.py::_run_perm_kernel_sweep": _PREEXISTING,
    "feature_selection/filters/_unary_elementwise_tuning.py::_run_unary_sweep": _PREEXISTING,
    "feature_selection/filters/batch_pair_mi_gpu.py::_run_batch_pair_mi_sweep": _PREEXISTING,
    "feature_selection/filters/info_theory/_cmi_cuda_ktc.py::_run_cmi_sweep": _PREEXISTING,
}


def _tolerance_kwargs(call: ast.Call) -> dict[str, float]:
    """``{'equiv_rtol': v, 'equiv_atol': v}`` for the literal tolerance kwargs on a sweep call."""
    out: dict[str, float] = {}
    for kw in call.keywords:
        if kw.arg in ("equiv_rtol", "equiv_atol") and isinstance(kw.value, ast.Constant):
            value = kw.value.value
            if isinstance(value, (int, float)):
                out[kw.arg] = float(value)
    return out


def _enclosing_function(tree: ast.Module, target: ast.Call) -> str:
    """Name of the function lexically containing ``target``, or ``"<module>"``."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if child is target:
                    return node.name
    return "<module>"


def _build_offending_list() -> list[str]:
    """``["relpath::fname equiv_rtol=..", ...]`` for every sweep whose tolerance exceeds the cap."""
    out: list[str] = []
    for py in MLFRAME_DIR.rglob("*.py"):
        if any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS):
            continue
        tree = parsed_ast(py)
        if tree is None:
            continue
        rel = py.relative_to(MLFRAME_DIR).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fname = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if fname != "sweep_backend_grid":
                continue
            tols = _tolerance_kwargs(node)
            loose = {k: v for k, v in tols.items() if v > _MAX_TOLERANCE}
            if not loose:
                continue
            key = f"{rel}::{_enclosing_function(tree, node)}"
            if key in _USER_DEFERRED_LOOSE_TOLERANCE:
                continue
            out.append(f"{key} (line {node.lineno}) " + ", ".join(f"{k}={v:g}" for k, v in sorted(loose.items())))
    return out


def test_no_ktc_sweep_uses_a_loose_equivalence_tolerance():
    """No backend-equivalence sweep accepts a divergence far larger than its own documented numerics."""
    offenders = _build_offending_list()
    assert not offenders, (
        f"{len(offenders)} kernel-tuning sweep(s) use an equivalence tolerance looser than {_MAX_TOLERANCE:g}. "
        "A sweep decides which backend a host runs, so a slack tolerance can crown a genuinely divergent kernel. "
        "Fix: tighten to the real FP-reorder divergence (e.g. equiv_rtol=1e-9, equiv_atol=1e-12) and bump the "
        "module's cache SALT so stale regions re-tune; or, if the two variants are genuinely different "
        "algorithms, add the entry to _USER_DEFERRED_LOOSE_TOLERANCE with the reason.\n  " + "\n  ".join(sorted(offenders))
    )
