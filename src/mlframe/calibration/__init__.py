"""Probability calibration quality and post-hoc calibrators.

The ``post`` submodule pulls heavy optional deps (netcal, pycalib,
ml_insights, betacal, venn-abers); it is NOT eager-imported. Use:

    from mlframe.calibration.post import ...

Submodules:
    quality        - calibration quality assessment (reliability diagrams, ECE, ...).
    post           - post-hoc calibration methods (isotonic, Platt, beta, Venn-Abers, ...).
    probabilities  - probability transformations and diagnostics.

``quality`` is also NOT eager-imported. quality.py cascades through
matplotlib + properscoring + sklearn (~2s per process); the suite reaches
calibration symbols via ``mlframe.calibration.policy`` (the honest-diagnostics
ECE wrapper) which loads its own narrow deps. PEP 562 ``__getattr__`` lazy-
resolves both ``quality`` and ``probabilities`` symbols on first attribute
access so external ``from mlframe.calibration import X`` callers keep
working at the cost of a one-time deferred import.
"""

from __future__ import annotations

from typing import Any

from mlframe.calibration.ensembling import odds_ratio_combine
from mlframe.calibration.isotonic_risk import isotonic_overfit_risk
from mlframe.calibration.confidence_shrinkage import compute_oof_confidence, apply_confidence_shrinkage
from mlframe.calibration.threshold_optimizer import optimize_decision_threshold, apply_decision_threshold
from mlframe.calibration.group_bias_correction import fit_group_bias_correction, apply_group_bias_correction
from mlframe.calibration.smoothed_override import apply_smoothed_override
from mlframe.calibration.smoothed_override_backtest import backtest_override, OverrideBacktestResult, ConfidenceBucket  # public facade re-export (F401 already blanket-ignored for __init__.py)
from mlframe.calibration.asymmetric_rescale import fit_asymmetric_rescale, apply_asymmetric_rescale, cross_validate_asymmetric_rescale
from mlframe.calibration.group_zero_sum_constraint import apply_group_zero_sum_constraint, apply_group_zero_sum_constraint_multi
from mlframe.calibration.sticky_state_persistence_floor import apply_sticky_state_persistence_floor, optimize_persistence_floor
from mlframe.calibration.prediction_band_correction import find_prediction_band_shift, apply_prediction_band_correction

# Only the eagerly-bound names above; ``quality``/``probabilities`` symbols are resolved lazily via
# __getattr__ (see the module docstring) and are intentionally NOT enumerated here -- listing them would
# require importing those heavy modules eagerly, defeating the whole point of the lazy resolution.
__all__ = [
    "ConfidenceBucket",
    "OverrideBacktestResult",
    "apply_asymmetric_rescale",
    "apply_confidence_shrinkage",
    "apply_decision_threshold",
    "apply_group_bias_correction",
    "apply_group_zero_sum_constraint",
    "apply_group_zero_sum_constraint_multi",
    "apply_prediction_band_correction",
    "apply_smoothed_override",
    "apply_sticky_state_persistence_floor",
    "backtest_override",
    "compute_oof_confidence",
    "cross_validate_asymmetric_rescale",
    "find_prediction_band_shift",
    "fit_asymmetric_rescale",
    "fit_group_bias_correction",
    "isotonic_overfit_risk",
    "odds_ratio_combine",
    "optimize_decision_threshold",
    "optimize_persistence_floor",
]


_LAZY_SUBMODULES = ("mlframe.calibration.quality", "mlframe.calibration.probabilities")
_LAZY_OWNERS: dict[str, str] | None = None


def _lazy_owners() -> dict[str, str]:
    """Public top-level names of the lazy submodules mapped to the module defining them, read from their SOURCE (no import).

    Lets ``__getattr__`` answer "does a name exist" without paying the ~2 s ``quality`` import for every unknown attribute, such as a
    ``hasattr`` probe from tooling (22-F30). A name found in both modules resolves to ``quality``, the broader surface, as before.
    """
    global _LAZY_OWNERS
    if _LAZY_OWNERS is None:
        import ast
        import importlib.util

        owners: dict[str, str] = {}
        for modname in reversed(_LAZY_SUBMODULES):  # later modules first, so earlier (quality) wins on a shared name
            spec = importlib.util.find_spec(modname)
            if spec is None or not spec.origin:
                continue
            with open(spec.origin, "rb") as fh:
                tree = ast.parse(fh.read())
            for node in tree.body:
                names: list[str] = []
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names = [node.name]
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [(a.asname or a.name).split(".")[0] for a in node.names]
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    names = [tg.id for tg in targets if isinstance(tg, ast.Name)]
                for nm in names:
                    if not nm.startswith("_") and nm != "*":
                        owners[nm] = modname
        _LAZY_OWNERS = owners
    return _LAZY_OWNERS


def __getattr__(name: str) -> Any:
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    owner = _lazy_owners().get(name)
    if owner is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # ``importlib.import_module`` bypasses ``_handle_fromlist``'s package-
    # attribute check which would otherwise re-enter __getattr__ and recurse.
    import importlib

    val = getattr(importlib.import_module(owner), name)
    globals()[name] = val
    return val


def __dir__() -> list[str]:
    import importlib
    _q = importlib.import_module("mlframe.calibration.quality")
    _p = importlib.import_module("mlframe.calibration.probabilities")
    return sorted(set(globals().keys()) | {n for n in dir(_q) if not n.startswith("_")} | {n for n in dir(_p) if not n.startswith("_")})
