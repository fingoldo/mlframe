"""Probability calibration quality and post-hoc calibrators.

The ``post`` submodule pulls heavy optional deps (netcal, pycalib,
ml_insights, betacal, venn-abers); it is NOT eager-imported. Use:

    from mlframe.calibration.post import ...

Submodules:
    quality        - calibration quality assessment (reliability diagrams, ECE, ...).
    post           - post-hoc calibration methods (isotonic, Platt, beta, Venn-Abers, ...).
    probabilities  - probability transformations and diagnostics.

iter631: ``quality`` is NOT eager-imported. quality.py cascades through
matplotlib + properscoring + sklearn (~2s per process); the suite reaches
calibration symbols via ``mlframe.calibration.policy`` (the honest-diagnostics
ECE wrapper) which loads its own narrow deps. PEP 562 ``__getattr__`` lazy-
resolves both ``quality`` and ``probabilities`` symbols on first attribute
access so external ``from mlframe.calibration import X`` callers keep
working at the cost of a one-time deferred import.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # ``importlib.import_module`` bypasses ``_handle_fromlist``'s package-
    # attribute check which would otherwise re-enter __getattr__ and recurse.
    # Resolution order: quality (broader surface) first, then probabilities.
    import importlib
    _q = importlib.import_module("mlframe.calibration.quality")
    if hasattr(_q, name):
        val = getattr(_q, name)
        globals()[name] = val
        return val
    _p = importlib.import_module("mlframe.calibration.probabilities")
    if hasattr(_p, name):
        val = getattr(_p, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    import importlib
    _q = importlib.import_module("mlframe.calibration.quality")
    _p = importlib.import_module("mlframe.calibration.probabilities")
    return sorted(set(globals().keys()) | {n for n in dir(_q) if not n.startswith("_")} | {n for n in dir(_p) if not n.startswith("_")})
