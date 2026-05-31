"""Model evaluation: performance reports across cv folds + holdout sets.

Submodules:
    reports   - full evaluation reporting (per-fold, summary tables, plots).
    bootstrap - bootstrap CIs + DeLong AUC test for honest-diagnostics.

iter631: ``reports`` is NOT eager-imported. Loading it cascades through
matplotlib + IPython + sklearn (~2.5-3s per process); the honest-diagnostics
suite only needs ``bootstrap`` for CI computation and never touches reports
during a training run. PEP 562 ``__getattr__`` lazy-loads reports symbols
on first attribute access so external ``from mlframe.evaluation import X``
callers keep working at the cost of a one-time deferred import.
"""

from __future__ import annotations


from mlframe.evaluation.bootstrap import bootstrap_metric, delong_test  # noqa: F401


def __getattr__(name: str):
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # ``from . import reports`` would route through ``_handle_fromlist`` ->
    # ``hasattr(package, 'reports')`` which re-enters this __getattr__ and
    # recurses. ``importlib.import_module`` bypasses the package-attribute
    # check and resolves the submodule via sys.modules directly.
    import importlib
    _r = importlib.import_module("mlframe.evaluation.reports")
    try:
        val = getattr(_r, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    globals()[name] = val
    return val


def __dir__():
    import importlib
    _r = importlib.import_module("mlframe.evaluation.reports")
    return sorted(set(globals().keys()) | {n for n in dir(_r) if not n.startswith("_")})
