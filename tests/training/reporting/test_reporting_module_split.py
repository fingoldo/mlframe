"""Wave 97 (2026-05-21): split training/_reporting.py (1223 lines)
into _reporting.py (now 709 lines) + new _reporting_probabilistic.py
(577 lines).

The ~520-line ``report_probabilistic_model_perf`` function moved to
the sibling file; the original re-exports it so existing
``from mlframe.training.reporting._reporting import report_probabilistic_model_perf``
imports continue to work.

The sibling lazy-imports ``_canonical_multilabel_y`` and ``_maybe_display``
from the parent module's partially-loaded state (the parent's bottom
re-export triggers our load AFTER those helpers are defined at the
parent's module top, so the partial-module lookup succeeds).
"""

from __future__ import annotations

from pathlib import Path


def test_report_probabilistic_model_perf_still_importable_from_facade() -> None:
    """Report probabilistic model perf still importable from facade."""
    from mlframe.training.reporting._reporting import report_probabilistic_model_perf

    assert callable(report_probabilistic_model_perf)


def test_other_reporting_symbols_still_importable() -> None:
    """Other reporting symbols still importable."""
    from mlframe.training.reporting._reporting import (
        report_model_perf,
        report_regression_model_perf,
        _maybe_display,
        _canonical_multilabel_y,
    )

    assert callable(report_model_perf)
    assert callable(report_regression_model_perf)
    assert callable(_maybe_display)
    assert callable(_canonical_multilabel_y)


def test_facade_below_1k_line_threshold() -> None:
    """Facade below 1k line threshold."""
    root = Path(__file__).resolve().parent.parent.parent.parent / "src" / "mlframe" / "training" / "reporting"
    facade = root / "_reporting.py"
    n = len(facade.read_text(encoding="utf-8").splitlines())
    assert n < 1000, f"_reporting.py is {n} lines, still over the 1k threshold"


def test_sibling_owns_the_moved_symbol() -> None:
    """Identity: the facade and the sibling expose the SAME function object."""
    from mlframe.training.reporting import _reporting, _reporting_probabilistic

    assert _reporting.report_probabilistic_model_perf is _reporting_probabilistic.report_probabilistic_model_perf


def test_sibling_resolves_parent_helpers_at_runtime() -> None:
    """Every probabilistic-report sibling that uses _canonical_multilabel_y or _maybe_display must resolve
    it to the parent's definition (not a local shadow).

    Later complexity splits carved report bodies further (``_maybe_display`` is now used from
    ``_reporting_probabilistic_helpers``), so the check follows the helpers to whichever sibling holds them
    and requires each helper to be held by at least one sibling, keeping the test's subject.
    """
    import importlib
    import pkgutil

    from mlframe.training import reporting as _pkg
    from mlframe.training.reporting import _reporting

    holders: dict[str, list[str]] = {"_canonical_multilabel_y": [], "_maybe_display": []}
    for info in pkgutil.iter_modules(_pkg.__path__):
        if not info.name.startswith("_reporting_probabilistic"):
            continue
        mod = importlib.import_module(f"{_pkg.__name__}.{info.name}")
        for name in holders:
            if name in vars(mod):
                assert getattr(mod, name) is getattr(_reporting, name), f"{info.name}.{name} shadows the parent's definition"
                holders[name].append(info.name)
    assert all(holders.values()), f"a parent helper is no longer used by any probabilistic-report sibling: {holders}"
