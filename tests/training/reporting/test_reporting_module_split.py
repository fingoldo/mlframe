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
    """The sibling's top-level imports of _canonical_multilabel_y and
    _maybe_display must resolve to the parent's definitions (not local
    shadows)."""
    from mlframe.training.reporting import _reporting, _reporting_probabilistic

    assert _reporting_probabilistic._canonical_multilabel_y is _reporting._canonical_multilabel_y
    assert _reporting_probabilistic._maybe_display is _reporting._maybe_display
