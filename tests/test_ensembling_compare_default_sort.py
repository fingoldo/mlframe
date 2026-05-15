"""Regression for compare_ensembles default sort metric.

Pre-fix: ``sort_metric`` defaulted to ``"test.1.integral_error"``, which biased
ensemble selection toward the holdout. Default flipped to the val-set metric;
explicit ``test.*`` callers still get the legacy behaviour but a WARN log fires.
"""

from __future__ import annotations

import logging
import types

import pandas as pd

from mlframe.models.ensembling import compare_ensembles


def _make_ens(val_err: float, test_err: float):
    """Build a minimal object that quacks like the ensemble-perf record
    expected by compare_ensembles (only ``.metrics`` is accessed)."""
    return types.SimpleNamespace(
        metrics={
            "val": {"1": {"integral_error": val_err}},
            "test": {"1": {"integral_error": test_err}},
        }
    )


def test_compare_ensembles_defaults_to_val_sort_not_test():
    # Conflicting orderings: A is best on VAL, C is best on TEST.
    # If the default still sorted by test we'd see C first.
    ensembles = {
        "A": _make_ens(val_err=0.10, test_err=0.30),
        "B": _make_ens(val_err=0.20, test_err=0.20),
        "C": _make_ens(val_err=0.30, test_err=0.10),
    }

    res = compare_ensembles(ensembles, show_plot=False)
    assert isinstance(res, pd.DataFrame)
    # val-sort: A (0.10) -> B (0.20) -> C (0.30).
    assert list(res.index) == ["A", "B", "C"], (
        f"compare_ensembles default sort should be val-based (A,B,C); got {list(res.index)}"
    )


def test_compare_ensembles_warns_when_caller_overrides_to_test(caplog):
    ensembles = {
        "A": _make_ens(val_err=0.10, test_err=0.30),
        "C": _make_ens(val_err=0.30, test_err=0.10),
    }
    with caplog.at_level(logging.WARNING, logger="mlframe.models.ensembling"):
        compare_ensembles(
            ensembles,
            sort_metric="test.1.integral_error",
            show_plot=False,
        )
    assert any("test-set selection bias" in rec.message for rec in caplog.records), (
        f"Explicit test.* sort_metric must emit a WARN; got records: {[r.message for r in caplog.records]}"
    )


def test_compare_ensembles_no_warn_on_default():
    ensembles = {"A": _make_ens(val_err=0.10, test_err=0.30)}
    import logging as _logging

    handler_records = []

    class _Capture(_logging.Handler):
        def emit(self, record):  # pragma: no cover - trivial
            handler_records.append(record)

    h = _Capture(level=_logging.WARNING)
    lg = _logging.getLogger("mlframe.models.ensembling")
    lg.addHandler(h)
    try:
        compare_ensembles(ensembles, show_plot=False)
    finally:
        lg.removeHandler(h)
    # No bias warning when the default is used.
    assert not any("test-set selection bias" in r.getMessage() for r in handler_records)
