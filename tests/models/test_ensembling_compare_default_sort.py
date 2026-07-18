"""Regression for compare_ensembles default sort metric.

Default lineage: test.* -> val.* -> oof.*. The latest move (Fix 3) flipped the default from val to OOF because val
is already burned for early-stopping; picking an ensemble flavour on val double-dips that surface. OOF (the
cross_val_predict held-out signal stamped by the trainer) is the only train-side metric never seen at fit and
never used for ES.

Both test.* and val.* overrides emit warnings: test.* via logger.warning, val.* via warnings.warn(UserWarning) so
the message shows up even when the logger is silenced.
"""

from __future__ import annotations

import logging
import types
import warnings as _warnings_mod

import pandas as pd
import pytest

from mlframe.models.ensembling import compare_ensembles


def _make_ens(oof_err: float, val_err: float, test_err: float):
    """Build a minimal object that quacks like the ensemble-perf record expected by compare_ensembles
    (only ``.metrics`` is accessed). Each surface gets a distinct value so a default-flip is observable."""
    return types.SimpleNamespace(
        metrics={
            "oof": {"1": {"integral_error": oof_err}},
            "val": {"1": {"integral_error": val_err}},
            "test": {"1": {"integral_error": test_err}},
        }
    )


def test_compare_ensembles_defaults_to_oof_sort_not_val_or_test():
    # Three conflicting orderings: A best on OOF, B best on VAL, C best on TEST.
    """Compare ensembles defaults to oof sort not val or test."""
    ensembles = {
        "A": _make_ens(oof_err=0.10, val_err=0.30, test_err=0.30),
        "B": _make_ens(oof_err=0.20, val_err=0.10, test_err=0.20),
        "C": _make_ens(oof_err=0.30, val_err=0.20, test_err=0.10),
    }

    res = compare_ensembles(ensembles, show_plot=False)
    assert isinstance(res, pd.DataFrame)
    # OOF-sort: A (0.10) -> B (0.20) -> C (0.30).
    assert list(res.index) == ["A", "B", "C"], f"compare_ensembles default sort should be OOF-based (A,B,C); got {list(res.index)}"


def test_compare_ensembles_warns_when_caller_overrides_to_test(caplog):
    """Compare ensembles warns when caller overrides to test."""
    ensembles = {
        "A": _make_ens(oof_err=0.10, val_err=0.30, test_err=0.30),
        "C": _make_ens(oof_err=0.30, val_err=0.20, test_err=0.10),
    }
    with caplog.at_level(logging.WARNING, logger="mlframe.models.ensembling"):
        compare_ensembles(
            ensembles,
            sort_metric="test.1.integral_error",
            show_plot=False,
        )
    assert any(
        "test-set selection bias" in rec.message for rec in caplog.records
    ), f"Explicit test.* sort_metric must emit a WARN; got records: {[r.message for r in caplog.records]}"


def test_compare_ensembles_warns_on_val_override():
    """val.* override now triggers a UserWarning (val is already burned for early stopping)."""
    ensembles = {"A": _make_ens(oof_err=0.10, val_err=0.30, test_err=0.30)}
    with pytest.warns(UserWarning, match="VAL split"):
        compare_ensembles(ensembles, sort_metric="val.1.integral_error", show_plot=False)


def test_compare_ensembles_no_warn_on_default():
    """Compare ensembles no warn on default."""
    ensembles = {"A": _make_ens(oof_err=0.10, val_err=0.30, test_err=0.30)}
    import logging as _logging

    handler_records = []

    class _Capture(_logging.Handler):
        """Groups tests for: Capture."""
        def emit(self, record):  # pragma: no cover - trivial
            """Emit."""
            handler_records.append(record)

    h = _Capture(level=_logging.WARNING)
    lg = _logging.getLogger("mlframe.models.ensembling")
    lg.addHandler(h)
    try:
        # Also confirm no UserWarning fires under the default.
        with _warnings_mod.catch_warnings():
            _warnings_mod.simplefilter("error", UserWarning)
            compare_ensembles(ensembles, show_plot=False)
    finally:
        lg.removeHandler(h)
    # No bias warning when the default is used.
    assert not any("test-set selection bias" in r.getMessage() for r in handler_records)
