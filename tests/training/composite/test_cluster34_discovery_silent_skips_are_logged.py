"""Audit 2026-11-20 findings 34-F7 and 34-F8: two silent skips in composite discovery are now logged at WARNING.

34-F8: a failed residual-std probe was logged at DEBUG, so a broken probe disappeared from every default run.
34-F7: a spec with fewer than 50 valid rows passed the honest RMSE gate without a score and without any log line.
"""

import logging
from types import SimpleNamespace

import numpy as np

from mlframe.training.composite.discovery import _eval, _honest_rmse_gate


def test_residual_probe_failure_is_logged_at_warning(monkeypatch, caplog):
    """A probe that raises returns None and leaves a WARNING record that names the base and transform."""

    def boom(*_args, **_kwargs):
        """Stand in for a transform that cannot run."""
        raise RuntimeError("probe broke")

    monkeypatch.setattr(_eval, "call_transform", boom)
    with caplog.at_level(logging.DEBUG, logger=_eval.logger.name):
        y = np.arange(20, dtype=float)
        result = _eval._residual_probe_rejection(None, "base_col", "tr_name", object(), y, y, {}, None, 0.1, 1.0)
    assert result is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "residual-std probe failed" in r.getMessage()]
    assert warnings, "a failed probe must be visible at WARNING"
    assert "base_col" in warnings[0].getMessage()


def test_spec_with_too_few_valid_rows_is_logged_as_unscored(monkeypatch, caplog):
    """Under 50 valid rows the spec survives, and a WARNING says the RMSE gate did not run on it."""
    spec = SimpleNamespace(name="spec_a", transform_name="t", fitted_params={}, base_column="b")
    monkeypatch.setattr(_honest_rmse_gate, "get_transform", lambda name: object())
    monkeypatch.setattr(_honest_rmse_gate, "spec_base_columns", lambda s: [])
    monkeypatch.setattr(_honest_rmse_gate, "_base_arg", lambda df, cols, rows: np.zeros(len(rows)))
    monkeypatch.setattr(_honest_rmse_gate, "_spec_fit_mask", lambda *a, **k: np.zeros(10, dtype=bool))
    y = np.arange(10, dtype=float)
    ctx = (np.arange(10), np.arange(10), y, y, 1.0, None, None)
    with caplog.at_level(logging.WARNING, logger=_honest_rmse_gate.logger.name):
        survivors, rejected = _honest_rmse_gate._gate_specs(None, None, [spec], ctx, y, 1.0, 1.0, False)
    assert survivors == [spec]
    assert rejected == []
    assert any("spec_a" in r.getMessage() and "unscored" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
