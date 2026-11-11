"""Regression test for the ``logger.isEnabledFor(INFO)`` gate that wraps the classification-report text
in ``report_probabilistic_model_perf``.

Profile of multilabel fuzz combo c0140 (iter12, 200k rows) attributed 2.94s to sklearn's ``classification_report`` via ``_param_validation`` (62 calls x ~45ms each). The text was
immediately handed to ``logger.info``, so when the logger's effective level filtered out INFO (the common verbose=0 path) the expensive computation was wasted.

iter105 (2026-05-20) added ``logger.isEnabledFor(logging.INFO)`` to the ``if print_report:`` gate. The text is built since then by the njit-backed
``_report_probabilist_identical_text_shape`` (the sklearn call it replaced is gone from the single-label path and the multilabel path builds no text), so this test pins the gate on
that helper: when no INFO handler will accept the message it must not run, and with INFO enabled it must (the control that keeps the silent run from proving nothing).
"""

from __future__ import annotations

import logging

import numpy as np


def test_classification_report_skipped_when_info_filtered(monkeypatch):
    """When ``logger.isEnabledFor(logging.INFO)`` is False (suite verbose=0 + no file handler) the report text must NOT be built; with INFO enabled it must be."""
    from mlframe.training.reporting import _reporting, _reporting_probabilistic

    # Silence the module logger so isEnabledFor(INFO) returns False.
    _reporting.logger.setLevel(logging.WARNING)

    # Tiny binary single-label inputs (8 samples): the body exits quickly, we only test the gate, not the math.
    targets = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int64)
    probs = np.stack([1.0 - np.linspace(0.1, 0.9, 8), np.linspace(0.1, 0.9, 8)], axis=1)

    calls = {"n": 0}
    real_text_builder = _reporting_probabilistic._report_probabilist_identical_text_shape

    def _counting_text_builder(*args, **kwargs):
        """Count the report-text builds, then run the real builder."""
        calls["n"] += 1
        return real_text_builder(*args, **kwargs)

    monkeypatch.setattr(_reporting_probabilistic, "_report_probabilist_identical_text_shape", _counting_text_builder)

    def _run():
        """Run the single-label report once with print_report on."""
        _reporting.report_probabilistic_model_perf(
            targets=targets,
            columns=["f1", "f2"],
            model_name="t",
            model=None,
            probs=probs,
            preds=(probs[:, 1] >= 0.5).astype(np.int64),
            classes=[0, 1],
            print_report=True,
            show_perf_chart=False,
            verbose=False,
        )

    try:
        _run()
        assert calls["n"] == 0, f"the report text was built {calls['n']} time(s) under a WARNING-level logger; the isEnabledFor(INFO) gate is regressed"

        _reporting.logger.setLevel(logging.INFO)
        _run()
        assert calls["n"] >= 1, "the report text is never built with INFO enabled, so the silent run above proves nothing"
    finally:
        _reporting.logger.setLevel(logging.NOTSET)
