"""Regression sensor for A3#2 (S41): `_ENSEMBLE_RANK_METRIC_CANDIDATES` test.* fallback.

The module-top comment in ``_ensemble_chooser.py`` promises a "one-time WARN at first
use" of the test.* fallback path because using the honest test split for model
selection biases every subsequent test-set metric optimistic. Prior to W11D the code
did NOT emit that WARN; the contradiction was the bug. This test pins the WARN in
place: a candidate dict where ONLY test.* metrics resolve must trip the WARN AND
return the deterministic winner.
"""

from __future__ import annotations

import logging

import pytest

from mlframe.training.core._ensemble_chooser import _choose_ensemble_flavour


class _FakeEns:
    def __init__(self, metrics: dict):
        self.metrics = metrics


def test_test_only_fallback_emits_warn(caplog):
    # Only test.* surfaces metrics; oof and val are empty on every candidate.
    ens_a = _FakeEns({"test": {"integral_error": 0.30}})
    ens_b = _FakeEns({"test": {"integral_error": 0.20}})
    with caplog.at_level(logging.WARNING, logger="mlframe.training.core._ensemble_chooser"):
        winner = _choose_ensemble_flavour({"a": ens_a, "b": ens_b})
    # Deterministic-lower winner is "b" (0.20 < 0.30).
    assert winner == "b"
    # The WARN message must mention test selection bias.
    bias_msgs = [r.getMessage() for r in caplog.records if "test" in r.getMessage().lower() and "selection" in r.getMessage().lower()]
    assert bias_msgs, f"expected test-selection WARN; got {[r.getMessage() for r in caplog.records]}"


def test_oof_winner_does_not_emit_test_warn(caplog):
    ens_a = _FakeEns({"oof": {"rmse": 0.30}, "test": {"rmse": 0.10}})
    ens_b = _FakeEns({"oof": {"rmse": 0.20}, "test": {"rmse": 0.99}})
    with caplog.at_level(logging.WARNING, logger="mlframe.training.core._ensemble_chooser"):
        winner = _choose_ensemble_flavour({"a": ens_a, "b": ens_b})
    # OOF path wins for ens_b (0.20 < 0.30); test.* metrics are never read.
    assert winner == "b"
    bias_msgs = [r.getMessage() for r in caplog.records if "test" in r.getMessage().lower() and "selection" in r.getMessage().lower()]
    assert not bias_msgs, f"OOF-resolved winner must not emit test-selection WARN; got {bias_msgs}"
