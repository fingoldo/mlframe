"""Regression sensor: the ensemble chooser must rank CLASSIFICATION flavours by the metric keys the
report path actually stamps (``ice`` / ``roc_auc`` / ``pr_auc`` / ``brier_loss``), not the
regression-only ``integral_error`` / ``rmse`` keys that never appear on a classification run.

Pre-fix ``_ENSEMBLE_RANK_METRIC_CANDIDATES`` probed only ``integral_error`` / ``rmse``. On a
classification task ``report_perf`` stamps none of those, so every probe scored None and the chooser
fell through to the deterministic first-emitted ('arithm') flavour -- silently selecting a
suboptimal canonical ensemble on every classification suite. These tests pin the metric-driven pick
on the real classification metric layout (binary metrics nest under class 1).
"""

from __future__ import annotations

import logging

import pytest

from mlframe.training.core._ensemble_chooser import (
    _choose_ensemble_flavour,
    _read_ensemble_metric,
)


class _FakeEns:
    def __init__(self, metrics: dict):
        self.metrics = metrics


def _binary_metrics(split: str, *, ice: float, roc_auc: float, pr_auc: float, brier: float) -> dict:
    """Mirror the real binary-classification layout: per-class block nested under class id 1."""
    return {
        split: {
            1: {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "brier_loss": brier,
                "ice": ice,
                "log_loss": brier * 2.0,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
            }
        }
    }


def test_chooser_picks_best_ice_flavour_not_arithm_fallback(caplog):
    """The flavour with the lowest ``oof.ice`` wins; the no-metric fallback WARN must NOT fire."""
    ensembles = {
        "arithm": _FakeEns(_binary_metrics("oof", ice=0.30, roc_auc=0.70, pr_auc=0.60, brier=0.22)),
        "harm": _FakeEns(_binary_metrics("oof", ice=0.05, roc_auc=0.92, pr_auc=0.88, brier=0.09)),
        "geo": _FakeEns(_binary_metrics("oof", ice=0.18, roc_auc=0.81, pr_auc=0.75, brier=0.15)),
    }
    with caplog.at_level(logging.WARNING, logger="mlframe.training.core._phase_train_one_target"):
        winner = _choose_ensemble_flavour(ensembles)
    assert winner == "harm"
    assert winner != "arithm"  # arithm is the first-emitted blind fallback; ice should beat it
    assert not any("no candidate exposed" in r.message for r in caplog.records)


def test_chooser_ranks_classification_by_roc_auc_when_calibration_absent():
    """With no ``ice`` / ``brier`` keys, the chooser falls to ``roc_auc`` (higher-is-better)."""

    def _auc_only(roc: float) -> dict:
        return {"oof": {1: {"roc_auc": roc, "pr_auc": roc - 0.05}}}

    ensembles = {
        "arithm": _FakeEns(_auc_only(0.70)),
        "harm": _FakeEns(_auc_only(0.93)),  # best AUC
        "geo": _FakeEns(_auc_only(0.82)),
    }
    assert _choose_ensemble_flavour(ensembles) == "harm"


def test_chooser_higher_is_better_direction_for_auc():
    """Direct direction check: the HIGHEST roc_auc must win, not the lowest."""
    ensembles = {
        "a": _FakeEns({"oof": {1: {"roc_auc": 0.60}}}),
        "b": _FakeEns({"oof": {1: {"roc_auc": 0.95}}}),
    }
    assert _choose_ensemble_flavour(ensembles) == "b"


def test_read_metric_case_insensitive_resolves_production_rmse_key():
    """Regression report stamps ``RMSE`` (uppercase); the lowercase ``rmse`` candidate must resolve it."""
    ens = _FakeEns({"oof": {"RMSE": 0.42, "MAE": 0.3, "R2": 0.8}})
    assert _read_ensemble_metric(ens, "oof", "rmse") == pytest.approx(0.42)
    assert _read_ensemble_metric(ens, "oof", "mae") == pytest.approx(0.3)


def test_chooser_ranks_regression_by_uppercase_rmse():
    """End-to-end on the real regression key casing: lowest ``RMSE`` wins via the case-insensitive lookup."""
    ensembles = {
        "arithm": _FakeEns({"oof": {"RMSE": 0.90, "MAE": 0.7, "R2": 0.1}}),
        "harm": _FakeEns({"oof": {"RMSE": 0.10, "MAE": 0.08, "R2": 0.95}}),  # best
        "geo": _FakeEns({"oof": {"RMSE": 0.50, "MAE": 0.4, "R2": 0.5}}),
    }
    assert _choose_ensemble_flavour(ensembles) == "harm"


def test_chooser_classification_roc_auc_beats_brier_when_they_disagree():
    """AUC-first contract: the higher-``roc_auc`` flavour wins even when another flavour has a much
    better (lower) ``brier_loss``. Calibration keys are now tie-breakers AFTER discrimination, per
    bench_ensemble_chooser_rank_metric (AUC-first wins honest held-out test AUC 21/21 cells)."""
    ensembles = {
        "arithm": _FakeEns({"oof": {1: {"brier_loss": 0.25, "roc_auc": 0.92}}}),  # worse brier, best AUC
        "harm": _FakeEns({"oof": {1: {"brier_loss": 0.05, "roc_auc": 0.80}}}),  # best brier, worse AUC
    }
    assert _choose_ensemble_flavour(ensembles) == "arithm"


def test_chooser_classification_brier_breaks_genuine_auc_tie():
    """When ``roc_auc`` AND ``pr_auc`` are absent, ``brier_loss`` (lower-is-better) decides."""
    ensembles = {
        "arithm": _FakeEns({"oof": {1: {"brier_loss": 0.25}}}),
        "harm": _FakeEns({"oof": {1: {"brier_loss": 0.05}}}),  # best brier, no AUC present
    }
    assert _choose_ensemble_flavour(ensembles) == "harm"


def test_chooser_still_honours_legacy_integral_error_key():
    """Back-compat: callers / fixtures stamping the legacy ``integral_error`` key still rank correctly."""
    ensembles = {
        "arithm": _FakeEns({"oof": {"integral_error": 0.30}}),
        "harm": _FakeEns({"oof": {"integral_error": 0.05}}),
    }
    assert _choose_ensemble_flavour(ensembles) == "harm"
