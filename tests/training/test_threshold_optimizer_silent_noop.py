"""Regression tests for the training-log audit 2026-09-20 follow-up, ``SEN-10`` / ``SEN-11`` / ``SEN-12``.

Several suite-end steps fit only on the disjoint calib slice (``TrainingSplitConfig.calib_size > 0``), and most are
ON by default. Without a slice each one finds no ``calib_probs`` / ``calib_preds``, ``continue``s past every entry,
and returns having logged nothing, because each reports only when it produced something.

The production run this comes from allocated ``'calib': 0``. Its only well-separated model (test ROC AUC 0.81) kept
the default 0.5 threshold at recall 0.39 / 0.48, under a prior shift the same run had flagged as ``Δ=+6.6pp``, and
nothing in the log said that threshold optimisation, probability calibration and conformal sets had all not run.

These tests pin ``report_calib_dependent_steps_skipped``: one line, naming every step the missing slice cost, and
never naming a step whose silence has a different, legitimate cause.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np

from mlframe.training.core._phase_finalize_calib_skipped import (
    calib_dependent_steps_skipped,
    oof_dependent_steps_inert,
    report_calib_dependent_steps_skipped,
    report_oof_dependent_steps_inert,
)
from mlframe.training.core._phase_finalize_calibration import _optimize_decision_threshold_on_calib_slice

N = 400


def _entry(*, calib: bool = False, oof: bool = False, seed: int = 0):
    """A model entry, optionally carrying the calib slice (and/or OOF preds) the steps need."""
    e = SimpleNamespace(model_name="cb_recency", calib_probs=None, calib_preds=None, calib_target=None, oof_preds=None)
    rng = np.random.default_rng(seed)
    if calib:
        y = (rng.random(N) < 0.3).astype(np.float64)
        p1 = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.15, N), 0.01, 0.99)
        e.calib_probs = np.column_stack([1 - p1, p1])
        e.calib_target = y
    if oof:
        e.oof_preds = rng.normal(size=N)
    return e


def _behavior(**kw):
    base = dict(auto_optimize_threshold=True, check_isotonic_overfit_risk=True, threshold_optimizer_kwargs=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _ctx(models, *, behavior="default", conformal=None, calib_idx=None):
    return SimpleNamespace(
        models=models, metadata={}, verbose=1, calib_idx=calib_idx,
        behavior_config=_behavior() if behavior == "default" else behavior,
        conformal_config=conformal,
    )


# --------------------------------------------------------------------------------------------------
# The production state
# --------------------------------------------------------------------------------------------------


def test_production_state_names_every_step_the_missing_slice_cost(caplog):
    """``'calib': 0`` on a binary target: calibration, isotonic check, threshold and conformal sets all lost."""
    ctx = _ctx({"binary_classification": {"target_total_hired_above_1": [_entry()]}})
    with caplog.at_level(logging.WARNING):
        skipped = report_calib_dependent_steps_skipped(ctx)
    text = " ".join(skipped)
    for step in ("post-hoc probability calibration", "isotonic overfit-risk check", "decision-threshold optimisation", "conformal prediction sets"):
        assert step in text, (step, skipped)
    assert "calib_size" in caplog.text
    # ONE line, not one per step: the log's own problem was repetition, not only silence.
    assert sum("no calibration slice was carved" in r.getMessage() for r in caplog.records) == 1


def test_the_threshold_step_itself_no_longer_emits_its_own_copy(caplog):
    """The per-step warning was consolidated into the aggregated report; it must not print the fact twice."""
    ctx = _ctx({"binary_classification": {"t": [_entry()]}})
    with caplog.at_level(logging.WARNING):
        _optimize_decision_threshold_on_calib_slice(ctx)
    assert caplog.text == ""
    assert "decision_threshold" not in ctx.metadata


# --------------------------------------------------------------------------------------------------
# Silences with a legitimate, different cause must not be attributed to the missing slice
# --------------------------------------------------------------------------------------------------


def test_a_carved_calib_slice_means_nothing_to_report():
    ctx = _ctx({"binary_classification": {"t": [_entry()]}}, calib_idx=np.arange(50))
    assert calib_dependent_steps_skipped(ctx) == []


def test_calib_predictions_on_any_entry_mean_nothing_to_report():
    ctx = _ctx({"binary_classification": {"t": [_entry(calib=True)]}})
    assert calib_dependent_steps_skipped(ctx) == []


def test_no_behavior_config_is_not_blamed_on_the_calib_slice():
    """Those steps return before touching the slice when there is no config; blaming the slice would be false."""
    ctx = _ctx({"binary_classification": {"t": [_entry()]}}, behavior=None)
    skipped = " ".join(calib_dependent_steps_skipped(ctx))
    assert "decision-threshold" not in skipped
    assert "isotonic" not in skipped


def test_disabled_steps_are_not_reported():
    ctx = _ctx(
        {"binary_classification": {"t": [_entry()]}},
        behavior=_behavior(auto_optimize_threshold=False, check_isotonic_overfit_risk=False),
        conformal=SimpleNamespace(enabled=False, classification_mode="sets_lac"),
    )
    assert calib_dependent_steps_skipped(ctx) == ["post-hoc probability calibration"]


def test_conformal_sets_off_is_not_reported():
    ctx = _ctx({"binary_classification": {"t": [_entry()]}}, conformal=SimpleNamespace(enabled=True, classification_mode="off"))
    assert "conformal prediction sets" not in calib_dependent_steps_skipped(ctx)


def test_regression_with_oof_still_gets_conformal_intervals():
    """Regression conformal falls back to OOF residuals, so a model carrying them lost nothing."""
    ctx = _ctx({"regression": {"t": [_entry(oof=True)]}})
    assert calib_dependent_steps_skipped(ctx) == []


def test_regression_without_oof_loses_its_intervals():
    ctx = _ctx({"regression": {"t": [_entry(oof=False)]}})
    assert calib_dependent_steps_skipped(ctx) == ["conformal regression intervals (models without OOF predictions)"]


def test_threshold_optimisation_is_only_reported_for_binary_targets():
    ctx = _ctx({"multiclass_classification": {"t": [_entry()]}})
    assert not any("decision-threshold" in s for s in calib_dependent_steps_skipped(ctx))


def test_no_models_means_nothing_to_report(caplog):
    with caplog.at_level(logging.WARNING):
        assert report_calib_dependent_steps_skipped(_ctx({})) == []
    assert caplog.text == ""


def test_a_present_calib_slice_still_lets_the_threshold_step_do_its_job():
    ctx = _ctx({"binary_classification": {"t": [_entry(calib=True)]}})
    _optimize_decision_threshold_on_calib_slice(ctx)
    assert "decision_threshold" in ctx.metadata


# --------------------------------------------------------------------------------------------------
# SEN-12: OOF-dependent steps that the default oof_n_splits=0 makes permanently inert
# --------------------------------------------------------------------------------------------------

def _oof_ctx(models, *, oof_n_splits=0, shrink=True, diversity=True):
    return SimpleNamespace(
        models=models, metadata={}, verbose=1, calib_idx=None,
        behavior_config=SimpleNamespace(oof_n_splits=oof_n_splits, recommend_diversity_additions_in_leaderboard=diversity),
        regression_calibration_config=SimpleNamespace(apply_confidence_shrinkage=shrink),
        conformal_config=None,
    )


def test_default_config_reports_both_oof_steps_as_inert_at_info(caplog):
    """At defaults both steps are ON and neither can run -- the documented design, now at least visible."""
    ctx = _oof_ctx({"regression": {"target_total_charge": [_entry()]}})
    with caplog.at_level(logging.INFO):
        inert = report_oof_dependent_steps_inert(ctx)
    assert any("confidence shrinkage" in s for s in inert)
    assert any("diversity" in s for s in inert)
    recs = [r for r in caplog.records if "out-of-fold" in r.getMessage()]
    assert len(recs) == 1
    # INFO, not WARNING: the codebase deliberately treats this as a caller config choice, not a failure.
    assert recs[0].levelno == logging.INFO
    assert "oof_n_splits" in recs[0].getMessage()


def test_oof_enabled_means_nothing_is_inert():
    assert oof_dependent_steps_inert(_oof_ctx({"regression": {"t": [_entry()]}}, oof_n_splits=5)) == []


def test_an_entry_already_carrying_oof_means_nothing_is_inert():
    """A reloaded or caller-supplied model with OOF preds lets the steps run even at oof_n_splits=0."""
    assert oof_dependent_steps_inert(_oof_ctx({"regression": {"t": [_entry(oof=True)]}})) == []


def test_disabled_oof_steps_are_not_reported():
    ctx = _oof_ctx({"regression": {"t": [_entry()]}}, shrink=False, diversity=False)
    assert oof_dependent_steps_inert(ctx) == []


def test_shrinkage_is_regression_only():
    """Confidence shrinkage applies to regression targets; a classification-only run must not list it."""
    ctx = _oof_ctx({"binary_classification": {"t": [_entry()]}})
    inert = oof_dependent_steps_inert(ctx)
    assert not any("shrinkage" in s for s in inert)
    assert any("diversity" in s for s in inert)
