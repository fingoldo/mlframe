"""Name, in one line, every enabled finalize step that needed a calibration slice and found none.

Why this exists
---------------
Several suite-end steps fit on the disjoint calib slice that ``TrainingSplitConfig.calib_size > 0`` carves. Most of
them are ON by default. Without a calib slice each one walks the model entries, finds no ``calib_probs`` /
``calib_preds`` on any of them, ``continue``s past every entry, and returns having logged nothing -- because each
reports only when it produced something. The "enabled, reached, unable to act" state was invisible.

A production run sat in exactly that state (``'calib': 0``) and the log said nothing about any of it, while:

* the only well-separated binary model (test ROC AUC 0.81) kept the default 0.5 decision threshold at recall
  0.39 / 0.48, under a prior shift the same run had flagged as ``Δ=+6.6pp``;
* no post-hoc probability calibration was fitted, while its calibration error rose 1.48% -> 3.74% from val to
  test (a calib slice carved from train would not necessarily have closed a TEST prior shift either -- the point
  is that nothing said the step had not run);
* no conformal prediction sets were produced for it.

Reporting from inside each step would print the same fact once per step. One aggregated line says it once, names
every step it cost, and names the single knob that restores all of them.

What it deliberately does NOT report
------------------------------------
* Steps that are opt-in and off (regression point-recalibration defaults to ``point="off"``) -- their silence is
  what the caller asked for.
* Regression conformal intervals for models that carry OOF predictions: that path falls back to OOF residuals
  and runs without a calib slice, so it was not lost.
* Anything at all when a calib slice exists: the steps then run and report their own results.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._training_context import TrainingContext

logger = logging.getLogger(__name__)

__all__ = [
    "calib_dependent_steps_skipped",
    "report_calib_dependent_steps_skipped",
    "oof_dependent_steps_inert",
    "report_oof_dependent_steps_inert",
]

_BINARY = "binary_classification"
_CLASSIFICATION_TYPES = frozenset({"binary_classification", "multiclass_classification", "multilabel_classification"})


def _entries(ctx: "TrainingContext"):
    """Yield ``(target_type, entry)`` for every model entry, unwrapping the ``(entry, ...)`` tuple form."""
    for ttype, by_name in (getattr(ctx, "models", None) or {}).items():
        if not isinstance(by_name, dict):
            continue
        for entries in by_name.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                yield str(ttype), (entry[0] if isinstance(entry, tuple) and entry else entry)


def _has_calib_slice(ctx: "TrainingContext") -> bool:
    """True when a calib slice was carved or any entry already carries calib predictions."""
    calib_idx = getattr(ctx, "calib_idx", None)
    if calib_idx is not None and len(calib_idx) > 0:
        return True
    return any(
        getattr(e, "calib_probs", None) is not None or getattr(e, "calib_preds", None) is not None
        for _t, e in _entries(ctx)
    )


def _behavior_step_enabled(cfg: Any, name: str) -> bool:
    """Mirror the exact gate the behaviour-config steps use: ``_cfg is None or not getattr(_cfg, name, False)`` -> off.

    Mirroring matters more than it looks. With no behaviour config those steps return before touching the calib
    slice at all, so reporting them as "skipped for lack of a calib slice" would be a false attribution -- the
    exact kind of claim-the-code-does-not-keep this module exists to remove.
    """
    return cfg is not None and bool(getattr(cfg, name, False))


def _conformal_enabled(cfg: Any) -> bool:
    """Mirror ``_conformal_on_calib_slice``'s gate: only an explicit ``enabled=False`` turns it off; no config means on."""
    return cfg is None or bool(getattr(cfg, "enabled", True))


def calib_dependent_steps_skipped(ctx: "TrainingContext") -> list[str]:
    """Human-readable names of the ENABLED steps that could not run for lack of a calib slice; empty when none."""
    if _has_calib_slice(ctx):
        return []
    types = [t for t, _e in _entries(ctx)]
    has_classifier = any(t in _CLASSIFICATION_TYPES for t in types)
    has_binary = _BINARY in types
    # Regression conformal falls back to OOF residuals, so only entries lacking them actually lost their intervals.
    regression_without_oof = any(
        t == "regression" and getattr(e, "oof_preds", None) is None for t, e in _entries(ctx)
    )

    behavior = getattr(ctx, "behavior_config", None)
    conformal = getattr(ctx, "conformal_config", None)
    conformal_on = _conformal_enabled(conformal)
    sets_on = conformal_on and str(getattr(conformal, "classification_mode", "sets_lac") if conformal is not None else "sets_lac") != "off"

    skipped: list[str] = []
    if has_classifier:
        skipped.append("post-hoc probability calibration")
        if _behavior_step_enabled(behavior, "check_isotonic_overfit_risk"):
            skipped.append("isotonic overfit-risk check")
    if has_binary and _behavior_step_enabled(behavior, "auto_optimize_threshold"):
        skipped.append("decision-threshold optimisation (binary models keep the default 0.5 cut)")
    if has_classifier and sets_on:
        skipped.append("conformal prediction sets")
    if regression_without_oof and conformal_on:
        skipped.append("conformal regression intervals (models without OOF predictions)")
    return skipped


def report_calib_dependent_steps_skipped(ctx: "TrainingContext") -> list[str]:
    """Emit ONE warning naming every enabled calib-slice step that could not run, and return their names."""
    skipped = calib_dependent_steps_skipped(ctx)
    if skipped:
        logger.warning(
            "[calib] no calibration slice was carved (TrainingSplitConfig.calib_size is unset or 0), so %d "
            "enabled suite-end step(s) had nothing to fit on and did not run: %s. Each is ON by default and "
            "fits only on the disjoint calib slice; set calib_size > 0 to restore all of them. This matters most "
            "when a prior shift has been reported, since that is when an uncalibrated model and a default 0.5 cut "
            "are furthest from what you want.",
            len(skipped), "; ".join(skipped),
        )
    return skipped


def _oof_n_splits(ctx: "TrainingContext") -> int:
    """The effective ``oof_n_splits`` -- the knob that decides whether any model carries OOF predictions."""
    behavior = getattr(ctx, "behavior_config", None)
    try:
        return int(getattr(behavior, "oof_n_splits", 0) or 0) if behavior is not None else 0
    except (TypeError, ValueError):
        return 0


def oof_dependent_steps_inert(ctx: "TrainingContext") -> list[str]:
    """Enabled steps that need OOF predictions but cannot get them because ``oof_n_splits < 2``; empty otherwise.

    Unlike the calib case this is a documented default rather than an accident: ``oof_n_splits`` stays 0 because
    OOF costs a K-fold refit of every model, and ``_model_configs_behavior`` records that the dependent steps no-op
    "silently (not a WARN)" since it is a caller choice. The gap that leaves is that the caller made no choice --
    the defaults switch these steps ON and simultaneously make them impossible, so ``apply_confidence_shrinkage:
    bool = True`` reads as a promise the default run can never keep. Reported, therefore, but at INFO: it is a
    configuration fact, not a failure.
    """
    if _oof_n_splits(ctx) >= 2:
        return []
    # Any entry already carrying OOF (a caller-supplied model, a reloaded artefact) means the steps can run.
    if any(getattr(e, "oof_preds", None) is not None for _t, e in _entries(ctx)):
        return []
    types = {t for t, _e in _entries(ctx)}
    inert: list[str] = []
    reg_cal = getattr(ctx, "regression_calibration_config", None)
    if "regression" in types and reg_cal is not None and bool(getattr(reg_cal, "apply_confidence_shrinkage", False)):
        inert.append("confidence shrinkage of weakly-discriminative regression targets")
    behavior = getattr(ctx, "behavior_config", None)
    if (
        types & {"regression", "binary_classification"}
        and behavior is not None
        and bool(getattr(behavior, "recommend_diversity_additions_in_leaderboard", False))
    ):
        inert.append("OOF diversity recommendations for the ensemble leaderboard")
    return inert


def report_oof_dependent_steps_inert(ctx: "TrainingContext") -> list[str]:
    """Emit ONE INFO line naming the enabled OOF-dependent steps that the default ``oof_n_splits=0`` makes inert."""
    inert = oof_dependent_steps_inert(ctx)
    if inert:
        logger.info(
            "[oof] %d enabled step(s) need out-of-fold predictions and did not run, because oof_n_splits=%d "
            "(the default; OOF costs a K-fold refit of every model, so it is off unless asked for): %s. Set "
            "TrainingBehaviorConfig.oof_n_splits >= 2 to enable them.",
            len(inert), _oof_n_splits(ctx), "; ".join(inert),
        )
    return inert
