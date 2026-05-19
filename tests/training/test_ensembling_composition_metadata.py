"""Tests for Arch-3/4/6 ensembling metadata + calibration-mix warning + ranker gates.

Arch-3: ``metadata["ensembles_chosen"]`` is sub-keyed per family ("simple" / "cross_target").
Arch-4: ``_combine_probs`` emits a WARN on calibrated/uncalibrated mix and stamps
        ``metadata["ensembles_calibrated"]``.
Arch-6: ``metadata["ensemble_composition"]`` exposes per-target {flavour, members, fallback_reason}.

Behavioural, not source-inspecting; assertions hit the metadata dict directly.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Arch-3: ensembles_chosen sub-keyed by family
# ---------------------------------------------------------------------------
def test_resolve_chosen_flavour_reads_simple_bucket():
    from mlframe.training.core.predict import _resolve_chosen_flavour

    metadata = {
        "ensembles_chosen": {
            "simple": {
                "binary": {"y_target": "harm"},
            },
            "cross_target": {},
        }
    }
    assert _resolve_chosen_flavour(metadata, "binary", "y_target") == "harm"


def test_resolve_chosen_flavour_reads_cross_target_bucket_for_ct_prefix():
    from mlframe.training.core.predict import _resolve_chosen_flavour

    metadata = {
        "ensembles_chosen": {
            "simple": {"binary": {"y_target": "harm"}},
            "cross_target": {"binary": {"_CT_ENSEMBLE__y_target": "linear_stack"}},
        }
    }
    # CT-prefixed target name dispatches to cross_target bucket
    assert _resolve_chosen_flavour(metadata, "binary", "_CT_ENSEMBLE__y_target") == "linear_stack"
    # Non-CT target reads simple bucket
    assert _resolve_chosen_flavour(metadata, "binary", "y_target") == "harm"


def test_resolve_chosen_flavour_missing_returns_none():
    from mlframe.training.core.predict import _resolve_chosen_flavour

    metadata = {"ensembles_chosen": {"simple": {}, "cross_target": {}}}
    assert _resolve_chosen_flavour(metadata, "binary", "y_missing") is None


# ---------------------------------------------------------------------------
# Arch-4: calibrated/uncalibrated mix WARN + metadata stamp
# ---------------------------------------------------------------------------
def test_combine_probs_warns_on_calibration_mix(caplog):
    from mlframe.training.core.predict import _combine_probs

    probs = [np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4]), np.array([0.3, 0.4, 0.5])]
    metadata: dict = {}
    with caplog.at_level(logging.WARNING):
        out = _combine_probs(
            probs, "arithm",
            is_calibrated_per_model=[True, False, True],
            metadata=metadata,
            target_label="binary/y",
        )
    assert out.shape == (3,)
    # WARN log mentions "mixing calibrated"
    assert any("mixing calibrated" in r.message for r in caplog.records)
    # Metadata stamped False because not all members calibrated
    assert metadata["ensembles_calibrated"] is False


def test_combine_probs_no_warn_when_all_calibrated(caplog):
    from mlframe.training.core.predict import _combine_probs

    probs = [np.array([0.1, 0.2]), np.array([0.2, 0.3])]
    metadata: dict = {}
    with caplog.at_level(logging.WARNING):
        _combine_probs(
            probs, "arithm",
            is_calibrated_per_model=[True, True],
            metadata=metadata,
        )
    assert not any("mixing calibrated" in r.message for r in caplog.records)
    assert metadata["ensembles_calibrated"] is True


def test_combine_probs_no_warn_when_none_calibrated(caplog):
    from mlframe.training.core.predict import _combine_probs

    probs = [np.array([0.1, 0.2]), np.array([0.2, 0.3])]
    metadata: dict = {}
    with caplog.at_level(logging.WARNING):
        _combine_probs(
            probs, "arithm",
            is_calibrated_per_model=[False, False],
            metadata=metadata,
        )
    assert not any("mixing calibrated" in r.message for r in caplog.records)
    assert metadata["ensembles_calibrated"] is False


def test_combine_probs_proceeds_on_mix_does_not_refuse():
    """Arch-4 contract: WARN, NOT REFUSE. Mix must still produce an answer."""
    from mlframe.training.core.predict import _combine_probs

    probs = [np.full(4, 0.1), np.full(4, 0.9)]
    out = _combine_probs(
        probs, "arithm",
        is_calibrated_per_model=[True, False],
        metadata={},
    )
    np.testing.assert_allclose(out, [0.5, 0.5, 0.5, 0.5])


def test_is_post_hoc_calibrated_model_class_name_match():
    from mlframe.training.core.predict import _is_post_hoc_calibrated_model

    class _PostHocCalibratedModel:
        pass

    class _SomethingElse:
        pass

    assert _is_post_hoc_calibrated_model(_PostHocCalibratedModel()) is True
    assert _is_post_hoc_calibrated_model(_SomethingElse()) is False
    assert _is_post_hoc_calibrated_model(None) is False


# ---------------------------------------------------------------------------
# Arch-6: ensemble_composition stamping
# ---------------------------------------------------------------------------
def test_stamp_ensemble_composition_simple_bucket_records_members_and_flavour():
    from mlframe.training.core._phase_finalize import _stamp_ensemble_composition

    class _Ctx:
        ensembles = {
            "binary": {
                "y_target": {"arithm": object(), "harm": object(), "geo": object()},
            }
        }
        models = {}
        metadata = {
            "ensembles_chosen": {
                "simple": {"binary": {"y_target": "harm"}},
                "cross_target": {},
            }
        }
        verbose = 0

    ctx = _Ctx()
    _stamp_ensemble_composition(ctx)
    comp = ctx.metadata["ensemble_composition"]
    assert "y_target" in comp["simple"]["binary"]
    entry = comp["simple"]["binary"]["y_target"]
    assert entry["flavour"] == "harm"
    assert entry["fallback_reason"] is None
    # Three methods => weight 1/3 each, names sorted
    assert [n for n, _ in entry["members"]] == ["arithm", "geo", "harm"]
    np.testing.assert_allclose([w for _, w in entry["members"]], [1 / 3, 1 / 3, 1 / 3])


def test_stamp_ensemble_composition_marks_fallback_when_no_flavour_chosen():
    from mlframe.training.core._phase_finalize import _stamp_ensemble_composition

    class _Ctx:
        ensembles = {"binary": {"y_target": {"arithm": object()}}}
        models = {}
        metadata = {"ensembles_chosen": {"simple": {}, "cross_target": {}}}
        verbose = 0

    ctx = _Ctx()
    _stamp_ensemble_composition(ctx)
    entry = ctx.metadata["ensemble_composition"]["simple"]["binary"]["y_target"]
    assert entry["flavour"] is None
    assert entry["fallback_reason"] is not None
    assert "first-emitted" in entry["fallback_reason"]


def test_stamp_ensemble_composition_cross_target_reads_stacker_state():
    from mlframe.training.core._phase_finalize import _stamp_ensemble_composition

    class _FakeEnsemble:
        def export_metadata(self):
            return {
                "strategy": "linear_stack",
                "component_names": ["modelA", "modelB"],
                "weights": [0.6, 0.4],
                "notes": {},
            }

    class _Entry:
        model = _FakeEnsemble()

    class _Ctx:
        ensembles = {}
        models = {"binary": {"_CT_ENSEMBLE__y_target": [_Entry()]}}
        metadata = {}
        verbose = 0

    ctx = _Ctx()
    _stamp_ensemble_composition(ctx)
    entry = ctx.metadata["ensemble_composition"]["cross_target"]["binary"]["_CT_ENSEMBLE__y_target"]
    assert entry["strategy"] == "linear_stack"
    assert entry["members"] == [("modelA", 0.6), ("modelB", 0.4)]
    assert entry["fallback_reason"] is None


def test_stamp_ensemble_composition_cross_target_single_best_fallback():
    from mlframe.training.core._phase_finalize import _stamp_ensemble_composition

    class _FakeEnsemble:
        def export_metadata(self):
            return {
                "strategy": "single_best_fallback",
                "component_names": ["modelA"],
                "weights": [1.0],
                "notes": {},
            }

    class _Entry:
        model = _FakeEnsemble()

    class _Ctx:
        ensembles = {}
        models = {"binary": {"_CT_ENSEMBLE__y_target": [_Entry()]}}
        metadata = {}
        verbose = 0

    ctx = _Ctx()
    _stamp_ensemble_composition(ctx)
    entry = ctx.metadata["ensemble_composition"]["cross_target"]["binary"]["_CT_ENSEMBLE__y_target"]
    assert "fell back" in entry["fallback_reason"]
