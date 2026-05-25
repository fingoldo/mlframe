"""Sensor for AP14 provenance trail.

The training pipeline producers (pre_screen, MRMR, RFECV, preprocessing pipeline,
calibration, target-distribution-analyzer) stamp a small ``provenance`` record into
``metadata["provenance"][<step_name>]`` so a reviewer can verify every fit step touched
only the expected split (train / train_only / oof / etc.).

This sensor exercises the helper directly (unit) and verifies the analyzer wire-in
on a synthetic call (integration) without needing a full suite run.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_record_provenance_basic():
    from mlframe.training.provenance import record_provenance, get_provenance, VALID_SOURCES

    md: dict = {}
    record_provenance(md, "pre_screen", source="train_only", n_rows=100)
    record_provenance(md, "mrmr", source="train_only", n_rows=80, seed=42)
    record_provenance(md, "rfecv", source="train_only", n_rows=80, seed=42, extra={"cv_folds": 5})
    record_provenance(md, "preprocessing_pipeline", source="train", n_rows=100)
    record_provenance(md, "post_calibrate", source="oof", n_rows=200)
    record_provenance(md, "target_distribution_analyzer", source="train_only", n_rows=100)

    trail = get_provenance(md)
    assert len(trail) >= 6, f"Expected >= 6 records, got {len(trail)}"
    for step in ("pre_screen", "mrmr", "rfecv", "preprocessing_pipeline", "post_calibrate", "target_distribution_analyzer"):
        assert step in trail, f"Missing step {step!r}"
        rec = trail[step]
        assert rec["source"] in VALID_SOURCES, f"{step}: invalid source {rec['source']!r}"
        assert "ts" in rec and rec["ts"].endswith("+00:00"), f"{step}: missing or non-UTC ts"


def test_record_provenance_none_metadata_is_noop():
    from mlframe.training.provenance import record_provenance

    record_provenance(None, "x", source="train")


def test_record_provenance_unknown_source_still_records_with_warning(caplog):
    from mlframe.training.provenance import record_provenance, get_provenance

    md: dict = {}
    with caplog.at_level("WARNING", logger="mlframe.training.provenance"):
        record_provenance(md, "weird", source="not_a_valid_source")
    assert "weird" in get_provenance(md), "unknown source must still record"
    assert any("unknown source" in r.message for r in caplog.records)


def test_record_provenance_preserves_extra_fields():
    from mlframe.training.provenance import record_provenance, get_provenance

    md: dict = {}
    record_provenance(md, "rfecv", source="train_only", n_rows=100, seed=7, extra={"cv_folds": 5, "n_features_in": 25})
    rec = get_provenance(md)["rfecv"]
    assert rec["cv_folds"] == 5
    assert rec["n_features_in"] == 25
    assert rec["seed"] == 7


def test_format_provenance_table_renders():
    from mlframe.training.provenance import record_provenance, format_provenance_table

    md: dict = {}
    record_provenance(md, "pre_screen", source="train_only", n_rows=100)
    table = format_provenance_table(md)
    assert "pre_screen" in table
    assert "train_only" in table


def test_analyzer_records_provenance_via_target_distribution():
    """Smoke: TargetDistributionReport.knob_overrides_provenance is populated for known pathologies."""
    from mlframe.training._target_distribution_analyzer import analyze_target_distribution

    rng = np.random.default_rng(42)
    # Heavy-tailed regression target -> should produce loss_fn/huber/scale_pos_weight overrides
    y = rng.standard_t(df=2.5, size=500)
    rep = analyze_target_distribution(y, target_type="regression", has_time_axis=False)
    assert hasattr(rep, "knob_overrides_provenance"), "report missing knob_overrides_provenance"
    if rep.knob_overrides:
        # At least one stamped knob -- check format.
        flat = rep.knob_overrides_provenance
        assert isinstance(flat, dict)
        for slot, knobs in flat.items():
            for knob_name, stamp in knobs.items():
                assert stamp.get("source") == "analyzer"
                assert "reason" in stamp
                assert "value" in stamp


def test_analyzer_provenance_imbalanced_classes():
    """Class imbalance triggers scale_pos_weight + class_weight + auto_class_weights stamping."""
    from mlframe.training._target_distribution_analyzer import analyze_target_distribution

    # 99/1 split
    y = np.concatenate([np.zeros(990), np.ones(10)]).astype(int)
    rep = analyze_target_distribution(y, target_type="classification", has_time_axis=False)
    prov = rep.knob_overrides_provenance
    assert "lgb_kwargs" in prov, f"missing lgb_kwargs provenance stamp; got {list(prov.keys())}"
    assert prov["lgb_kwargs"]["class_weight"]["source"] == "analyzer"
    assert "class_imbalance" in prov["lgb_kwargs"]["class_weight"]["reason"]
    assert prov["xgb_kwargs"]["scale_pos_weight"]["source"] == "analyzer"
    assert prov["cb_kwargs"]["auto_class_weights"]["source"] == "analyzer"
