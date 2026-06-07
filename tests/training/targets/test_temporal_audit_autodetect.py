"""Tests for the temporal_audit ts_field auto-detect path.

When the operator already configured ``ts_field`` on the FTE for
splitting / date-feature extraction, requiring them to ALSO state the
column on ``behavior_config.target_temporal_audit_column`` was
needless friction. The auto-detect introduced 2026-04-27 falls
through from ``behavior_config`` (operator override) to
``features_and_targets_extractor.ts_field`` (FTE-discovered).

Resolution order verified here:
  1. behavior_config.target_temporal_audit_column = "<col>"  → explicit opt-in to <col>
  2. behavior_config.target_temporal_audit_column = ""       → explicit opt-out (audit disabled)
  3. behavior_config.target_temporal_audit_column = None     → fall through to FTE.ts_field
  4. FTE.ts_field set + column present in df                 → auto-detect (audit fires)
  5. neither                                                  → audit silent
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Minimal FTE stand-ins
# ---------------------------------------------------------------------------


class _FTEWithTsField:
    """Mock FTE that exposes ts_field — mirrors prod FTE shape.

    Returns the timestamp column as the 5th tuple element ``timestamps``
    so the suite's audit auto-detect can use it as a fallback when
    ``ts_field`` is also in ``columns_to_drop`` (the prod pattern).
    """

    def __init__(self, target_col="y", ts_field="ts"):
        self.target_col = target_col
        self.ts_field = ts_field

    def transform(self, df):
        from mlframe.training.configs import TargetTypes
        target = df[self.target_col].to_numpy() if hasattr(df[self.target_col], "to_numpy") else df[self.target_col].values
        ts_values = None
        if self.ts_field is not None and self.ts_field in df.columns:
            col = df[self.ts_field]
            ts_values = col.to_numpy() if hasattr(col, "to_numpy") else col.values
        return (
            df,
            {TargetTypes.BINARY_CLASSIFICATION: {self.target_col: target}},
            None,        # group_ids_raw
            None,        # group_ids
            ts_values,   # timestamps
            None,        # artifacts
            [self.target_col],
            {},
        )


class _FTEWithoutTsField:
    """Mock FTE without ts_field — auto-detect should NOT fire."""

    def __init__(self, target_col="y"):
        self.target_col = target_col
        # No self.ts_field attribute.

    def transform(self, df):
        from mlframe.training.configs import TargetTypes
        target = df[self.target_col].to_numpy() if hasattr(df[self.target_col], "to_numpy") else df[self.target_col].values
        return (
            df,
            {TargetTypes.BINARY_CLASSIFICATION: {self.target_col: target}},
            None, None, None, None,
            [self.target_col],
            {},
        )


def _make_fixture(n=400, with_drift=True, seed=0):
    """Small frame with a 'ts' column spanning ~year so granularity
    auto-picker lands on month/quarter (>30 bins) — enough to compute
    audit reliably."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2024-01-01", periods=n, freq="d")
    if with_drift:
        # Half-year regime change in P(y=1)
        rates = np.where(np.arange(n) < n // 2, 0.20, 0.80)
        y = (rng.uniform(size=n) < rates).astype(np.int8)
    else:
        y = (rng.uniform(size=n) < 0.5).astype(np.int8)
    return pd.DataFrame({
        "ts": days,
        "x_num": rng.standard_normal(n),
        "y": y,
    })


# ---------------------------------------------------------------------------
# Direct API: audit_target_over_time on a synthetic frame
# ---------------------------------------------------------------------------


def test_audit_fires_with_explicit_ts_col():
    """Sanity-baseline: when caller passes the timestamp column
    explicitly, audit produces multi-segment output."""
    from mlframe.training.targets.target_temporal_audit import audit_target_over_time

    df = _make_fixture(n=400, with_drift=True)
    result = audit_target_over_time(
        df, timestamp_col="ts", target_col="y",
        target_type="binary_classification",
    )
    # 400 days at month granularity = 13 bins; with sparse-bin filter
    # we expect ~9-13 kept bins. Need at least 2 segments for the
    # drift to register.
    assert len(result.bins) >= 8
    # 50/50 split between the two regimes → segments captured
    assert len(result.segments) >= 2
    assert any("not stable over time" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Auto-detect path through train_mlframe_models_suite (resolution rules)
# ---------------------------------------------------------------------------
# We don't need to run the full suite — just exercise the resolution
# logic. The relevant block lives in core.py around the
# ``_audit_ts_override`` resolution.


def _resolve_audit_ts_col(behavior_config, fte, df):
    """Mirrors the resolution logic in core.py (just the audit-column
    pick). Lets us unit-test the rules without spinning up
    train_mlframe_models_suite.

    Keep this in sync with core.py: any future change there should be
    reflected here AND the test should fail loudly until the new
    contract is locked in.
    """
    _audit_ts_override = getattr(behavior_config, "target_temporal_audit_column", None) if behavior_config else None
    if _audit_ts_override is None:
        _fte_ts = getattr(fte, "ts_field", None)
        if _fte_ts and df is not None and hasattr(df, "columns") and _fte_ts in df.columns:
            return _fte_ts
        return None
    return _audit_ts_override


class _BC:
    """Tiny stand-in for TrainingBehaviorConfig holding only the audit knob."""
    def __init__(self, target_temporal_audit_column=None):
        self.target_temporal_audit_column = target_temporal_audit_column


def test_resolve_explicit_override_wins():
    """Explicit behavior_config column wins over FTE auto-detect."""
    df = _make_fixture(n=10)
    fte = _FTEWithTsField(ts_field="ts")
    bc = _BC(target_temporal_audit_column="some_other_col")
    assert _resolve_audit_ts_col(bc, fte, df) == "some_other_col"


def test_resolve_empty_string_disables_audit():
    """Empty string explicitly disables audit, even with FTE.ts_field set."""
    df = _make_fixture(n=10)
    fte = _FTEWithTsField(ts_field="ts")
    bc = _BC(target_temporal_audit_column="")
    # Empty string is the override → returned as-is. The truthy check
    # downstream makes this fall through.
    result = _resolve_audit_ts_col(bc, fte, df)
    assert result == ""
    assert not result  # falsy, audit silent


def test_resolve_none_falls_through_to_fte_ts_field():
    """When behavior_config doesn't override, FTE.ts_field is used."""
    df = _make_fixture(n=10)
    fte = _FTEWithTsField(ts_field="ts")
    bc = _BC(target_temporal_audit_column=None)
    assert _resolve_audit_ts_col(bc, fte, df) == "ts"


def test_resolve_no_fte_attribute():
    """FTE without ts_field attribute → audit silent (None returned)."""
    df = _make_fixture(n=10)
    fte = _FTEWithoutTsField()
    bc = _BC(target_temporal_audit_column=None)
    assert _resolve_audit_ts_col(bc, fte, df) is None


def test_resolve_fte_ts_field_set_but_column_missing_from_df():
    """FTE says ts_field='timestamp' but df has only 'ts' column →
    audit silent (column-presence check guards against typos)."""
    df = _make_fixture(n=10)  # has 'ts', not 'timestamp'
    fte = _FTEWithTsField(ts_field="timestamp")
    bc = _BC(target_temporal_audit_column=None)
    assert _resolve_audit_ts_col(bc, fte, df) is None


def test_resolve_no_behavior_config_no_fte():
    """Both None → audit silent."""
    df = _make_fixture(n=10)
    assert _resolve_audit_ts_col(None, _FTEWithoutTsField(), df) is None


# ---------------------------------------------------------------------------
# End-to-end via train_mlframe_models_suite — capture the auto-detect
# log line. Heavier test, runs the actual suite on a small frame.
#
# 2026-04-27: skipped pending an unrelated bug in splitting.py
# (``DatetimeIndex`` object has no attribute ``dt`` when the suite
# passes timestamps as a numpy datetime64 ndarray — separate fix).
# The fuzz suite combos with ``with_datetime_col=True`` already
# exercise the same auto-detect path through SimpleFeaturesAndTargetsExtractor;
# fix the splitting.py path and re-enable these.
# ---------------------------------------------------------------------------


def test_suite_auto_detects_ts_field_from_fte_and_logs(caplog, tmp_path):
    """Spin up a real (tiny) suite run with FTE.ts_field set and
    behavior_config.target_temporal_audit_column unset. The
    auto-detect INFO log line must fire AND the audit must populate
    metadata['target_temporal_audit']."""
    from mlframe.training import train_mlframe_models_suite, TrainingBehaviorConfig
    from mlframe.training.configs import (
        ModelHyperparamsConfig, PreprocessingBackendConfig, FeatureTypesConfig,
    )
    df = _make_fixture(n=400, with_drift=True)
    fte = _FTEWithTsField(target_col="y", ts_field="ts")

    with caplog.at_level(logging.INFO, logger="mlframe.training.core"):
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="autodetect_smoke",
            model_name="autodetect_smoke",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            hyperparams_config=ModelHyperparamsConfig(
                iterations=3,
                early_stopping_rounds=2,
                cb_kwargs={"task_type": "CPU", "verbose": 0},
            ),
            behavior_config=TrainingBehaviorConfig(
                # target_temporal_audit_column intentionally NOT set.
                # Auto-detect from FTE.ts_field should kick in.
                enable_crash_reporting=False,
            ),
            verbose=0,
        )

    # Auto-detect log line must have fired.
    assert any(
        "auto-detected timestamp column 'ts'" in rec.message
        for rec in caplog.records
    ), "expected the FTE.ts_field auto-detect INFO log line"

    # And the audit must have written its result on metadata.
    assert "target_temporal_audit" in metadata
    by_target = metadata["target_temporal_audit"]
    assert any(
        "y" in inner for inner in by_target.values()
    ), f"expected 'y' as a target_name key inside {list(by_target.keys())}"


def test_suite_explicit_override_disables_via_empty_string(caplog, tmp_path):
    """Operator passes target_temporal_audit_column='' to disable
    even when FTE.ts_field is set. Auto-detect must NOT fire."""
    from mlframe.training import train_mlframe_models_suite, TrainingBehaviorConfig
    from mlframe.training.configs import ModelHyperparamsConfig

    df = _make_fixture(n=400, with_drift=True)
    fte = _FTEWithTsField(target_col="y", ts_field="ts")

    with caplog.at_level(logging.INFO, logger="mlframe.training.core"):
        train_mlframe_models_suite(
            df=df,
            target_name="autodetect_disable",
            model_name="autodetect_disable",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            hyperparams_config=ModelHyperparamsConfig(
                iterations=3,
                early_stopping_rounds=2,
                cb_kwargs={"task_type": "CPU", "verbose": 0},
            ),
            behavior_config=TrainingBehaviorConfig(
                target_temporal_audit_column="",  # explicit disable
                enable_crash_reporting=False,
            ),
            verbose=0,
        )

    # Auto-detect must NOT have fired.
    assert not any(
        "auto-detected timestamp column" in rec.message for rec in caplog.records
    ), "auto-detect should be suppressed by empty-string override"
