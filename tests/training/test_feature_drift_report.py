"""Regression coverage for ``compute_feature_distribution_drift``.

Sensor lands 2026-05-22 to complement the existing
``label_distribution_drift`` -- catches the feature-side shift that broke
the TVT-2026-05-21 MLP path (Ridge tolerates 14-sigma TVT_prev drift fine
via linear extrapolation; MLP collapses).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.training.feature_drift_report import (
    DEFAULT_FEATURE_DRIFT_WARN_THRESHOLD_Z,
    compute_feature_distribution_drift,
)


def _make_frames(*, drift_z: float, n: int = 1000, seed: int = 0):
    """Build train/val/test pandas frames where ``f_shift`` has its test mean
    drifted by exactly ``drift_z`` train-stds. f_stable matches across splits."""
    rng = np.random.default_rng(seed)
    train = pd.DataFrame({
        "f_stable": rng.normal(0.0, 1.0, n),
        "f_shift": rng.normal(0.0, 1.0, n),
    })
    val = pd.DataFrame({
        "f_stable": rng.normal(0.0, 1.0, n // 2),
        "f_shift": rng.normal(0.0, 1.0, n // 2),
    })
    # Inject a deterministic mean shift into f_shift on the test slice. The
    # underlying noise std stays ~1 so train_std=1 and the z is exactly drift_z.
    test = pd.DataFrame({
        "f_stable": rng.normal(0.0, 1.0, n // 2),
        "f_shift": rng.normal(drift_z, 1.0, n // 2),
    })
    return train, val, test


class TestFeatureDriftSensor:
    def test_no_drift_clean_iid_splits(self):
        train, val, test = _make_frames(drift_z=0.0)
        rep = compute_feature_distribution_drift(train, val, test)
        assert rep["n_numeric_features"] == 2
        assert rep["drift_candidates"] == []

    def test_moderate_drift_logged_info_not_warn(self, caplog):
        """8-sigma drift -- moderate by absolute scale, NOT escalated to WARN
        because (a) drift does NOT prove harm and (b) per-model FS may drop
        the feature. INFO is the right level."""
        train, val, test = _make_frames(drift_z=8.0)
        with caplog.at_level(logging.INFO):
            rep = compute_feature_distribution_drift(train, val, test)
        cands = rep["drift_candidates"]
        names = [c for c, _z in cands]
        assert "f_shift" in names
        assert "f_stable" not in names
        # The log line surfaces at INFO level with the top-drifter list.
        info_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any("[feature-distribution-drift]" in m for m in info_msgs), (
            f"INFO log missing on moderate drift; got info_msgs={info_msgs}"
        )
        # And NO WARN should fire at this magnitude without FI weighting.
        warn_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("[feature-distribution-drift]" in m for m in warn_msgs), (
            f"WARN level fired on moderate drift without FI weighting; "
            f"design says only escalate at >=10x sigma OR weighted>=1.0. warn_msgs={warn_msgs}"
        )

    def test_extreme_drift_escalates_to_warn(self, caplog):
        """>= 10x threshold (so >= 30 sigma at default) escalates to WARN."""
        train, val, test = _make_frames(drift_z=35.0)
        with caplog.at_level(logging.WARNING):
            rep = compute_feature_distribution_drift(train, val, test)
        assert any(c == "f_shift" for c, _z in rep["drift_candidates"])
        msgs = " | ".join(rec.getMessage() for rec in caplog.records)
        assert "[feature-distribution-drift]" in msgs

    def test_fi_weighted_aggregate_grounds_harm_signal(self, caplog):
        """Per-feature z-score alone isn't a grounded harm signal -- a 5-sigma
        drift on an irrelevant feature is harmless. With FI weighting we get
        an aggregate that DOES correlate with model harm: high z * high FI
        = the important feature is drifting.

        Scenario: only ONE feature drifts strongly (8 sigma), and we tell the
        sensor that feature has FI=1.0 (dominant). The weighted score should
        be near the feature's z, escalating to WARN."""
        train, val, test = _make_frames(drift_z=8.0)
        with caplog.at_level(logging.WARNING):
            rep = compute_feature_distribution_drift(
                train, val, test,
                feature_importance={"f_shift": 1.0, "f_stable": 0.0},
            )
        ws = rep["weighted_drift_score"]
        assert ws is not None and ws > 5.0, (
            f"FI-weighted aggregate should be ~ z of the drifting dominant feature; got {ws}"
        )
        # And WARN fires because weighted_drift_score >= 1.0.
        msgs = " | ".join(rec.getMessage() for rec in caplog.records)
        assert "[feature-distribution-drift]" in msgs
        assert "weighted_drift=" in msgs

    def test_fi_weighted_aggregate_NOT_alarmed_when_drift_on_unimportant_feature(self, caplog):
        """Inverse scenario: f_shift drifts 8 sigma but its FI is 0; f_stable
        has FI=1 but no drift. Weighted score should be ~0 -- the system is
        safe even though one feature's z is high."""
        train, val, test = _make_frames(drift_z=8.0)
        with caplog.at_level(logging.WARNING):
            rep = compute_feature_distribution_drift(
                train, val, test,
                feature_importance={"f_shift": 0.0, "f_stable": 1.0},
            )
        ws = rep["weighted_drift_score"]
        assert ws is not None and ws < 0.5, (
            f"With drift on FI=0 feature, weighted score should stay low; got {ws}"
        )
        # No WARN should fire -- the harm signal is grounded and doesn't escalate.
        warn_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("[feature-distribution-drift]" in m for m in warn_msgs), (
            f"WARN should NOT fire when the drift is on an unimportant feature. "
            f"warn_msgs={warn_msgs}"
        )

    def test_threshold_respected(self):
        train, val, test = _make_frames(drift_z=2.5)
        # Default threshold is 3.0; 2.5-sigma drift must NOT fire.
        rep_default = compute_feature_distribution_drift(train, val, test)
        assert rep_default["drift_candidates"] == []
        # Tighter threshold (2.0) should catch it.
        rep_tight = compute_feature_distribution_drift(
            train, val, test, warn_threshold_z=2.0,
        )
        assert any(c == "f_shift" for c, _z in rep_tight["drift_candidates"])

    def test_constant_feature_skipped_via_nan_z(self):
        """A feature with zero train-std produces NaN z (no drift signal can
        be computed). The sensor must NOT crash and the feature must NOT be
        flagged as a drift candidate."""
        n = 500
        rng = np.random.default_rng(1)
        train = pd.DataFrame({"const": np.full(n, 5.0), "f": rng.normal(0, 1, n)})
        val = pd.DataFrame({"const": np.full(n // 2, 5.0), "f": rng.normal(0, 1, n // 2)})
        test = pd.DataFrame({"const": np.full(n // 2, 5.0), "f": rng.normal(0, 1, n // 2)})
        rep = compute_feature_distribution_drift(train, val, test)
        const_entry = rep["per_feature"]["const"]
        assert const_entry["train_std"] == 0.0
        assert np.isnan(const_entry["val_z"])
        assert np.isnan(const_entry["test_z"])
        assert not any(c == "const" for c, _z in rep["drift_candidates"])

    def test_polars_input_handled(self):
        pl = pytest.importorskip("polars")
        n = 500
        rng = np.random.default_rng(2)
        train = pl.DataFrame({"f": rng.normal(0, 1, n).astype(np.float32)})
        val = pl.DataFrame({"f": rng.normal(0, 1, n // 2).astype(np.float32)})
        test = pl.DataFrame({"f": rng.normal(5, 1, n // 2).astype(np.float32)})  # 5-sigma drift
        rep = compute_feature_distribution_drift(train, val, test)
        assert any(c == "f" for c, _z in rep["drift_candidates"]), (
            f"Polars frame with clear drift not flagged: {rep}"
        )

    def test_no_val_frame_falls_back_to_test_only(self):
        train, _, test = _make_frames(drift_z=6.0)
        rep = compute_feature_distribution_drift(train, val_df=None, test_df=test)
        # val_z is NaN, test_z is high; f_shift still in candidates.
        per = rep["per_feature"]["f_shift"]
        assert np.isnan(per["val_z"])
        assert abs(per["test_z"]) > 5.0
        assert any(c == "f_shift" for c, _z in rep["drift_candidates"])
