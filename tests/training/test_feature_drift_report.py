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

    def test_strong_drift_flagged_and_logged(self, caplog):
        train, val, test = _make_frames(drift_z=8.0)
        with caplog.at_level(logging.WARNING):
            rep = compute_feature_distribution_drift(train, val, test)
        cands = rep["drift_candidates"]
        # f_shift must be in the candidates; f_stable must not.
        names = [c for c, _z in cands]
        assert "f_shift" in names
        assert "f_stable" not in names
        # WARN log line surfaces with the top-drifter list.
        msgs = " | ".join(rec.getMessage() for rec in caplog.records)
        assert "[feature-distribution-drift]" in msgs
        assert "f_shift" in msgs

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
