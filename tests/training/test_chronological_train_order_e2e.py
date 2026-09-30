"""End-to-end: a shuffled-time frame through ``train_mlframe_models_suite`` gets chronological train rows, CatBoost ``has_time``, temporal OOF and stays row-aligned."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytest.importorskip("catboost")

N = 600


def _frame(seed=7):
    """Polars frame of shuffled hourly rows with a target linear in x0 and the row id."""
    rng = np.random.default_rng(seed)
    ts = np.datetime64("2024-01-01T00:00:00", "ns") + (np.arange(N) * 3_600_000_000_000).astype("timedelta64[ns]")
    ts = ts[rng.permutation(N)]  # rows are NOT chronological
    x0 = rng.normal(size=N)
    df = pl.DataFrame({"rid": np.arange(N), "ts": ts, "x0": x0, "target": 3.0 * x0 + 0.01 * np.arange(N)})
    return df


def _run(tmp_path, order):
    """Train a suite on the frame with chronological train ordering on or off."""
    from mlframe.training.configs import BaselineDiagnosticsConfig, DummyBaselinesConfig, OutputConfig, PreprocessingBackendConfig, ReportingConfig, TrainingBehaviorConfig
    from mlframe.training._preprocessing_configs import TrainingSplitConfig
    from mlframe.training.core import train_mlframe_models_suite
    from .shared import SimpleFeaturesAndTargetsExtractor

    return train_mlframe_models_suite(
        df=_frame(), target_name="chron", model_name="chron_run",
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True, ts_field="ts"),
        mlframe_models=["cb"], use_ordinary_models=True, use_mlframe_ensembles=False,
        pipeline_config=PreprocessingBackendConfig(prefer_polarsds=False, categorical_encoding=None, scaler_name=None, imputer_strategy=None),
        split_config=TrainingSplitConfig(test_size=0.2, val_size=0.1, chronological_train_order=order),
        behavior_config=TrainingBehaviorConfig(prefer_gpu_configs=False, oof_n_splits=3),
        hyperparams_config={"iterations": 15, "cb_kwargs": {"thread_count": 1}},
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False), dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(honest_estimator_diagnostics=False), enable_target_distribution_analyzer=False,
        output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"), verbose=0,
    )


def _entries(models):
    """Flatten trained model entries across targets into plain model results."""
    trained = [e for per_target in models.values() for entries in per_target.values() for e in entries]
    return [e[0] if isinstance(e, tuple) and e else e for e in trained]


def test_shuffled_time_frame_is_chronological_aligned_and_has_time(tmp_path):
    """Shuffled time frame is chronological aligned and has time."""
    models, md = _run(tmp_path, True)
    assert md["train_chronological_order"] == "reordered"
    fitted = _entries(models)[0]
    inner = getattr(fitted, "model", fitted)
    assert inner.get_params()["has_time"] is True
    full = _frame().to_pandas()
    ts = full["ts"].to_numpy()
    y_by_rid = (3.0 * full["x0"] + 0.01 * full["rid"]).to_numpy()  # unique per row -> recovers each train row's id from its target
    rid_of = {round(float(v), 6): i for i, v in enumerate(y_by_rid)}
    train_target = np.asarray(fitted.train_target, dtype=float)
    rids = np.array([rid_of[round(float(v), 6)] for v in train_target])
    assert np.all(np.diff(ts[rids].astype(np.int64)) >= 0), "train rows are not chronological"
    # OOF targets / predictions are row-aligned with the (reordered) train rows
    np.testing.assert_allclose(np.asarray(fitted.oof_target, dtype=float), train_target, rtol=1e-6)
    oof = np.asarray(fitted.oof_preds, dtype=float)
    ok = np.isfinite(oof)
    assert np.isnan(oof[0]) and ok[-1], "temporal OOF leaves the oldest block unpredicted and scores the newest rows"
    assert np.corrcoef(oof[ok], train_target[ok])[0, 1] > 0.5
    assert np.corrcoef(np.asarray(fitted.test_preds, dtype=float), np.asarray(fitted.test_target, dtype=float))[0, 1] > 0.5


def test_opt_out_keeps_row_order_and_has_time_off(tmp_path):
    """Opt out keeps row order and has time off."""
    models, md = _run(tmp_path, False)
    assert "train_chronological_order" not in md
    fitted = _entries(models)[0]
    inner = getattr(fitted, "model", fitted)
    assert not inner.get_params().get("has_time")
