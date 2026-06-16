"""End-to-end: train_mlframe_models_suite stamps regression conformal intervals + coverage.

With ``calib_size>0`` the trainer now stamps the calib-slice POINT predictions (``entry.calib_preds``)
for regression, and finalize's ``_conformal_on_calib_slice`` builds split-conformal intervals + achieved
test coverage into ``metadata["conformal"]``. ``calib_size=0`` leaves the regression calib predict inert.
"""

from __future__ import annotations

import numpy as np
import pytest


def _regression_frame(seed: int = 17, n: int = 1600):
    import polars as pl

    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n).astype(np.float32)
    x1 = rng.normal(size=n).astype(np.float32)
    x2 = rng.normal(size=n).astype(np.float32)
    y = (2.0 * x0 + 1.0 * x1 - 0.7 * x2 + 0.3 * rng.normal(size=n)).astype(np.float32)
    return pl.DataFrame({"f0": x0, "f1": x1, "f2": x2, "target": y})


def _run_regression_suite(tmp_path, calib_size, seed=17):
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.configs import (
        PreprocessingBackendConfig,
        OutputConfig,
        TrainingBehaviorConfig,
        BaselineDiagnosticsConfig,
        DummyBaselinesConfig,
        ReportingConfig,
    )
    from mlframe.training._preprocessing_configs import TrainingSplitConfig
    from .shared import SimpleFeaturesAndTargetsExtractor

    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)
    return train_mlframe_models_suite(
        df=_regression_frame(seed=seed),
        target_name="conf_e2e",
        model_name=f"conf_e2e_cs{calib_size}",
        features_and_targets_extractor=fte,
        mlframe_models=["xgb"],
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        pipeline_config=PreprocessingBackendConfig(
            prefer_polarsds=False,
            categorical_encoding=None,
            scaler_name=None,
            imputer_strategy=None,
        ),
        split_config=TrainingSplitConfig(test_size=0.25, val_size=0.1, calib_size=calib_size, random_seed=seed),
        behavior_config=TrainingBehaviorConfig(prefer_gpu_configs=False),
        hyperparams_config={"iterations": 40, "xgb_kwargs": {"device": "cpu"}},
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(honest_estimator_diagnostics=False),
        enable_target_distribution_analyzer=False,
        output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
        verbose=0,
    )


def _first_entry(models):
    for _by_name in models.values():
        for _entries in _by_name.values():
            if isinstance(_entries, list) and _entries:
                return _entries[0]
    return None


def test_e2e_calib_size_stamps_regression_conformal_intervals(tmp_path):
    pytest.importorskip("xgboost")
    models, metadata = _run_regression_suite(tmp_path / "with_calib", calib_size=0.2)

    entry = _first_entry(models)
    assert entry is not None
    # Trainer stamped the regression calib-slice point predictions (the split-conformal residual source).
    assert getattr(entry, "calib_preds", None) is not None, "trainer did not stamp entry.calib_preds for regression"

    # Finalize stamped conformal intervals + achieved coverage into metadata.
    assert "conformal" in metadata, "finalize did not stamp metadata['conformal']"
    rep = next(iter(metadata["conformal"].values()))
    assert rep["method"] in ("split_conformal", "cv_plus")
    per_alpha = rep["per_alpha"]
    assert per_alpha, "no per-alpha coverage entries"
    a0 = next(iter(per_alpha))
    # Achieved coverage is in (0,1] and within a generous band of nominal on this clean signal.
    cov = per_alpha[a0]["achieved_coverage"]
    assert 0.6 <= cov <= 1.0, f"implausible coverage {cov} at alpha={a0}"


def test_e2e_calib_size_zero_no_conformal(tmp_path):
    pytest.importorskip("xgboost")
    models, metadata = _run_regression_suite(tmp_path, calib_size=0.0)
    entry = _first_entry(models)
    assert entry is not None
    assert getattr(entry, "calib_preds", None) is None, "calib_size=0 must add no regression calib predict"
    # Without a calib slice the split-conformal source is absent; metadata['conformal'] may still appear via the
    # OOF/CV+ fallback if OOF preds exist, but the calib-driven path must be inert.
    if "conformal" in metadata:
        for rep in metadata["conformal"].values():
            assert rep["method"] == "cv_plus", "calib_size=0 must not produce split_conformal"
