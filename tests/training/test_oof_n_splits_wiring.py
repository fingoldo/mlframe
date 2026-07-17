"""Live end-to-end coverage for ``TrainingBehaviorConfig.oof_n_splits`` (wave-2 wiring, 2026-07-12).

Prior to this session's wiring effort, ``oof_n_splits``/``oof_has_time``/``oof_random_seed`` were
UNREACHABLE from the public ``train_mlframe_models_suite`` entry point: the trainer already supported
K-fold OOF prediction stamping (``model.oof_preds``/``oof_probs``/``oof_target``) via
``_trainer_train_and_evaluate.py``'s ``oof_n_splits`` kwarg, but no config surface ever set it, so it
was internal-only. ``_build_suite_common_params_dict`` (``core/_phase_helpers.py``) now threads
``TrainingBehaviorConfig.oof_n_splits`` through ``common_params_dict["oof_n_splits"]``. Default ``0``
preserves the legacy no-OOF behaviour (real extra compute: a K-fold retrain per model), so this test
verifies BOTH the opt-in path fires and the default stays a genuine no-op.
"""

from __future__ import annotations

import numpy as np


def _regression_frame(seed=5, n=600):
    import polars as pl

    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n).astype(np.float32)
    x1 = rng.normal(size=n).astype(np.float32)
    y = (2 * x0 - x1 + 0.3 * rng.normal(size=n)).astype(np.float32)
    return pl.DataFrame({"f0": x0, "f1": x1, "target": y})


def _run_suite(tmp_path, oof_n_splits):
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

    return train_mlframe_models_suite(
        df=_regression_frame(),
        target_name="ce",
        model_name="ce_run",
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True),
        mlframe_models=["lgb"],
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        pipeline_config=PreprocessingBackendConfig(prefer_polarsds=False, categorical_encoding=None, scaler_name=None, imputer_strategy=None),
        split_config=TrainingSplitConfig(test_size=0.25, val_size=0.1),
        behavior_config=TrainingBehaviorConfig(prefer_gpu_configs=False, oof_n_splits=oof_n_splits),
        hyperparams_config={"iterations": 40},
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(honest_estimator_diagnostics=False),
        enable_target_distribution_analyzer=False,
        output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
        verbose=0,
    )


def _trained_entries(models):
    trained = [e for per_target in models.values() for entries in per_target.values() for e in entries]
    return [e[0] if isinstance(e, tuple) and e else e for e in trained]


def test_e2e_oof_n_splits_stamps_oof_preds_on_model(tmp_path):
    """``oof_n_splits=3`` must produce a real K-fold OOF array, row-aligned with the train target."""
    models, _metadata = _run_suite(tmp_path, oof_n_splits=3)
    trained = _trained_entries(models)
    assert trained, "no models trained"

    fitted = trained[0]
    oof_preds = getattr(fitted, "oof_preds", None)
    oof_target = getattr(fitted, "oof_target", None)
    assert oof_preds is not None, "oof_n_splits=3 did not stamp model.oof_preds"
    assert oof_target is not None, "oof_n_splits=3 did not stamp model.oof_target"
    assert np.asarray(oof_preds).shape[0] == np.asarray(oof_target).shape[0]
    # A real K-fold CV OOF array must vary across rows on a non-degenerate regression target --
    # a constant/degenerate array would indicate the OOF loop silently no-op'd rather than genuinely
    # holding out and refitting.
    assert np.nanstd(np.asarray(oof_preds, dtype=np.float64)) > 0.0


def test_default_oof_n_splits_leaves_oof_preds_unset(tmp_path):
    """Default oof_n_splits=0 must be a genuine no-op: no oof_preds attribute, no extra K-fold compute."""
    models, _metadata = _run_suite(tmp_path, oof_n_splits=0)
    trained = _trained_entries(models)
    assert trained, "no models trained"

    fitted = trained[0]
    assert getattr(fitted, "oof_preds", None) is None, "default oof_n_splits=0 unexpectedly stamped oof_preds"


def test_e2e_oof_n_splits_unlocks_diversity_recommendations(tmp_path):
    """The full oof_n_splits -> oof_preds/oof_target -> recommend_diversity_additions_in_leaderboard
    pipeline must actually populate metadata["diversity_recommendations"] with >=2 fitted regression
    models and oof_n_splits>=2 -- the end-to-end path both the OOF wiring fix (this file) and the
    oof_target mirroring fix (2026-07-13, see DEFAULTS_CHANGELOG.md) were required to unlock.
    """
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

    models, metadata = train_mlframe_models_suite(
        df=_regression_frame(),
        target_name="ce",
        model_name="ce_run",
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True),
        mlframe_models=["lgb", "xgb"],
        use_ordinary_models=True,
        # ens_models (what compute_diversity_recommendations reads) is only ever populated when
        # use_mlframe_ensembles=True (see _phase_train_one_target_body.py:187) -- required for this test.
        use_mlframe_ensembles=True,
        pipeline_config=PreprocessingBackendConfig(prefer_polarsds=False, categorical_encoding=None, scaler_name=None, imputer_strategy=None),
        split_config=TrainingSplitConfig(test_size=0.25, val_size=0.1),
        behavior_config=TrainingBehaviorConfig(prefer_gpu_configs=False, oof_n_splits=3, recommend_diversity_additions_in_leaderboard=True),
        hyperparams_config={"iterations": 40},
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(honest_estimator_diagnostics=False),
        enable_target_distribution_analyzer=False,
        output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
        verbose=0,
    )
    trained = _trained_entries(models)
    assert len(trained) >= 2, f"need >=2 fitted models to exercise diversity recommendations, got {len(trained)}"
    # Per-member oof_preds/oof_target stamping onto INDIVIDUAL "lgb"/"xgb" entries is already covered by
    # test_e2e_oof_n_splits_stamps_oof_preds_on_model above; with use_mlframe_ensembles=True, ``models``
    # also contains composite ensemble entries which don't carry oof_preds the same way, so re-asserting
    # it per-entry here would be redundant AND fragile. The real end-to-end claim this test makes is that
    # metadata["diversity_recommendations"] gets populated -- check that directly.
    div = (metadata or {}).get("diversity_recommendations")
    assert div is not None, 'metadata["diversity_recommendations"] was never stamped despite >=2 OOF-complete models'
    assert isinstance(div, dict) and div, "diversity_recommendations metadata is empty"
