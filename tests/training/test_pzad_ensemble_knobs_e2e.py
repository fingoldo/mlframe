"""End-to-end: the PZAD ensemble blend knobs (Caruana weights + rank_average flavour) flow from
TrainingBehaviorConfig through train_mlframe_models_suite into the ensembling phase. Pins the production wiring the
new fuzz axes (use_caruana_weights_in_ensemble_cfg / ens_rank_average_cfg) depend on."""

from __future__ import annotations

import time

import pytest

from tests.training._fuzz_combo.combo import FuzzCombo
from tests.training._fuzz_suite_helpers import _configs_for_combo


def _combo(caruana: bool, rank_average: bool) -> FuzzCombo:
    # 2 tree models + binary target + ensembling on -> the simple-ensemble phase fires and reads the blend knobs.
    return FuzzCombo(
        models=("cb", "xgb"),
        input_type="pandas",
        n_rows=240,
        cat_feature_count=0,
        null_fraction_cats=0.0,
        use_mrmr_fs=False,
        weight_schemas=("uniform",),
        target_type="binary_classification",
        target_carrier="native",
        auto_detect_cats=True,
        align_polars_categorical_dicts=False,
        seed=7,
        use_ensembles=True,
        use_caruana_weights_in_ensemble_cfg=caruana,
        ens_rank_average_cfg=rank_average,
    )


def _run(combo, tmp_path):
    pytest.importorskip("catboost")
    pytest.importorskip("xgboost")
    from mlframe.training import FeatureSelectionConfig, OutlierDetectionConfig, OutputConfig
    from mlframe.training.core import train_mlframe_models_suite
    from tests.training.shared import SimpleFeaturesAndTargetsExtractor
    from tests.training._fuzz_combo.frame_builder import build_frame_for_combo
    from tests.training._fuzz_suite_helpers import _config_for_models, _preprocessing_for_combo

    df, target_col, _ = build_frame_for_combo(combo)
    fte = SimpleFeaturesAndTargetsExtractor(target_column=target_col, regression=False)
    trained, meta = train_mlframe_models_suite(
        df=df,
        target_name=combo.short_id(),
        model_name=combo.short_id(),
        features_and_targets_extractor=fte,
        mlframe_models=list(combo.models),
        hyperparams_config=_config_for_models(combo.models, combo.n_rows, iterations=combo.iterations, early_stopping_rounds=combo.early_stopping_rounds_cfg),
        preprocessing_config=_preprocessing_for_combo(combo),
        verbose=0,
        use_ordinary_models=True,
        use_mlframe_ensembles=True,
        outlier_detection_config=OutlierDetectionConfig(detector=None),
        output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
        feature_selection_config=FeatureSelectionConfig(use_mrmr_fs=False),
        **_configs_for_combo(combo),
    )
    return trained, meta


def test_pzad_ensemble_knobs_run_end_to_end(tmp_path):
    """Suite runs with use_caruana_weights_in_ensemble + extra_ensembling_methods=('rank_average',) and produces a
    non-empty model dict (the new behavior_config knobs are consumed by the ensembling phase without crashing), and a
    rank_average-flavoured ensemble is stamped."""
    trained, meta = _run(_combo(caruana=True, rank_average=True), tmp_path)
    assert trained, "suite produced no models with the PZAD ensemble knobs on"
    # rank_average must appear among the built ensemble flavours (names carry the flavour token).
    names = []
    for _tt, per_name in trained.items() if isinstance(trained, dict) else []:
        if isinstance(per_name, dict):
            for _nm, entries in per_name.items():
                for e in entries if isinstance(entries, (list, tuple)) else [entries]:
                    nm = getattr(getattr(e, "model", e), "__mlframe_name__", "") or str(getattr(e, "name", "")) or repr(e)
                    names.append(nm.lower())
    assert any("rank_average" in n for n in names), f"no rank_average ensemble stamped; saw {names[:8]}"


def test_pzad_ensemble_knobs_off_is_still_fine(tmp_path):
    """Baseline: knobs OFF still runs (no regression from the added wiring)."""
    trained, _ = _run(_combo(caruana=False, rank_average=False), tmp_path)
    assert trained
