"""Regression: an mlp-only suite run with inf/nan injected must TRAIN, not crash.

The fuzz combo enumerator previously canonicalised ``inject_inf_nan=True`` -> ``False`` for the mlp-only model subset,
claiming the value always hit the fit-entry ``_validate_no_nan_inf`` raise. That canon was a prohibited
"collapse-because-it-crashes" rule (CLAUDE.md). In fact the suite's default SimpleImputer + inf_to_nan step clean the
frame before the MLP fit, so the combo trains fine. This test pins that behaviour so the canon stays retired.
"""

import pytest

from tests.training._fuzz_combo import FuzzCombo, build_frame_for_combo
from tests.training._fuzz_suite_helpers import _config_for_models, _preprocessing_for_combo
from tests.training.shared import SimpleFeaturesAndTargetsExtractor


@pytest.mark.parametrize("inject_inf_nan", [True, False])
def test_mlp_only_trains_with_inject_inf_nan(inject_inf_nan):
    combo = FuzzCombo(
        models=("mlp",),
        input_type="pandas",
        n_rows=1000,
        cat_feature_count=2,
        null_fraction_cats=0.0,
        use_mrmr_fs=False,
        weight_schemas=("none",),
        target_type="binary_classification",
        auto_detect_cats=True,
        align_polars_categorical_dicts=False,
        seed=20260612,
        iterations=3,
        inject_inf_nan=inject_inf_nan,
    )
    df, target_col, _cat_names = build_frame_for_combo(combo)
    fte = SimpleFeaturesAndTargetsExtractor(target_column=target_col)

    from mlframe.training.core import train_mlframe_models_suite

    trained, _meta = train_mlframe_models_suite(
        df=df,
        target_name=combo.short_id(),
        model_name=combo.short_id(),
        features_and_targets_extractor=fte,
        mlframe_models=list(combo.models),
        hyperparams_config=_config_for_models(combo.models, combo.n_rows, iterations=combo.iterations),
        preprocessing_config=_preprocessing_for_combo(combo),
    )
    # The MLP leg must have trained -- a NaN/inf crash would raise above and never reach here.
    assert trained is not None
