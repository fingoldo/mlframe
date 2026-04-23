"""Verify whether _rule_mrmr_plus_xgb_lgb_polars_utf8_small (c0098) is still live.

Combo: seed=98, mrmr=True, xgb+lgb, polars_utf8, n=300, ncats=1.
"""
import sys
sys.path.insert(0, r"D:/Upd/Programming/PythonCodeRepository/mlframe")
sys.path.insert(0, r"D:/Upd/Programming/PythonCodeRepository/mlframe/tests/training")

from _fuzz_combo import FuzzCombo, build_frame_for_combo
from shared import SimpleFeaturesAndTargetsExtractor

combo = FuzzCombo(
    models=("lgb", "xgb"), input_type="polars_utf8", n_rows=300,
    cat_feature_count=1, null_fraction_cats=0.0, use_mrmr_fs=True,
    weight_schemas=("uniform",), target_type="binary_classification",
    auto_detect_cats=True, align_polars_categorical_dicts=False, seed=98,
)
df, target_col, _ = build_frame_for_combo(combo)
fte = SimpleFeaturesAndTargetsExtractor(target_column=target_col, regression=False)

from mlframe.training.core import train_mlframe_models_suite

try:
    trained, _ = train_mlframe_models_suite(
        df=df, target_name="repro", model_name="repro",
        features_and_targets_extractor=fte,
        mlframe_models=["lgb", "xgb"],
        hyperparams_config={
            "iterations": 3,
            "xgb_kwargs": {"device": "cpu", "verbosity": 0},
            "lgb_kwargs": {"device_type": "cpu", "verbose": -1},
        },
        init_common_params={"drop_columns": [], "verbose": 0},
        use_ordinary_models=True, use_mlframe_ensembles=False,
        data_dir=r"D:/Temp/repro_mrmr_xgb_lgb_models", models_dir="models", verbose=0,
        use_mrmr_fs=True,
        mrmr_kwargs={
            "verbose": 0, "max_runtime_mins": 1, "n_workers": 1,
            "quantization_nbins": 5, "use_simple_mode": True,
            "min_nonzero_confidence": 0.9, "max_consec_unconfirmed": 3,
            "full_npermutations": 3,
        },
    )
    print(f"PASS — trained {len(trained) if trained else 0} target_type(s)")
except Exception as e:
    import traceback
    print(f"FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()
