"""Reproduce _rule_mrmr_plus_linear_multi_pandas — feature names mismatch."""
import sys
sys.path.insert(0, r"D:/Upd/Programming/PythonCodeRepository/mlframe")
sys.path.insert(0, r"D:/Upd/Programming/PythonCodeRepository/mlframe/tests/training")

from _fuzz_combo import FuzzCombo, build_frame_for_combo
from shared import SimpleFeaturesAndTargetsExtractor

combo = FuzzCombo(
    models=("cb", "hgb", "lgb", "linear", "xgb"),
    input_type="pandas", n_rows=300, cat_feature_count=0,
    null_fraction_cats=0.0, use_mrmr_fs=True,
    weight_schemas=("uniform",), target_type="binary_classification",
    auto_detect_cats=False, align_polars_categorical_dicts=False, seed=31,
)
df, target_col, _ = build_frame_for_combo(combo)
fte = SimpleFeaturesAndTargetsExtractor(target_column=target_col, regression=False)

from mlframe.training.core import train_mlframe_models_suite

try:
    trained, _ = train_mlframe_models_suite(
        df=df, target_name="repro", model_name="repro",
        features_and_targets_extractor=fte,
        mlframe_models=["cb", "hgb", "lgb", "linear", "xgb"],
        hyperparams_config={
            "iterations": 3,
            "cb_kwargs": {"task_type": "CPU", "verbose": 0},
            "xgb_kwargs": {"device": "cpu", "verbosity": 0},
            "lgb_kwargs": {"device_type": "cpu", "verbose": -1},
        },
        init_common_params={"drop_columns": [], "verbose": 0},
        use_ordinary_models=True, use_mlframe_ensembles=False,
        data_dir=r"D:/Temp/repro_fnmismatch_models", models_dir="models", verbose=0,
        use_mrmr_fs=True,
        mrmr_kwargs={
            "verbose": 0, "max_runtime_mins": 1, "n_workers": 1,
            "quantization_nbins": 5, "use_simple_mode": True,
            "min_nonzero_confidence": 0.9, "max_consec_unconfirmed": 3,
            "full_npermutations": 3,
        },
    )
    print("PASS")
except Exception as e:
    import traceback
    print(f"FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()
