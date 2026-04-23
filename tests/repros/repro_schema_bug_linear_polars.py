"""Reproduce _rule_linear_polars_gating_bug (c0011) with full traceback."""
import sys
sys.path.insert(0, r"D:/Upd/Programming/PythonCodeRepository/mlframe")
sys.path.insert(0, r"D:/Upd/Programming/PythonCodeRepository/mlframe/tests/training")

from _fuzz_combo import FuzzCombo, build_frame_for_combo
from shared import SimpleFeaturesAndTargetsExtractor

combo = FuzzCombo(
    models=("linear", "xgb"),
    input_type="polars_nullable", n_rows=300, cat_feature_count=3,
    null_fraction_cats=0.0, use_mrmr_fs=False,
    weight_schemas=("uniform",), target_type="binary_classification",
    auto_detect_cats=True, align_polars_categorical_dicts=False, seed=11,
)
df, target_col, _ = build_frame_for_combo(combo)
fte = SimpleFeaturesAndTargetsExtractor(target_column=target_col, regression=False)

import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from mlframe.training.core import train_mlframe_models_suite

try:
    trained, _ = train_mlframe_models_suite(
        df=df, target_name="repro", model_name="repro",
        features_and_targets_extractor=fte,
        mlframe_models=["linear", "xgb"],
        hyperparams_config={"iterations": 3, "xgb_kwargs": {"device": "cpu", "verbosity": 0}},
        init_common_params={
            "drop_columns": [], "verbose": 0,
            "category_encoder": ce.CatBoostEncoder(),
            "scaler": StandardScaler(),
            "imputer": SimpleImputer(strategy="mean"),
        },
        use_ordinary_models=True, use_mlframe_ensembles=False,
        data_dir=r"D:/Temp/repro_models", models_dir="models", verbose=0,
    )
    print("PASS")
except Exception as e:
    import traceback
    print(f"FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()
